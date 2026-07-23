"""
Kopterix Phase 3, Task 2D -- periodicity check on the residual correlation floor.

Follow-up to Task 2C. Task 2C found that per-month centering and linear detrending each
roughly halve the grand-mean-centered correlation floor at lags 12-16 (~0.185 ->
~0.067-0.092 across variants), but every variant's floor 95% CI excludes zero -- about
half of the grand-mean floor is the cross-month drift artifact, the rest unexplained.

This task tests the cheap remaining mundane explanation: weekly or diurnal periodicity.
At dt = 5.7405h, weekly recurrence (168h) falls near lag 168/5.7405 = 29.26, i.e. inside
lags 26-32; diurnal recurrence (24h) falls near lag 24/5.7405 = 4.18, i.e. period ~4.2
lags (imperfect because dt does not evenly divide 24h).

Conventions inherited unchanged from Task 1/2/2B/2C:
  - Window [2026-04-01, 2026-06-01) UTC, half-open.
  - Seed 42.
  - kopterix_state.csv: Timestamp (ISO+tz), MeanEmbedding (384-d), LayerCentroids
    (dict surface/mid/residue, each 384-d).
  - n=210 valid layer-centroid rows (104 April + 106 May), dt = median consecutive gap
    = 5.7405h, original-adjacent pairing only (no bridging gaps > 12h).
  - Cosine similarity = dot(a,b)/(|a||b|); pair excluded if either norm < 1e-12.
  - Raw inputs are READ-ONLY; all outputs are new files in phase3/.

Variants extended to lags 1-56 (~13.4 days at dt=5.7405h):
  - per_month_incl (PRIMARY): per-coordinate centroid minus its own month's mean
    centroid (Task 2C "per_month_incl"; ACF over all original-adjacent valid pairs,
    incl. the few straddling the April/May boundary). Reused unchanged from Task 2C.
  - detrended (CHECK): per-coordinate OLS linear-in-time-detrended centroid (Task 2C
    "detrended"). Reused unchanged from Task 2C.
  - phi_t_detrended (COMPARISON, aggregate scalar): linearly detrended phi_t on its
    regular 6h grid (phase3/logs/C_series.npz, from Task 2 Analysis C). Cosine
    similarity is undefined for a 1-d scalar, so we use the standardized-product
    analog z(t)*z(t+k) (z = (x - mean)/std over the full detrended series); its
    lag-k mean is the standard (biased) autocovariance-based ACF estimator, and it
    plugs into the same sim_curve()/bootstrap machinery as the cosine-similarity
    matrices (S = outer(z, z)).

COMPUTATION
  1. Extend the ACF from lag 16 to lag 56, same original-adjacent pairing rules.
     Report n_pairs per lag; flag lags with n_pairs < 80.
  2. Diurnal bins: for all valid pairs at lags 1-56, bin by
     (hour-of-day(t+k) - hour-of-day(t)) mod 24 into four 6h bins; report mean
     similarity + n_pairs per bin. Diurnal cycle => "0-6h" bin > "12-18h" bin.
  3. Weekly bins: same pairs binned by (dow(t+k) - dow(t)) mod 7 (7 bins); report mean
     similarity + n_pairs per bin. Weekly cycle => "0d" bin > "3d"/"4d" bins, and/or an
     ACF rise near lag 29.
  4. phi_t_detrended: same lag-56 extension + diurnal/weekly bins, as a comparison
     series (floor expected near zero if the layer-geometry floor is layer-specific).
  5. Bootstrap: adjacency-preserving moving-block bootstrap, fixed seeds (42+offset).
     A block of length Lb can only contain pairs up to lag Lb-1, so block=24 (used by
     Task 2C for the lag 12-16 floor CI) cannot reach lag 56. We use a SINGLE block=64
     bootstrap (smallest multiple of 8 exceeding 56), 1000 resamples, for the entire
     lags 1-56 ACF-CI table plus floor_lag12_16 (cross-ref vs Task 2C), floor_mid
     (lags 19-25), and floor_far (lags 26-32). block=64 is valid for every lag in
     1-56 (Lb-1=63 >= 56), so it supersedes Task 2C's block=8 for the short lags too;
     block=8 is not separately recomputed here.
  6. Decision metric per layer/variant: floor_mid = mean ACF lags 19-25 (between-band),
     floor_far = mean ACF lags 26-32 (weekly recurrence band). Compared against Task
     2C's floor_lag12_16 (per_month_incl / detrended), recomputed here for the same
     point estimate as a sanity check.

OUTPUTS
  - phase3/tables/phase3_floor_periodicity.csv
  - phase3/figures/phase3_task2d_longlag_acf.png
  - phase3/TASK2D_SUMMARY.md (written separately)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parents[1]
TABLES  = BASE / "phase3" / "tables"
FIGURES = BASE / "phase3" / "figures"
LOGS    = BASE / "phase3" / "logs"
for d in (TABLES, FIGURES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
SEED   = 42
LAYERS = ["surface", "mid", "residue"]
BIG    = 12.0     # hours; do not bridge gaps larger than this (same as Task 2/2B/2C)
MAXK   = 56       # extended from Task 2C's 16
NBOOT  = 1000
LB_LONG = 64      # smallest multiple of 8 exceeding MAXK=56 (Lb-1=63 >= 56)
FLOOR_LAGS = list(range(12, 17))  # lags 12-16 (Task 2C floor, recomputed for cross-ref)
MID_LAGS   = list(range(19, 26))  # lags 19-25 (between-band)
FAR_LAGS   = list(range(26, 33))  # lags 26-32 (weekly recurrence band, ~168h/dt=29.26)
N_PAIRS_FLAG_THRESH = 80

import time
_T0 = time.time()
log_lines = []
def log(msg): print(f"[{time.time()-_T0:6.2f}s] {msg}", flush=True); log_lines.append(str(msg))

log("=== Kopterix Phase 3 Task 2D: periodicity check on the residual correlation floor ===")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD layer centroids (same parsing as Task 2/2B/2C)
# ─────────────────────────────────────────────────────────────────────────────
state = pd.read_csv(BASE / "kopterix_state.csv")
state["ts"] = pd.to_datetime(state["Timestamp"], utc=True)
state = state.sort_values("ts").reset_index(drop=True)

def parse_vec(s):
    if pd.isna(s): return None
    try:
        v = np.asarray(json.loads(s), dtype=float)
        return v if (v.size == 384 and np.all(np.isfinite(v))) else None
    except Exception:
        return None

def parse_layers(s):
    if pd.isna(s): return None
    try:
        d = json.loads(s)
        out = {}
        for l in LAYERS:
            v = d.get(l)
            if v is None: return None
            a = np.asarray(v, dtype=float)
            if a.size != 384 or not np.all(np.isfinite(a)) or np.allclose(a, 0):
                return None
            out[l] = a
        return out
    except Exception:
        return None

state["mean_emb"] = state["MeanEmbedding"].apply(parse_vec)
state["lc"]       = state["LayerCentroids"].apply(parse_layers)

in_win = (state["ts"] >= WINDOW_START) & (state["ts"] < WINDOW_END)
sw = state[in_win].copy().reset_index(drop=True)

me_valid = sw[sw["mean_emb"].notna()].copy()
grand_mean = np.stack(me_valid["mean_emb"].values).mean(axis=0)
log(f"Grand mean: n={len(me_valid)} (expect 215), L2 norm={np.linalg.norm(grand_mean):.10f}")
assert len(me_valid) == 215
assert abs(np.linalg.norm(grand_mean) - 0.3191144627010268) < 1e-9

lc_ok = sw["lc"].notna() & sw["mean_emb"].notna()
slc = sw[lc_ok].copy().sort_values("ts").reset_index(drop=True)
n = len(slc)
log(f"Valid layer-centroid rows: {n} (expect 210)")
assert n == 210

gaps = slc["ts"].diff().dt.total_seconds().values / 3600.0  # hours, gaps[0]=nan
DT = float(np.nanmedian(gaps[1:]))
log(f"dt (median consecutive gap) = {DT:.6f} h; lag56 ~ {56*DT:.1f} h ~ {56*DT/24:.2f} days; "
    f"weekly (168h) ~ lag {168/DT:.2f}; diurnal (24h) ~ lag {24/DT:.2f}")

raw_vecs = {l: np.array([slc.iloc[i]["lc"][l] for i in range(n)]) for l in LAYERS}

months = slc["ts"].dt.month.values
n_apr, n_may = int((months == 4).sum()), int((months == 5).sum())
log(f"Month split: April={n_apr}, May={n_may} (expect 104 + 106 = 210)")
assert n_apr == 104 and n_may == 106

hours_frac = (slc["ts"].dt.hour + slc["ts"].dt.minute/60.0 + slc["ts"].dt.second/3600.0).values
dow = slc["ts"].dt.dayofweek.values  # Monday=0..Sunday=6

# ─────────────────────────────────────────────────────────────────────────────
# Adjacency mask: OK[i,k] = True iff rows i..i+k are gap-valid (no NaN gap, none > BIG)
# extended to maxk=56 (Task 2C used maxk=16)
# ─────────────────────────────────────────────────────────────────────────────
def build_ok(n, gaps, big=BIG, maxk=MAXK):
    ok = np.zeros((n, maxk + 1), bool)
    for i in range(n):
        for k in range(1, maxk + 1):
            if i + k >= n: break
            seg = gaps[i+1:i+k+1]
            ok[i, k] = (not np.any(np.isnan(seg))) and (not np.any(seg > big))
    return ok
OK = build_ok(n, gaps)

n_pairs_lag = OK.sum(axis=0)  # n_pairs_lag[k] for k=1..56
log(f"n_pairs at lag 16 (Task 2C cross-ref): {int(n_pairs_lag[16])} (Task 2C reported 153)")
log(f"n_pairs at lags 26-56 (every 5th): "
    f"{[(k, int(n_pairs_lag[k])) for k in range(26, 57, 5)]}")

# ─────────────────────────────────────────────────────────────────────────────
# Build the two layer-centroid variants (reused unchanged from Task 2C)
# ─────────────────────────────────────────────────────────────────────────────
apr_mask = (months == 4)
may_mask = (months == 5)
pm_vecs = {}
for l in LAYERS:
    mean_apr = raw_vecs[l][apr_mask].mean(axis=0)
    mean_may = raw_vecs[l][may_mask].mean(axis=0)
    v = raw_vecs[l].copy()
    v[apr_mask] -= mean_apr[None, :]
    v[may_mask] -= mean_may[None, :]
    pm_vecs[l] = v

t_hours = (slc["ts"] - slc["ts"].iloc[0]).dt.total_seconds().values / 3600.0
dt_vecs = {}
for l in LAYERS:
    Y = raw_vecs[l]
    coeffs = np.polyfit(t_hours, Y, 1)
    pred = coeffs[0][None, :] * t_hours[:, None] + coeffs[1][None, :]
    dt_vecs[l] = Y - pred

VARIANTS = [
    ("per_month_incl", pm_vecs),
    ("detrended",      dt_vecs),
]

# ─────────────────────────────────────────────────────────────────────────────
# Generic similarity-curve + bootstrap machinery (reused/extended from Task 2C)
# ─────────────────────────────────────────────────────────────────────────────
def Smat_cosine(vecs):
    Vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return Vn @ Vn.T

def sim_curve(S, n, OKm, maxk=MAXK):
    """Point-estimate mean similarity per lag k=1..maxk over original-adjacent valid
    pairs (full data)."""
    num = np.zeros(maxk + 1); den = np.zeros(maxk + 1)
    for i in range(n):
        for k in range(1, maxk + 1):
            if i + k < n and OKm[i, k]:
                num[k] += S[i, i+k]; den[k] += 1
    ks = np.array([k for k in range(1, maxk + 1) if den[k] > 0])
    ys = np.array([num[k] / den[k] for k in ks])
    dn = np.array([den[k] for k in ks], dtype=int)
    return ks, ys, dn

def precompute_lag_arrays(S, OKm, n, maxk=MAXK):
    """For each lag k=1..maxk, cumulative sums (length n+1) of the per-i contributions
    arr_sum[i]=S[i,i+k] (if OKm[i,k] else 0) and arr_cnt[i]=1 (if OKm[i,k] else 0), both
    implicitly 0 for i>n-1-k. csum[k][hi]-csum[k][lo] then gives, in O(1), the sum over
    i in [lo,hi) -- used by the block bootstrap to avoid recomputing per-pair sums for
    every one of the 1000 resamples."""
    csum, ccnt = {}, {}
    for k in range(1, maxk + 1):
        a_sum = np.zeros(n); a_cnt = np.zeros(n)
        valid_i = np.arange(0, n - k)
        ok_k = OKm[valid_i, k]
        iv = valid_i[ok_k]
        a_sum[iv] = S[iv, iv + k]
        a_cnt[iv] = 1.0
        csum[k] = np.concatenate([[0.0], np.cumsum(a_sum)])
        ccnt[k] = np.concatenate([[0.0], np.cumsum(a_cnt)])
    return csum, ccnt

def mean_over_lags(ks, ys, lag_set):
    idx = [i for i, k in enumerate(ks) if k in lag_set]
    if not idx: return np.nan, 0
    return float(np.mean(ys[idx])), len(idx)

def _ci(a):
    a = np.array(a)
    a = a[~np.isnan(a)]
    return (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) if len(a) > 10 else (np.nan, np.nan)

def boot_long(S, OKm, n, seed, Lb=LB_LONG, n_boot=NBOOT, maxk=MAXK):
    """Single block=Lb adjacency-preserving moving-block bootstrap. Returns, for each
    lag k=1..maxk, the bootstrap distribution of the lag-k mean similarity, plus the
    distributions of floor_lag12_16 / floor_mid / floor_far (means over their lag sets,
    per resample). Block range-sums are looked up in O(1) via precompute_lag_arrays --
    numerically identical to recomputing per-pair sums each resample, just fast enough
    for NBOOT=1000."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / Lb))
    csum, ccnt = precompute_lag_arrays(S, OKm, n, maxk)
    kmax = min(maxk, Lb - 1)  # = maxk = 56 for Lb=64
    acf_samples = {k: [] for k in range(1, maxk + 1)}
    floor_samples, mid_samples, far_samples = [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - Lb, size=nb)
        sums = np.zeros(kmax + 1); cnts = np.zeros(kmax + 1)
        for s0 in starts:
            s0 = int(s0)
            for k in range(1, kmax + 1):
                hi = s0 + Lb - k
                sums[k] += csum[k][hi] - csum[k][s0]
                cnts[k] += ccnt[k][hi] - ccnt[k][s0]
        ks = np.array([k for k in range(1, kmax + 1) if cnts[k] > 0])
        if ks.size == 0: continue
        ys = np.array([sums[k] / cnts[k] for k in ks])
        for k, y in zip(ks, ys):
            acf_samples[int(k)].append(y)
        f, nf = mean_over_lags(ks, ys, FLOOR_LAGS)
        if nf > 0: floor_samples.append(f)
        m, nm = mean_over_lags(ks, ys, MID_LAGS)
        if nm > 0: mid_samples.append(m)
        fa, nfa = mean_over_lags(ks, ys, FAR_LAGS)
        if nfa > 0: far_samples.append(fa)
    return acf_samples, floor_samples, mid_samples, far_samples

log(f"\n[DONE: setup] n={n}, dt={DT:.4f}h, MAXK={MAXK}, NBOOT={NBOOT}, LB_LONG={LB_LONG}, "
    f"FLOOR_LAGS={FLOOR_LAGS}, MID_LAGS={MID_LAGS}, FAR_LAGS={FAR_LAGS}")

# ─────────────────────────────────────────────────────────────────────────────
# phi_t_detrended comparison series (Task 2 Analysis C, regular 6h grid)
# ─────────────────────────────────────────────────────────────────────────────
C = np.load(LOGS / "C_series.npz", allow_pickle=True)
grid_phi = pd.to_datetime(C["grid_phi"])
g_phi = C["g_phi"].astype(float)
n_phi = len(g_phi)
log(f"\nphi_t grid: n={n_phi}, n_nan={np.isnan(g_phi).sum()}, "
    f"span={grid_phi[0]} .. {grid_phi[-1]}, dt_phi=6.0h "
    f"(weekly~lag {168/6:.1f}, diurnal~lag {24/6:.1f} EXACT)")

def linear_detrend(x):
    x = np.asarray(x, float); idx = np.arange(len(x)); m = ~np.isnan(x)
    if m.sum() < 3: return x.copy()
    b1, b0 = np.polyfit(idx[m], x[m], 1)
    return x - (b0 + b1*idx)

phi_detr = linear_detrend(g_phi)
mu, sd = np.nanmean(phi_detr), np.nanstd(phi_detr)
z_phi = (phi_detr - mu) / sd  # NaN-preserving

S_phi = np.outer(z_phi, z_phi)  # S_phi[i,j] = z_i * z_j; NaN where either is NaN
OK_phi = np.zeros((n_phi, MAXK + 1), bool)
for i in range(n_phi):
    for k in range(1, MAXK + 1):
        if i + k < n_phi and not np.isnan(z_phi[i]) and not np.isnan(z_phi[i+k]):
            OK_phi[i, k] = True

phi_hours_frac = (grid_phi.hour + grid_phi.minute/60.0 + grid_phi.second/3600.0).values
phi_dow = grid_phi.dayofweek.values

# ─────────────────────────────────────────────────────────────────────────────
# All series to process: (key, label, S matrix, OK mask, n, hours_frac, dow,
#                          dt_hours, n_series_rows)
# ─────────────────────────────────────────────────────────────────────────────
SERIES = []
for vname, vecs_d in VARIANTS:
    for l in LAYERS:
        SERIES.append({
            "variant": vname, "layer": l,
            "S": Smat_cosine(vecs_d[l]), "OK": OK, "n": n,
            "hours_frac": hours_frac, "dow": dow, "dt_hours": DT, "n_rows": n,
        })
SERIES.append({
    "variant": "phi_t_detrended", "layer": "aggregate",
    "S": S_phi, "OK": OK_phi, "n": n_phi,
    "hours_frac": phi_hours_frac, "dow": phi_dow, "dt_hours": 6.0, "n_rows": n_phi,
})

SEED_OFFSETS_2D = {
    ("per_month_incl", "surface"): 300, ("per_month_incl", "mid"): 301, ("per_month_incl", "residue"): 302,
    ("detrended", "surface"): 310, ("detrended", "mid"): 311, ("detrended", "residue"): 312,
    ("phi_t_detrended", "aggregate"): 320,
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-series: point-estimate ACF lags1-56, n_pairs/flag, block=64 bootstrap CIs
# (full curve + floor_lag12_16/floor_mid/floor_far), diurnal + weekly bins
# ─────────────────────────────────────────────────────────────────────────────
acf_rows, bin_rows, summary_rows = [], [], []

for sdef in SERIES:
    vname, l = sdef["variant"], sdef["layer"]
    S, OKm, nn = sdef["S"], sdef["OK"], sdef["n"]
    hf, dw, dth = sdef["hours_frac"], sdef["dow"], sdef["dt_hours"]
    log(f"\n--- {vname} / {l} (n={nn}, dt={dth}h) ---")

    ks, ys, dn = sim_curve(S, nn, OKm, MAXK)
    ks_list = list(ks)

    seed = SEED + SEED_OFFSETS_2D[(vname, l)]
    acf_samples, floor_samples, mid_samples, far_samples = boot_long(S, OKm, nn, seed=seed)
    log(f"  [timing] bootstrap done for {vname}/{l}")

    for k, y, d in zip(ks, ys, dn):
        ci = _ci(acf_samples[int(k)])
        flag = bool(d < N_PAIRS_FLAG_THRESH)
        if flag:
            log(f"  ** lag {k:2d}: n_pairs={d} < {N_PAIRS_FLAG_THRESH} (FLAGGED)")
        acf_rows.append({
            "table": "acf_long", "variant": vname, "layer": l, "lag": int(k),
            "n_pairs": int(d), "flag_low_n": flag,
            "acf": float(y), "acf_ci95_lo": ci[0], "acf_ci95_hi": ci[1],
        })

    floor_pt, n_floor = mean_over_lags(ks, ys, FLOOR_LAGS)
    mid_pt, n_mid     = mean_over_lags(ks, ys, MID_LAGS)
    far_pt, n_far     = mean_over_lags(ks, ys, FAR_LAGS)
    floor_ci = _ci(floor_samples)
    mid_ci   = _ci(mid_samples)
    far_ci   = _ci(far_samples)
    n_pairs_floor_min = int(min(dn[[i for i,k in enumerate(ks_list) if k in FLOOR_LAGS]])) if n_floor else None
    n_pairs_mid_min   = int(min(dn[[i for i,k in enumerate(ks_list) if k in MID_LAGS]])) if n_mid else None
    n_pairs_far_min   = int(min(dn[[i for i,k in enumerate(ks_list) if k in FAR_LAGS]])) if n_far else None

    log(f"  floor(12-16)={floor_pt:.4f} CI95=({floor_ci[0]:.4f},{floor_ci[1]:.4f}) n_floor_lags={n_floor} min_n_pairs={n_pairs_floor_min}")
    log(f"  floor_mid(19-25)={mid_pt:.4f} CI95=({mid_ci[0]:.4f},{mid_ci[1]:.4f}) n_mid_lags={n_mid} min_n_pairs={n_pairs_mid_min}")
    log(f"  floor_far(26-32)={far_pt:.4f} CI95=({far_ci[0]:.4f},{far_ci[1]:.4f}) n_far_lags={n_far} min_n_pairs={n_pairs_far_min}")

    summary_rows.append({
        "table": "floor_summary", "variant": vname, "layer": l,
        "dt_hours": dth, "n_series_rows": nn,
        "floor_lag12_16": floor_pt, "floor_lag12_16_ci95_lo": floor_ci[0], "floor_lag12_16_ci95_hi": floor_ci[1],
        "floor_mid_19_25": mid_pt, "floor_mid_ci95_lo": mid_ci[0], "floor_mid_ci95_hi": mid_ci[1],
        "floor_far_26_32": far_pt, "floor_far_ci95_lo": far_ci[0], "floor_far_ci95_hi": far_ci[1],
        "n_pairs_floor_min": n_pairs_floor_min, "n_pairs_mid_min": n_pairs_mid_min, "n_pairs_far_min": n_pairs_far_min,
        "boot_seed": seed, "n_boot": NBOOT, "boot_block": LB_LONG,
    })

    # ── diurnal + weekly bins over all valid pairs at lags 1-56 ──
    diurnal_sum = np.zeros(4); diurnal_n = np.zeros(4, int)
    weekly_sum  = np.zeros(7); weekly_n  = np.zeros(7, int)
    for i in range(nn):
        for k in range(1, MAXK + 1):
            if i + k < nn and OKm[i, k]:
                val = S[i, i+k]
                hour_diff = (hf[i+k] - hf[i]) % 24.0
                bin_h = int(hour_diff // 6.0)
                if bin_h == 4: bin_h = 0  # guard against floating-point edge at 24.0
                diurnal_sum[bin_h] += val; diurnal_n[bin_h] += 1
                dow_diff = int((dw[i+k] - dw[i]) % 7)
                weekly_sum[dow_diff] += val; weekly_n[dow_diff] += 1

    DIURNAL_LABELS = ["0-6h", "6-12h", "12-18h", "18-24h"]
    for b, label in enumerate(DIURNAL_LABELS):
        mean_sim = diurnal_sum[b] / diurnal_n[b] if diurnal_n[b] > 0 else np.nan
        bin_rows.append({
            "table": "diurnal", "variant": vname, "layer": l,
            "bin_label": label, "bin_mean_sim": mean_sim, "bin_n_pairs": int(diurnal_n[b]),
        })
    log(f"  diurnal bins (mean sim, n_pairs): " +
        ", ".join(f"{lab}={diurnal_sum[b]/diurnal_n[b]:.4f}(n={diurnal_n[b]})" if diurnal_n[b]>0 else f"{lab}=NA"
                  for b, lab in enumerate(DIURNAL_LABELS)))

    for b in range(7):
        mean_sim = weekly_sum[b] / weekly_n[b] if weekly_n[b] > 0 else np.nan
        bin_rows.append({
            "table": "weekly", "variant": vname, "layer": l,
            "bin_label": f"{b}d", "bin_mean_sim": mean_sim, "bin_n_pairs": int(weekly_n[b]),
        })
    log(f"  weekly bins (mean sim, n_pairs): " +
        ", ".join(f"{b}d={weekly_sum[b]/weekly_n[b]:.4f}(n={weekly_n[b]})" if weekly_n[b]>0 else f"{b}d=NA"
                  for b in range(7)))

    # stash for figure
    sdef["ks"], sdef["ys"], sdef["dn"] = ks, ys, dn
    sdef["acf_samples"] = acf_samples
    sdef["floor_pt"], sdef["mid_pt"], sdef["far_pt"] = floor_pt, mid_pt, far_pt

# ─────────────────────────────────────────────────────────────────────────────
# Cross-check floor_lag12_16 against Task 2C's phase3_floor_origin.csv
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Cross-check floor(12-16) vs Task 2C phase3_floor_origin.csv ---")
fo = pd.read_csv(TABLES / "phase3_floor_origin.csv", comment="#")
for row in summary_rows:
    if row["variant"] == "phi_t_detrended": continue
    ref = fo[(fo["variant"] == row["variant"]) & (fo["layer"] == row["layer"])]
    ref_floor = float(ref["floor"].iloc[0])
    ok = abs(row["floor_lag12_16"] - ref_floor) < 1e-9
    log(f"  {row['variant']:15s} {row['layer']:8s} 2D={row['floor_lag12_16']:.6f}  "
        f"2C={ref_floor:.6f}  match={ok}")
    assert ok, f"floor(12-16) mismatch for {row['variant']}/{row['layer']}"
log("All per_month_incl/detrended floor(12-16) point estimates reproduce Task 2C exactly.")

# ─────────────────────────────────────────────────────────────────────────────
# Decision metric per layer (primary=per_month_incl, check=detrended)
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Decision metric: floor_far vs floor_mid vs floor(12-16) ---")
verdicts = {}
for row in summary_rows:
    vname, l = row["variant"], row["layer"]
    f12, fm, ff = row["floor_lag12_16"], row["floor_mid_19_25"], row["floor_far_26_32"]
    if ff > fm > 1e-9 and (ff - fm) > 0:
        regime = "weekly-leaning (floor_far > floor_mid)"
    elif fm > ff:
        regime = "decaying (floor_far < floor_mid)"
    else:
        regime = "flat"
    log(f"  {vname:16s} {l:8s} floor12-16={f12:.4f}  floor_mid={fm:.4f} CI95=({row['floor_mid_ci95_lo']:.4f},{row['floor_mid_ci95_hi']:.4f})  "
        f"floor_far={ff:.4f} CI95=({row['floor_far_ci95_lo']:.4f},{row['floor_far_ci95_hi']:.4f})  -> {regime}")
    verdicts[(vname, l)] = regime

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 1: phase3_floor_periodicity.csv
# ─────────────────────────────────────────────────────────────────────────────
out_path = TABLES / "phase3_floor_periodicity.csv"
with open(out_path, "w") as f:
    f.write(
        "# Task 2D: periodicity check on the residual correlation floor (follow-up to\n"
        "# Task 2C). Single stacked table, discriminated by the 'table' column:\n"
        "#\n"
        "#   table='acf_long'      : per (variant, layer), lag k=1..56 -- n_pairs,\n"
        "#                           flag_low_n (n_pairs<80), acf (point estimate,\n"
        "#                           mean similarity over original-adjacent valid\n"
        "#                           pairs), acf_ci95_lo/hi (block=64 adjacency-\n"
        "#                           preserving moving-block bootstrap, 1000 resamples).\n"
        "#   table='diurnal'       : per (variant, layer), bin_label in\n"
        "#                           {0-6h,6-12h,12-18h,18-24h} = (hour-of-day(t+k) -\n"
        "#                           hour-of-day(t)) mod 24, pooled over all valid pairs\n"
        "#                           at lags 1-56. bin_mean_sim, bin_n_pairs.\n"
        "#   table='weekly'        : per (variant, layer), bin_label in {0d..6d} =\n"
        "#                           (dow(t+k) - dow(t)) mod 7, pooled over all valid\n"
        "#                           pairs at lags 1-56. bin_mean_sim, bin_n_pairs.\n"
        "#   table='floor_summary' : per (variant, layer) -- floor_lag12_16 (mean ACF\n"
        "#                           lags 12-16, recomputed here, exact match to Task\n"
        "#                           2C's 'floor' column), floor_mid_19_25 (lags 19-25,\n"
        "#                           between-band), floor_far_26_32 (lags 26-32, weekly\n"
        "#                           recurrence band, ~168h/dt=29.26 lags), each with\n"
        "#                           block=64 bootstrap 95% CI; n_pairs_*_min = minimum\n"
        "#                           n_pairs across the lags in that band.\n"
        "#\n"
        "# variants: per_month_incl / detrended = layer-centroid variants from Task 2C\n"
        "#   (per-month-centered / linear-detrended), 3 layers, n=210 rows, dt=5.7405h,\n"
        "#   cosine similarity. phi_t_detrended = comparison aggregate scalar (Task 2\n"
        "#   Analysis C linearly-detrended phi_t on its regular 6h grid, n=225,\n"
        "#   layer='aggregate'); 'similarity' = standardized-product z(t)*z(t+k), whose\n"
        "#   lag-mean is the standard biased-autocovariance ACF estimator (cosine\n"
        "#   similarity is undefined for a 1-d scalar).\n"
        "#\n"
        "# Bootstrap: single block=64 (smallest multiple of 8 exceeding lag 56; a block\n"
        "# of length Lb can only contain pairs up to lag Lb-1, so block=24 from Task 2C\n"
        "# cannot reach lag 56) adjacency-preserving moving-block bootstrap, 1000\n"
        "# resamples, fixed seed = 42+offset (see floor_summary.boot_seed; offsets\n"
        "# 300-302/310-312/320, distinct from Tasks 2B/2C). block=64 (Lb-1=63>=56)\n"
        "# covers every lag 1-56, superseding Task 2C's block=8 for the short lags too.\n"
    )
    acf_df = pd.DataFrame(acf_rows)
    bin_df = pd.DataFrame(bin_rows)
    sum_df = pd.DataFrame(summary_rows)
    all_df = pd.concat([acf_df, bin_df, sum_df], ignore_index=True, sort=False)
    cols = ["table", "variant", "layer", "lag", "n_pairs", "flag_low_n", "acf",
            "acf_ci95_lo", "acf_ci95_hi", "bin_label", "bin_mean_sim", "bin_n_pairs",
            "dt_hours", "n_series_rows",
            "floor_lag12_16", "floor_lag12_16_ci95_lo", "floor_lag12_16_ci95_hi",
            "floor_mid_19_25", "floor_mid_ci95_lo", "floor_mid_ci95_hi",
            "floor_far_26_32", "floor_far_ci95_lo", "floor_far_ci95_hi",
            "n_pairs_floor_min", "n_pairs_mid_min", "n_pairs_far_min",
            "boot_seed", "n_boot", "boot_block"]
    all_df = all_df[cols]
    all_df.to_csv(f, index=False)
log(f"\nWrote {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 2: figure -- per-layer long-lag ACF (3 panels) + phi_t comparison panel
# ─────────────────────────────────────────────────────────────────────────────
log("[timing] starting figure")
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
axes = axes.flatten()
COLORS = {"per_month_incl": "#2ecc71", "detrended": "#9b59b6", "phi_t_detrended": "#34495e"}
LABELS = {"per_month_incl": "per-month centered (primary)", "detrended": "linear-detrended (check)",
          "phi_t_detrended": "phi_t, linear-detrended (comparison)"}

panel_defs = [("surface", ["per_month_incl", "detrended"]),
               ("mid", ["per_month_incl", "detrended"]),
               ("residue", ["per_month_incl", "detrended"]),
               ("aggregate", ["phi_t_detrended"])]

for ax, (l, vnames) in zip(axes, panel_defs):
    log(f"[timing] panel {l}")
    # shaded bands
    ax.axvspan(12, 16, color="grey", alpha=0.12, label="lags 12-16 (Task 2C floor)")
    ax.axvspan(19, 25, color="#f1c40f", alpha=0.12, label="lags 19-25 (floor_mid)")
    ax.axvspan(26, 32, color="#e67e22", alpha=0.15, label="lags 26-32 (floor_far / weekly recurrence)")
    ax.axhline(0, color="grey", lw=0.6)
    if l == "aggregate":
        ax.axvline(168/6.0, color="#c0392b", ls="--", lw=1, label="168h (lag 28.0 @ dt=6h)")
    else:
        ax.axvline(168/DT, color="#c0392b", ls="--", lw=1, label=f"168h (lag {168/DT:.2f} @ dt={DT:.2f}h)")
    for vname in vnames:
        sd = next(s for s in SERIES if s["variant"] == vname and s["layer"] == l)
        ks, ys = sd["ks"], sd["ys"]
        ci_lo = np.array([_ci(sd["acf_samples"][int(k)])[0] for k in ks])
        ci_hi = np.array([_ci(sd["acf_samples"][int(k)])[1] for k in ks])
        color = COLORS[vname]
        ax.plot(ks, ys, "-o", ms=3, lw=1.2, color=color, label=LABELS[vname])
        ax.fill_between(ks, ci_lo, ci_hi, color=color, alpha=0.15)
    ax.set_title(l)
    ax.set_xlabel("lag k")
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("mean cosine similarity")
axes[2].set_ylabel("mean cosine similarity")
axes[3].set_ylabel("standardized-product ACF (z·z)")
fig.suptitle("Phase 3 Task 2D -- long-lag ACF (lags 1-56, ~13 days), block=64 bootstrap 95% CI bands", y=1.0)
log("[timing] tight_layout")
fig.tight_layout()
fig_path = FIGURES / "phase3_task2d_longlag_acf.png"
log("[timing] savefig")
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig_path}")

(LOGS / "task2d_console_log.txt").write_text("\n".join(log_lines))
log("\n=== DONE (Task 2D) ===")
