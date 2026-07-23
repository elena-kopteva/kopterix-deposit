"""
Kopterix Phase 3, Task 2C -- origin of the centered correlation floor.

Follow-up to Task 2B. Task 2B found that grand-mean-centered layer centroids show a fast
decorrelation (lag-1 similarity 0.36-0.39, AR1 tau 5.6-6.0h) above a persistent
correlation floor (~0.17-0.19 at lag 16). A single grand mean computed over the whole
two-month window under-subtracts if April and May have different month-level centroids
(Phase 2 cross-month drift ratio 1.06 for the mean embedding); rows within a month would
then share a residual month-direction, producing a non-decaying within-month plateau.

This task tests whether the floor is that slow-drift artifact or survives stronger
centering, via two independent stronger-centering variants:
  - per-month centering (subtract April mean from April rows, May mean from May rows,
    per layer), checked both including and excluding lag pairs that straddle the
    April/May boundary;
  - linear-in-time detrending (per coordinate, per layer, OLS fit over the full window,
    subtract fitted line) -- removes any slow secular drift, not just a month-step.

Conventions inherited unchanged from Task 1/2/2B:
  - Window [2026-04-01, 2026-06-01) UTC, half-open.
  - Seed 42 (per-series bootstrap streams documented inline, distinct offsets from 2B).
  - kopterix_state.csv: Timestamp (ISO+tz), MeanEmbedding (384-d), LayerCentroids
    (dict surface/mid/residue, each 384-d).
  - Cosine similarity = dot(a,b)/(|a||b|); pair excluded if either norm < 1e-12.
  - Raw inputs are READ-ONLY; all outputs are new files in phase3/.
  - n=210 valid layer-centroid rows (104 April + 106 May), original-adjacent pairing
    only (no bridging gaps > 12h), exactly as Task 2/2B.
  - tau fit: sim(k) = A*exp(-k*dt/tau) + C on lags k=1..16 (same model as Task 2/2B).
  - Bootstrap: adjacency-preserving moving-block bootstrap, fixed per-(layer,variant)
    seeds (42 + offset, offsets distinct from Task 2B's). block=8 for rho1/tau_ar1 (as
    specified); a SEPARATE block=24 bootstrap (smallest multiple of 8 exceeding lag 16,
    seed = ar1 seed + 5000) is used for the floor's CI, since lags 12-16 cannot occur
    within an 8-row block by construction (max representable lag in an Lb-row block is
    Lb-1).
  - "floor" := mean ACF value over lags 12-16 (point estimate and bootstrap CI).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
BIG    = 12.0     # hours; do not bridge gaps larger than this (same as Task 2/2B)
LB     = 8        # bootstrap block length (rows), used for rho1/tau_ar1 CIs (as specified)
NBOOT  = 1000
MAXK   = 16
FLOOR_LAGS = list(range(12, 17))  # lags 12-16
# A block of length 8 cannot contain a lag-12..16 pair by construction (max representable
# lag within a block of length Lb is Lb-1). The floor (mean ACF over lags 12-16) therefore
# needs a larger block length to have a well-defined adjacency-preserving bootstrap. We
# use LB_FLOOR=24 (smallest multiple of 8 exceeding MAXK=16), same n_boot/seed-offset
# convention, documented explicitly in outputs.
LB_FLOOR = 24

log_lines = []
def log(msg): print(msg); log_lines.append(str(msg))

log("=== Kopterix Phase 3 Task 2C: origin of the centered correlation floor ===")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD (same parsing as Task 2 / 2B)
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

# Grand mean over the 215 valid state rows (Phase 2 zero-mode definition) -- same as 2B
me_valid = sw[sw["mean_emb"].notna()].copy()
grand_mean = np.stack(me_valid["mean_emb"].values).mean(axis=0)
log(f"Grand mean: n={len(me_valid)} (expect 215), L2 norm={np.linalg.norm(grand_mean):.10f} "
    f"(Phase 2 zero-mode: 0.3191144627010268)")
assert len(me_valid) == 215, f"expected 215 valid mean_emb rows, got {len(me_valid)}"
assert abs(np.linalg.norm(grand_mean) - 0.3191144627010268) < 1e-9

lc_ok = sw["lc"].notna() & sw["mean_emb"].notna()
slc = sw[lc_ok].copy().sort_values("ts").reset_index(drop=True)
n = len(slc)
log(f"Valid layer-centroid rows: {n} (expect 210)")
assert n == 210

gaps = slc["ts"].diff().dt.total_seconds().values / 3600.0  # hours, gaps[0]=nan
median_gap = float(np.nanmedian(gaps[1:]))
DT = median_gap
log(f"dt (median consecutive gap) = {DT:.6f} h")

raw_vecs = {l: np.array([slc.iloc[i]["lc"][l] for i in range(n)]) for l in LAYERS}

# Month split (April vs May) for the 210 valid layer-centroid rows
months = slc["ts"].dt.month.values
n_apr, n_may = int((months == 4).sum()), int((months == 5).sum())
log(f"Month split: April={n_apr}, May={n_may} (expect 104 + 106 = 210)")
assert n_apr == 104 and n_may == 106

# ─────────────────────────────────────────────────────────────────────────────
# Adjacency mask: OK[i,k] = True iff rows i..i+k are gap-valid (no NaN gap, none > BIG)
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

# OK_excl: same as OK, but additionally requires month(i) == month(i+k) (excludes
# lag pairs that straddle the April/May boundary)
OK_excl = OK.copy()
n_straddle = np.zeros(MAXK + 1, int)
for i in range(n):
    for k in range(1, MAXK + 1):
        if i + k < n and OK[i, k] and months[i] != months[i+k]:
            OK_excl[i, k] = False
            n_straddle[k] += 1
log(f"Cross-month (April->May) straddling pairs per lag (k=1..16): "
    f"{[int(x) for x in n_straddle[1:]]}")

# ─────────────────────────────────────────────────────────────────────────────
# Lag-k cosine-similarity ACF + AexpC fit + AR(1) conversion + adjacency-preserving
# moving-block bootstrap (tau_ar1 + floor), reused/extended from Task 2B
# ─────────────────────────────────────────────────────────────────────────────
def Smat(vecs):
    Vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return Vn @ Vn.T

def sim_curve(S, n, OKm, idx_blocks=None, maxk=MAXK):
    """Mean cosine similarity per lag k=1..maxk over original-adjacent valid pairs.
    idx_blocks: list of (start,length) contiguous blocks (bootstrap); None = full data."""
    num = np.zeros(maxk + 1); den = np.zeros(maxk + 1)
    if idx_blocks is None:
        for i in range(n):
            for k in range(1, maxk + 1):
                if i + k < n and OKm[i, k]:
                    num[k] += S[i, i+k]; den[k] += 1
    else:
        for (s0, Lb) in idx_blocks:
            for i in range(s0, s0 + Lb):
                for k in range(1, maxk + 1):
                    j = i + k
                    if j < s0 + Lb and j < n and OKm[i, k]:
                        num[k] += S[i, j]; den[k] += 1
    ks = np.array([k for k in range(1, maxk + 1) if den[k] > 0])
    ys = np.array([num[k] / den[k] for k in ks])
    dn = np.array([den[k] for k in ks], dtype=int)
    return ks, ys, dn

def fit_tau_AexpC_full(ks, ys, dt):
    """Returns A, tau, C, R2 for sim(k) = A*exp(-k*dt/tau) + C on lags ks."""
    if len(ks) < 3: return np.nan, np.nan, np.nan, np.nan
    def model(k, A, tau, C): return A * np.exp(-k * dt / tau) + C
    try:
        p0 = [ys[0]-ys[-1], 24.0, ys[-1]]
        popt, _ = curve_fit(model, ks, ys, p0=p0, bounds=([0,0.1,-1],[2,1e5,1]), maxfev=20000)
        resid = ys - model(ks, *popt)
        ss = np.sum(resid**2); st = np.sum((ys - ys.mean())**2)
        r2 = 1 - ss/st if st > 0 else np.nan
        A, tau, C = popt
        return float(A), float(tau), float(C), float(r2)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def tau_ar1(rho1, dt):
    if rho1 is None or np.isnan(rho1) or rho1 <= 0: return np.nan
    return float(-dt / np.log(rho1))

def floor_of(ks, ys, floor_lags=FLOOR_LAGS):
    idx = [i for i, k in enumerate(ks) if k in floor_lags]
    if not idx: return np.nan, 0
    return float(np.mean(ys[idx])), len(idx)

def _ci(a):
    a = np.array(a)
    return (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) if len(a) > 10 else (np.nan, np.nan)

def boot_ci_full(S, OKm, dt, seed, n_boot=NBOOT, Lb=LB, Lb_floor=LB_FLOOR, maxk=MAXK):
    """Two adjacency-preserving moving-block bootstraps from the same point-estimate
    similarity matrix S / mask OKm:
      - block=Lb (=8, as specified) for tau_ar1, derived from each resample's lag-1
        similarity. (lags 1..Lb-1 are representable within an Lb-row block.)
      - block=Lb_floor (=24) for the floor (mean ACF over FLOOR_LAGS=12-16 of each
        resample), since lags 12-16 cannot occur within an 8-row block by construction.
        Uses a distinct seed (seed + 5000) so the two bootstraps are independent but
        each individually reproducible.
    """
    # --- tau_ar1, block=Lb ---
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / Lb))
    taus_ar1 = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - Lb, size=nb)
        blocks = [(int(s), Lb) for s in starts]
        ks, ys, dn = sim_curve(S, n, OKm, blocks, maxk)
        if len(ks) == 0: continue
        if ks[0] == 1:
            ta = tau_ar1(ys[0], dt)
            if not np.isnan(ta): taus_ar1.append(ta)

    # --- floor, block=Lb_floor ---
    seed_floor = seed + 5000
    rng_f = np.random.default_rng(seed_floor)
    nb_f = int(np.ceil(n / Lb_floor))
    floors = []
    for _ in range(n_boot):
        starts = rng_f.integers(0, n - Lb_floor, size=nb_f)
        blocks = [(int(s), Lb_floor) for s in starts]
        ks, ys, dn = sim_curve(S, n, OKm, blocks, maxk)
        if len(ks) == 0: continue
        fl, nfl = floor_of(ks, ys)
        if nfl > 0: floors.append(fl)

    return _ci(taus_ar1), _ci(floors), len(taus_ar1), len(floors), seed_floor

log(f"\n[DONE: setup] n={n}, dt={DT:.4f}h, FLOOR_LAGS={FLOOR_LAGS}, "
    f"NBOOT={NBOOT}, LB={LB}, LB_FLOOR={LB_FLOOR}")

# ─────────────────────────────────────────────────────────────────────────────
# Build the three centering variants
# ─────────────────────────────────────────────────────────────────────────────
# (1) grand-mean: same single grand_mean subtracted from every row (Task 2B "centered")
gm_vecs = {l: raw_vecs[l] - grand_mean[None, :] for l in LAYERS}

# (2) per-month: subtract April-mean (from April rows) / May-mean (from May rows), per layer
apr_mask = (months == 4)
may_mask = (months == 5)
pm_vecs = {}
pm_means = {}
for l in LAYERS:
    mean_apr = raw_vecs[l][apr_mask].mean(axis=0)
    mean_may = raw_vecs[l][may_mask].mean(axis=0)
    pm_means[l] = (mean_apr, mean_may)
    v = raw_vecs[l].copy()
    v[apr_mask] -= mean_apr[None, :]
    v[may_mask] -= mean_may[None, :]
    pm_vecs[l] = v
    log(f"  per-month means {l:8s}: |mean_apr|={np.linalg.norm(mean_apr):.4f} "
        f"|mean_may|={np.linalg.norm(mean_may):.4f} "
        f"cos(mean_apr,mean_may)={np.dot(mean_apr,mean_may)/(np.linalg.norm(mean_apr)*np.linalg.norm(mean_may)):.4f}")

# (3) detrended: per-coordinate OLS linear-in-time fit over the full window, subtract
t_hours = (slc["ts"] - slc["ts"].iloc[0]).dt.total_seconds().values / 3600.0
dt_vecs = {}
dt_slopes = {}
for l in LAYERS:
    Y = raw_vecs[l]  # (n, 384)
    coeffs = np.polyfit(t_hours, Y, 1)  # shape (2, 384): [slope, intercept]
    pred = coeffs[0][None, :] * t_hours[:, None] + coeffs[1][None, :]
    dt_vecs[l] = Y - pred
    dt_slopes[l] = coeffs[0]
    log(f"  detrend {l:8s}: |slope|_2 = {np.linalg.norm(coeffs[0]):.6f} per-h, "
        f"mean|residual|={np.linalg.norm(dt_vecs[l],axis=1).mean():.4f} "
        f"(mean|raw|={np.linalg.norm(Y,axis=1).mean():.4f}, "
        f"mean|grand-mean centered|={np.linalg.norm(gm_vecs[l],axis=1).mean():.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions: (name, vecs-dict, OK-mask)
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = [
    ("grand_mean",     gm_vecs, OK),
    ("per_month_incl", pm_vecs, OK),
    ("per_month_excl", pm_vecs, OK_excl),
    ("detrended",      dt_vecs, OK),
]

# ─────────────────────────────────────────────────────────────────────────────
# Point-estimate ACF for all variants x layers (no bootstrap yet) -- sanity pass
# ─────────────────────────────────────────────────────────────────────────────
log("\n--- Point-estimate ACF (no bootstrap) ---")
point = {}
for vname, vecs_d, OKm in VARIANTS:
    for l in LAYERS:
        S = Smat(vecs_d[l])
        ks, ys, dn = sim_curve(S, n, OKm, None)
        rho1 = float(ys[list(ks).index(1)]) if 1 in ks else np.nan
        A, tau_exp, C, r2 = fit_tau_AexpC_full(ks, ys, DT)
        tau_a1 = tau_ar1(rho1, DT)
        fl, nfl = floor_of(ks, ys)
        point[(vname, l)] = dict(ks=ks, ys=ys, dn=dn, rho1=rho1, A=A, tau_exp=tau_exp,
                                  C=C, r2=r2, tau_ar1=tau_a1, floor=fl, n_floor_lags=nfl)
        n1 = int(dn[list(ks).index(1)]) if 1 in ks else -1
        log(f"  {vname:14s} {l:8s} rho1={rho1:.4f}  tau_exp={tau_exp:7.2f}h(R2={r2:.3f},C={C:+.4f})  "
            f"tau_ar1={tau_a1:6.2f}h  floor={fl:.4f} (n_floor_lags={nfl})  n_pairs_lag1={n1}")

# Sanity check against Task 2B's stored centered values
log("\n--- Sanity check vs Task 2B phase3_centered_layer_tau.csv (centered_* cols) ---")
T2B = {
    "surface": dict(rho1=0.3663944181739324, tau_exp=57.37091432884792, r2=0.9473660060548983, tau_ar1=5.717331997840964),
    "mid":     dict(rho1=0.3611842705723368, tau_exp=100.87473239146053, r2=0.971292890544437, tau_ar1=5.63692449626452),
    "residue": dict(rho1=0.3861223439620865, tau_exp=64.52960952959782, r2=0.9712344756693799, tau_ar1=6.032421029157965),
}
for l in LAYERS:
    p = point[("grand_mean", l)]
    ref = T2B[l]
    ok_rho1 = abs(p["rho1"] - ref["rho1"]) < 1e-9
    ok_tau  = abs(p["tau_exp"] - ref["tau_exp"]) < 1e-4
    ok_r2   = abs(p["r2"] - ref["r2"]) < 1e-6
    ok_ar1  = abs(p["tau_ar1"] - ref["tau_ar1"]) < 1e-9
    log(f"  {l:8s} rho1 match={ok_rho1}  tau_exp match={ok_tau}  R2 match={ok_r2}  tau_ar1 match={ok_ar1}")
    assert ok_rho1 and ok_tau and ok_r2 and ok_ar1, f"Task2B reproduction mismatch for {l}"
log("All grand-mean point estimates reproduce Task 2B centered_* values exactly.")

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: tau_ar1 CI and floor CI, per variant x layer (fixed seeds, distinct
# from Task 2B's offsets {0,1,2,10,11,12} -> seeds {42,43,44,52,53,54})
# ─────────────────────────────────────────────────────────────────────────────
SEED_OFFSETS_2C = {
    ("grand_mean","surface"):200, ("grand_mean","mid"):201, ("grand_mean","residue"):202,
    ("per_month_incl","surface"):210, ("per_month_incl","mid"):211, ("per_month_incl","residue"):212,
    ("per_month_excl","surface"):220, ("per_month_excl","mid"):221, ("per_month_excl","residue"):222,
    ("detrended","surface"):230, ("detrended","mid"):231, ("detrended","residue"):232,
}

log("\n--- Bootstrap CIs (tau_ar1: block=8; floor: block=24): n_boot=1000, fixed seeds ---")
for vname, vecs_d, OKm in VARIANTS:
    for l in LAYERS:
        S = Smat(vecs_d[l])
        seed = SEED + SEED_OFFSETS_2C[(vname, l)]
        ci_ar1, ci_floor, n_ar1, n_floor, seed_floor = boot_ci_full(S, OKm, DT, seed=seed)
        p = point[(vname, l)]
        p["seed"] = seed
        p["seed_floor"] = seed_floor
        p["ci_ar1"] = ci_ar1
        p["ci_floor"] = ci_floor
        p["n_boot_ar1"] = n_ar1
        p["n_boot_floor"] = n_floor
        log(f"  {vname:14s} {l:8s} tau_ar1={p['tau_ar1']:6.2f}h CI95=({ci_ar1[0]:.2f},{ci_ar1[1]:.2f})  "
            f"floor={p['floor']:.4f} CI95=({ci_floor[0]:.4f},{ci_floor[1]:.4f})  "
            f"(seed={seed}, seed_floor={seed_floor}, n_boot_ar1={n_ar1}, n_boot_floor={n_floor})")

# floor_ratio relative to grand_mean (same layer)
for vname, _, _ in VARIANTS:
    for l in LAYERS:
        point[(vname,l)]["floor_ratio"] = point[(vname,l)]["floor"] / point[("grand_mean",l)]["floor"]

log("\n--- floor_ratio = floor(variant) / floor(grand_mean), per layer ---")
for l in LAYERS:
    for vname, _, _ in VARIANTS:
        log(f"  {l:8s} {vname:14s} floor={point[(vname,l)]['floor']:.4f}  "
            f"floor_ratio={point[(vname,l)]['floor_ratio']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 1: phase3_floor_origin.csv
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for vname, vecs_d, OKm in VARIANTS:
    for l in LAYERS:
        p = point[(vname,l)]
        ks, ys, dn = p["ks"], p["ys"], p["dn"]
        ks_list = list(ks)
        n_pairs_lag1  = int(dn[ks_list.index(1)])  if 1  in ks_list else None
        n_pairs_lag16 = int(dn[ks_list.index(16)]) if 16 in ks_list else None
        floor_idx = [i for i,k in enumerate(ks_list) if k in FLOOR_LAGS]
        n_pairs_floor_min = int(min(dn[floor_idx])) if floor_idx else None
        rows.append({
            "variant": vname,
            "layer": l,
            "n_layer_rows": n,
            "dt_hours": DT,
            "n_pairs_lag1": n_pairs_lag1,
            "n_pairs_lag16": n_pairs_lag16,
            "n_pairs_floor_min": n_pairs_floor_min,
            "rho1": p["rho1"],
            "tau_ar1_h": p["tau_ar1"],
            "tau_ar1_ci95_lo": p["ci_ar1"][0],
            "tau_ar1_ci95_hi": p["ci_ar1"][1],
            "tau_exp_h": p["tau_exp"],
            "tau_exp_R2": p["r2"],
            "tau_exp_A": p["A"],
            "tau_exp_C": p["C"],
            "floor": p["floor"],
            "floor_ci95_lo": p["ci_floor"][0],
            "floor_ci95_hi": p["ci_floor"][1],
            "floor_ratio_vs_grand_mean": p["floor_ratio"],
            "boot_seed": p["seed"],
            "boot_seed_floor": p["seed_floor"],
            "n_boot_ar1": p["n_boot_ar1"],
            "n_boot_floor": p["n_boot_floor"],
        })
out_df = pd.DataFrame(rows)
out_path = TABLES / "phase3_floor_origin.csv"
with open(out_path, "w") as f:
    f.write(
        "# Task 2C: origin of the centered correlation floor. Four centering variants,\n"
        "# n=210 valid layer-centroid rows (104 April + 106 May), original-adjacent\n"
        "# pairs only (no bridging gaps>12h), dt=median consecutive gap (h).\n"
        "#   grand_mean     = layer centroid minus the single grand-mean MeanEmbedding\n"
        "#                     over all 215 in-window valid state rows (= Task 2B 'centered').\n"
        "#   per_month_incl = layer centroid minus its own month's mean centroid (April\n"
        "#                     mean from the 104 April rows, May mean from the 106 May\n"
        "#                     rows, per layer); ACF over ALL original-adjacent valid\n"
        "#                     pairs (incl. the few pairs straddling the April/May boundary).\n"
        "#   per_month_excl = same per-month-centered series, but ACF EXCLUDES lag pairs\n"
        "#                     whose two rows fall in different months.\n"
        "#   detrended       = layer centroid minus a per-coordinate OLS linear-in-time fit\n"
        "#                     (fit over the full 210-row window, all 384 dims), independent\n"
        "#                     of grand-mean/per-month centering.\n"
        "# rho1 = lag-1 mean cosine similarity. tau_ar1_h = -dt/ln(rho1) with adjacency-\n"
        "# preserving moving-block-bootstrap 95% CI (block=8, 1000 resamples, fixed seed\n"
        "# = 42+offset, see boot_seed; offsets distinct from Task 2B's {0,1,2,10,11,12}).\n"
        "# tau_exp_h/R2/A/C = NLS fit sim(k)=A*exp(-k*dt/tau)+C on lags 1-16 (same model as\n"
        "# Task 2/2B); C is the fitted asymptotic offset, reported for comparison with the\n"
        "# empirical floor. floor = mean ACF over lags 12-16 (point estimate). Its 95% CI\n"
        "# uses a SEPARATE adjacency-preserving moving-block bootstrap with block=24\n"
        "# (seed = boot_seed_floor = boot_seed+5000, 1000 resamples): a block of length 8\n"
        "# cannot contain any lag-12..16 pair by construction (max representable lag in an\n"
        "# Lb-row block is Lb-1), so block=8 cannot yield a floor CI; block=24 is the\n"
        "# smallest multiple of 8 exceeding lag 16.\n"
        "# floor_ratio_vs_grand_mean = floor(variant)/floor(grand_mean) for the same layer\n"
        "# (the Task 2C decision metric; grand_mean rows have ratio=1).\n"
        "# n_pairs_lag1/lag16/floor_min = number of original-adjacent valid pairs at lag 1,\n"
        "# lag 16, and the minimum across lags 12-16 respectively.\n"
    )
    out_df.to_csv(f, index=False)
log(f"\nWrote {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 2: figure -- per layer, overlay grand_mean / per_month_incl / detrended ACF
# ─────────────────────────────────────────────────────────────────────────────
FIG_VARIANTS = [
    ("grand_mean",     "#3498db", "o", "grand-mean centered"),
    ("per_month_incl", "#2ecc71", "s", "per-month centered"),
    ("detrended",      "#9b59b6", "^", "linear-detrended"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
kk = np.linspace(1, 16, 100)
for ax, l in zip(axes, LAYERS):
    for vname, color, marker, label in FIG_VARIANTS:
        p = point[(vname, l)]
        ks, ys = p["ks"], p["ys"]
        ax.plot(ks, ys, marker, ms=5, color=color,
                label=f"{label} (rho1={p['rho1']:.3f}, floor={p['floor']:.3f})")
        if not np.isnan(p["tau_exp"]):
            A, tau, C = p["A"], p["tau_exp"], p["C"]
            ax.plot(kk, A*np.exp(-kk*DT/tau)+C, "-", color=color, lw=1.2, alpha=0.7)
    ax.axvspan(12, 16, color="grey", alpha=0.10, label="floor lags (12-16)" if l == LAYERS[0] else None)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{l}")
    ax.set_xlabel("lag k (steps, dt=%.2fh)" % DT)
    ax.legend(fontsize=7.5)
axes[0].set_ylabel("mean cosine similarity")
fig.suptitle("Phase 3 Task 2C -- centered-centroid ACF: grand-mean vs per-month vs "
              "linear-detrended (lags 1-16, original-adjacent pairs)", y=1.04)
fig.tight_layout()
fig_path = FIGURES / "phase3_task2c_floor_acf.png"
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig_path}")

(LOGS / "task2c_console_log.txt").write_text("\n".join(log_lines))
log("\n=== DONE (Task 2C) ===")
