"""
Kopterix Phase 3, Task 2B -- centered per-layer memory check (supplementary to Task 2C).

Motivation: Task 2 found per-layer centroid tau point estimates of 166-193h using cosine
similarity between RAW centroids. Raw centroids share a large static component: the
Phase 2 grand-mean MeanEmbedding has L2 norm 0.3191 while per-row zero-mode residual norms
average ~0.07. High lag-1 similarity (~0.956) may therefore mostly reflect the shared
static direction rather than slowly drifting content. This task measures the memory of
the drifting part alone by centering each layer centroid on the same grand-mean vector
before recomputing the lag-k cosine ACF and tau.

Conventions inherited from Task 1/2/Phase 2 (unchanged):
  - Window [2026-04-01, 2026-06-01) UTC, half-open.
  - Seed 42 (per-series bootstrap streams documented inline).
  - kopterix_state.csv: Timestamp (ISO+tz), MeanEmbedding (384-d), LayerCentroids
    (dict surface/mid/residue, each 384-d).
  - Cosine similarity = dot(a,b)/(|a||b|); pair excluded if either norm < 1e-12.
  - Raw inputs are READ-ONLY; all outputs are new files in phase3/.
  - Per-layer ACF uses only ORIGINAL-ADJACENT valid pairs (no bridging gaps > 12h
    across sparse LayerCentroid rows), exactly as Task 2's per-layer analysis (C5).
  - tau fit: sim(k) = A*exp(-k*dt/tau) + C  on lags k=1..16 (same model as Task 2).
  - Bootstrap: adjacency-preserving moving-block bootstrap (lag-k pairs only counted
    within a block, so original row adjacency is preserved), run as TWO SEPARATE CI
    FAMILIES:
      * exponential-fit CIs (raw and centered tau_exp): corrected NON-CIRCULAR blocks
        of 24 rows, 5000 resamples, lags 1-16, valid starts 0..n-Lb inclusive, every
        replicate supplying all 16 lags. These match the raw per-layer CIs written by
        kopterix_phase3_task2_perlayer_ci.py / kopterix_phase3_task2_D.py.
      * AR(1) CIs (raw and centered tau_ar1): UNCHANGED blocks of 8 rows, 1000
        resamples, lag 1 only.
    A block of length L_b supplies within-block pairs only through lag L_b-1, so the
    exponential fit over lags 1-16 needs blocks of at least 17 rows; block 24 is used.
    Point estimates are identical regardless of bootstrap parameters since they depend
    only on the full-data fit.

Grand mean: mean of MeanEmbedding over the 215 in-window state rows with a valid
MeanEmbedding -- the SAME definition as the Phase 2 zero-mode analysis
(phase3_zero_mode_residuals grand_mean_norm = 0.3191144627010268, reproduced here).
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
BIG    = 12.0     # hours; do not bridge gaps larger than this (same as Task 2)
LB     = 8        # bootstrap block length (rows)
NBOOT  = 1000
MAXK   = 16

log_lines = []
def log(msg): print(msg); log_lines.append(str(msg))

log("=== Kopterix Phase 3 Task 2B: centered per-layer memory check ===")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD (same parsing as Task 2)
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

# Grand mean over the 215 valid state rows (Phase 2 zero-mode definition)
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

raw_vecs      = {l: np.array([slc.iloc[i]["lc"][l] for i in range(n)]) for l in LAYERS}
centered_vecs = {l: raw_vecs[l] - grand_mean[None, :] for l in LAYERS}

log("\nNorm check (mean over rows):")
norm_ratio = {}
for l in LAYERS:
    rn = np.linalg.norm(raw_vecs[l], axis=1)
    cn = np.linalg.norm(centered_vecs[l], axis=1)
    norm_ratio[l] = float(np.mean(cn / rn))
    log(f"  {l:8s} mean|raw|={rn.mean():.4f}  mean|centered|={cn.mean():.4f}  "
        f"mean ratio(centered/raw)={norm_ratio[l]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Adjacency mask: OK[i,k] = True iff rows i..i+k are gap-valid (no NaN gap, none > BIG)
# Shared by raw and centered (depends only on timestamps).
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

# ─────────────────────────────────────────────────────────────────────────────
# Lag-k cosine-similarity ACF (original-adjacent valid pairs only) + AexpC fit
# + AR(1)-style conversion from lag-1 + adjacency-preserving moving-block bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def Smat(vecs):
    Vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return Vn @ Vn.T

def sim_curve(S, n, OK, idx_blocks=None, maxk=MAXK):
    """Mean cosine similarity per lag k=1..maxk over original-adjacent valid pairs.
    idx_blocks: list of (start,length) contiguous blocks (bootstrap); None = full data."""
    num = np.zeros(maxk + 1); den = np.zeros(maxk + 1)
    if idx_blocks is None:
        for i in range(n):
            for k in range(1, maxk + 1):
                if i + k < n and OK[i, k]:
                    num[k] += S[i, i+k]; den[k] += 1
    else:
        for (s0, Lb) in idx_blocks:
            for i in range(s0, s0 + Lb):
                for k in range(1, maxk + 1):
                    j = i + k
                    if j < s0 + Lb and j < n and OK[i, k]:
                        num[k] += S[i, j]; den[k] += 1
    ks = np.array([k for k in range(1, maxk + 1) if den[k] > 0])
    ys = np.array([num[k] / den[k] for k in ks])
    return ks, ys

def fit_tau_AexpC(ks, ys, dt):
    if len(ks) < 3: return np.nan, np.nan
    def model(k, A, tau, C): return A * np.exp(-k * dt / tau) + C
    try:
        p0 = [ys[0]-ys[-1], 24.0, ys[-1]]
        popt, _ = curve_fit(model, ks, ys, p0=p0, bounds=([0,0.1,-1],[2,1e5,1]), maxfev=20000)
        resid = ys - model(ks, *popt)
        ss = np.sum(resid**2); st = np.sum((ys - ys.mean())**2)
        r2 = 1 - ss/st if st > 0 else np.nan
        return float(popt[1]), float(r2)
    except Exception:
        return np.nan, np.nan

def tau_ar1(rho1, dt):
    if rho1 is None or np.isnan(rho1) or rho1 <= 0: return np.nan
    return float(-dt / np.log(rho1))

EXP_LB    = 24     # exponential-fit CI block length (rows) -- corrected
EXP_NBOOT = 5000   # exponential-fit CI resamples -- corrected
AR1_LB    = 8      # AR(1) CI block length (rows) -- unchanged
AR1_NBOOT = 1000   # AR(1) CI resamples -- unchanged

def boot_ci(S, n, OK, dt, seed=SEED, maxk=MAXK,
            exp_Lb=EXP_LB, exp_nboot=EXP_NBOOT, ar1_Lb=AR1_LB, ar1_nboot=AR1_NBOOT):
    """Adjacency-preserving moving-block bootstrap CIs, run as TWO SEPARATE FAMILIES:

      * tau_exp (AexpC fit over lags 1-16): corrected NON-CIRCULAR blocks of 24 rows,
        5000 resamples, valid starts 0..n-Lb INCLUSIVE, every replicate supplying all
        16 lags (block 24 reaches lag 23). This is the corrected exponential CI.
      * tau_ar1 (from lag-1 of each resample): UNCHANGED blocks of 8 rows, 1000
        resamples, lag 1 only -- reproduces the earlier AR(1) intervals exactly (same
        seed, block, resample count and draw order).

    In both families lagged pairs are counted only within a sampled original block
    (original adjacency preserved), never across block boundaries and never wrapping."""
    def ci(a):
        a = np.array(a)
        return (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) if len(a) > 10 else (np.nan, np.nan)

    # -- exponential-fit CI: corrected block=24, 5000 resamples, all 16 lags ---------
    # Pooling ported VERBATIM from the authoritative repair script
    # (kopterix_phase3_task2_perlayer_ci.py :: boot_ci_exp). For each lag k precompute
    # prefix sums over the ORIGINAL index i of the adjacency-valid similarity S[i,i+k] and
    # the pair count; a sampled block [s0, s0+exp_Lb) contributes, at lag k, the pooled
    # sum/count over i in [s0, s0+exp_Lb-k) via one prefix-sum difference. This is the SAME
    # set of within-block pairs (original adjacency mask, no wrap-around) as the earlier
    # explicit nested loop, but follows the repair script's exact summation order, so the
    # pooled per-lag values -- and hence the percentile CIs -- reproduce the authoritative
    # raw and centered intervals to the bit. The exponential RNG (rng_e) is a SEPARATE
    # generator seeded with `seed`; its draw sequence (starts uniform on 0..n-exp_Lb
    # inclusive, size=ceil(n/exp_Lb), once per resample) is unchanged, and it is fully
    # isolated from the AR(1) generator (rng_a) below.
    rng_e = np.random.default_rng(seed)
    nb_e = int(np.ceil(n / exp_Lb))
    csum_s = np.zeros((maxk + 1, n + 1)); csum_d = np.zeros((maxk + 1, n + 1))
    for k in range(1, maxk + 1):
        sval = np.zeros(n); dval = np.zeros(n)
        for i in range(n - k):
            if OK[i, k]: sval[i] = S[i, i + k]; dval[i] = 1.0
        csum_s[k, 1:] = np.cumsum(sval); csum_d[k, 1:] = np.cumsum(dval)
    ks_all = np.arange(1, maxk + 1)
    max_start_e = n - exp_Lb
    taus_exp = []
    for _ in range(exp_nboot):
        starts = rng_e.integers(0, max_start_e + 1, size=nb_e)  # 0..n-Lb inclusive
        ys = np.empty(maxk); ok_all = True
        for k in range(1, maxk + 1):
            hi = starts + (exp_Lb - k); lo = starts   # within-block: i in [s0, s0+Lb-k)
            num = float(np.sum(csum_s[k, hi] - csum_s[k, lo]))
            den = float(np.sum(csum_d[k, hi] - csum_d[k, lo]))
            if den > 0: ys[k - 1] = num / den
            else: ok_all = False; break               # lag has no within-block pair -> discard
        if not ok_all: continue                       # require all 16 fitted lags
        te, _ = fit_tau_AexpC(ks_all, ys, dt)
        if not np.isnan(te): taus_exp.append(te)

    # -- AR(1) CI: UNCHANGED block=8, 1000 resamples, lag 1 only --------------------
    rng_a = np.random.default_rng(seed)
    nb_a = int(np.ceil(n / ar1_Lb))
    taus_ar1 = []
    for _ in range(ar1_nboot):
        starts = rng_a.integers(0, n - ar1_Lb, size=nb_a)
        blocks = [(int(s), ar1_Lb) for s in starts]
        ks, ys = sim_curve(S, n, OK, blocks, maxk)
        if len(ks) < 4: continue
        if ks[0] == 1:
            ta = tau_ar1(ys[0], dt)
            if not np.isnan(ta): taus_ar1.append(ta)

    return ci(taus_exp), ci(taus_ar1), len(taus_exp), len(taus_ar1)

# Fixed, distinct, reproducible bootstrap seeds per (layer, raw/centered)
SEED_OFFSETS = {
    ("surface","raw"):0, ("mid","raw"):1, ("residue","raw"):2,
    ("surface","centered"):10, ("mid","centered"):11, ("residue","centered"):12,
}

log(f"\nFitting AexpC tau (lags 1-16, dt={DT:.4f}h) + AR(1) conversion + "
    f"adjacency-preserving block bootstrap: exponential CI block={EXP_LB}/n_boot={EXP_NBOOT}, "
    f"AR(1) CI block={AR1_LB}/n_boot={AR1_NBOOT}")
results = {}
for l in LAYERS:
    for kind, vecs in [("raw", raw_vecs[l]), ("centered", centered_vecs[l])]:
        S = Smat(vecs)
        ks, ys = sim_curve(S, n, OK, None)
        rho1 = float(ys[0])
        n_pairs_lag1 = None
        # recover den[1] for reporting
        num = np.zeros(MAXK+1); den = np.zeros(MAXK+1)
        for i in range(n):
            if i+1 < n and OK[i,1]:
                den[1] += 1
        n_pairs_lag1 = int(den[1])
        tau_exp, r2 = fit_tau_AexpC(ks, ys, DT)
        tau_a1 = tau_ar1(rho1, DT)
        seed = SEED + SEED_OFFSETS[(l, kind)]
        ci_e, ci_a, n_e, n_a = boot_ci(S, n, OK, DT, seed=seed)
        out_e = (not (ci_e[0] <= tau_exp <= ci_e[1])) if not np.isnan(tau_exp) and not np.any(np.isnan(ci_e)) else None
        out_a = (not (ci_a[0] <= tau_a1 <= ci_a[1])) if not np.isnan(tau_a1) and not np.any(np.isnan(ci_a)) else None
        results[(l, kind)] = dict(rho1=rho1, tau_exp=tau_exp, r2=r2, ci_exp=ci_e,
                                   tau_ar1=tau_a1, ci_ar1=ci_a,
                                   out_exp=out_e, out_ar1=out_a,
                                   n_boot_exp=n_e, n_boot_ar1=n_a,
                                   n_pairs_lag1=n_pairs_lag1, ks=ks, ys=ys, seed=seed)
        log(f"  {l:8s} {kind:9s} rho1={rho1:.4f}  tau_exp={tau_exp:7.2f}h R2={r2:.3f} "
            f"CI95=({ci_e[0]:.2f},{ci_e[1]:.2f}) outCI={out_e}  |  "
            f"tau_ar1={tau_a1:7.2f}h CI95=({ci_a[0]:.2f},{ci_a[1]:.2f}) outCI={out_a}  "
            f"(seed={seed}, n_boot_exp={n_e}, n_boot_ar1={n_a})")

# ─────────────────────────────────────────────────────────────────────────────
# Static-component share metrics (item 6)
# ─────────────────────────────────────────────────────────────────────────────
log("\nStatic-component share (lag-1 similarity, raw vs centered):")
static_share = {}
for l in LAYERS:
    rho1_raw = results[(l,"raw")]["rho1"]
    rho1_cen = results[(l,"centered")]["rho1"]
    share = (rho1_raw - rho1_cen) / rho1_raw
    static_share[l] = share
    log(f"  {l:8s} cos_raw(t,t+1)={rho1_raw:.4f}  cos_centered(t,t+1)={rho1_cen:.4f}  "
        f"share_static={share:.4f}  mean|centered|/|raw|={norm_ratio[l]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 1: side-by-side table
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for l in LAYERS:
    r_raw = results[(l,"raw")]; r_cen = results[(l,"centered")]
    rows.append({
        "layer": l,
        "n_layer_rows": n,
        "n_pairs_lag1": r_raw["n_pairs_lag1"],
        "dt_hours": DT,
        "raw_lag1_sim": r_raw["rho1"],
        "centered_lag1_sim": r_cen["rho1"],
        "static_share_lag1_sim": static_share[l],
        "mean_norm_ratio_centered_over_raw": norm_ratio[l],
        "raw_tau_exp_h": r_raw["tau_exp"], "raw_tau_exp_R2": r_raw["r2"],
        "raw_tau_exp_ci95_lo": r_raw["ci_exp"][0], "raw_tau_exp_ci95_hi": r_raw["ci_exp"][1],
        "raw_tau_exp_outside_ci": r_raw["out_exp"],
        "raw_tau_ar1_h": r_raw["tau_ar1"],
        "raw_tau_ar1_ci95_lo": r_raw["ci_ar1"][0], "raw_tau_ar1_ci95_hi": r_raw["ci_ar1"][1],
        "raw_tau_ar1_outside_ci": r_raw["out_ar1"],
        "centered_tau_exp_h": r_cen["tau_exp"], "centered_tau_exp_R2": r_cen["r2"],
        "centered_tau_exp_ci95_lo": r_cen["ci_exp"][0], "centered_tau_exp_ci95_hi": r_cen["ci_exp"][1],
        "centered_tau_exp_outside_ci": r_cen["out_exp"],
        "centered_tau_ar1_h": r_cen["tau_ar1"],
        "centered_tau_ar1_ci95_lo": r_cen["ci_ar1"][0], "centered_tau_ar1_ci95_hi": r_cen["ci_ar1"][1],
        "centered_tau_ar1_outside_ci": r_cen["out_ar1"],
        "tau_exp_ratio_centered_over_raw": r_cen["tau_exp"]/r_raw["tau_exp"],
        "boot_seed_raw": r_raw["seed"], "boot_seed_centered": r_cen["seed"],
    })
out_df = pd.DataFrame(rows)
out_path = TABLES / "phase3_centered_layer_tau.csv"
with open(out_path, "w") as f:
    f.write(
        "# Task 2B: per-layer centroid memory, RAW vs CENTERED (centered = layer centroid\n"
        "# minus the grand-mean MeanEmbedding over the 215 in-window valid state rows;\n"
        "# grand_mean L2 norm = 0.3191144627010268, same definition as the Phase 2\n"
        "# zero-mode analysis). n=210 valid layer rows, dt=median consecutive gap (h).\n"
        "# *_lag1_sim = mean cosine similarity at lag 1 over original-adjacent valid pairs\n"
        "# (no bridging gaps>12h). static_share_lag1_sim = (raw_lag1_sim -\n"
        "# centered_lag1_sim)/raw_lag1_sim. mean_norm_ratio_centered_over_raw = mean over\n"
        "# rows of |centered_centroid|/|raw_centroid|.\n"
        "# *_tau_exp = NLS fit of sim(k)=A*exp(-k*dt/tau)+C on lags 1-16 (same model as\n"
        "# Task 2's per-layer analysis); *_tau_ar1 = AR(1)-style conversion\n"
        "# -dt/ln(lag1_sim) (NaN if lag1_sim<=0). TWO SEPARATE CI FAMILIES, both adjacency-\n"
        "# preserving (lag-k pairs counted only within a resampled block, so original-\n"
        "# adjacency is preserved; no pairs across block boundaries, no wrap-around):\n"
        "#   - Exponential-fit CIs (raw_tau_exp_ci95_* and centered_tau_exp_ci95_*):\n"
        "#     corrected NON-CIRCULAR block=24 rows, 5000 resamples, lags 1-16 (every\n"
        "#     replicate supplies all 16 lags). Match phase3_memory_timescales.csv raw CIs.\n"
        "#   - AR(1) CIs (raw_tau_ar1_ci95_* and centered_tau_ar1_ci95_*): UNCHANGED\n"
        "#     block=8 rows, 1000 resamples, lag 1 only.\n"
        "# Per-series seeds in boot_seed_raw/boot_seed_centered (=42+offset). *_outside_ci\n"
        "# flags whether the point estimate falls outside its own 95% CI.\n"
        "# tau_exp_ratio_centered_over_raw = centered_tau_exp_h / raw_tau_exp_h.\n"
    )
    out_df.to_csv(f, index=False)
log(f"\nWrote {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 2: figure -- raw vs centered ACF per layer, one panel per layer
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
kk = np.linspace(1, 16, 100)
for ax, l in zip(axes, LAYERS):
    r_raw = results[(l,"raw")]; r_cen = results[(l,"centered")]
    # raw
    ax.plot(r_raw["ks"], r_raw["ys"], "o", ms=5, color="#3498db", label="raw centroid")
    if not np.isnan(r_raw["tau_exp"]):
        def model_raw(k, tau=r_raw["tau_exp"]):
            A, C = None, None
            return None
        # refit A,C at the reported tau for plotting (recompute via curve_fit popt not stored;
        # quick re-derive using least squares on A,C given tau fixed)
        ks_r, ys_r, tau_r = r_raw["ks"], r_raw["ys"], r_raw["tau_exp"]
        X = np.exp(-ks_r*DT/tau_r)
        Amat = np.vstack([X, np.ones_like(X)]).T
        (A_r, C_r), *_ = np.linalg.lstsq(Amat, ys_r, rcond=None)
        ax.plot(kk, A_r*np.exp(-kk*DT/tau_r)+C_r, "-", color="#3498db", lw=1.5,
                label=f"raw fit, tau={tau_r:.0f}h")
    # centered
    ax.plot(r_cen["ks"], r_cen["ys"], "s", ms=5, color="#e74c3c", label="centered centroid")
    if not np.isnan(r_cen["tau_exp"]):
        ks_c, ys_c, tau_c = r_cen["ks"], r_cen["ys"], r_cen["tau_exp"]
        X = np.exp(-ks_c*DT/tau_c)
        Amat = np.vstack([X, np.ones_like(X)]).T
        (A_c, C_c), *_ = np.linalg.lstsq(Amat, ys_c, rcond=None)
        ax.plot(kk, A_c*np.exp(-kk*DT/tau_c)+C_c, "-", color="#e74c3c", lw=1.5,
                label=f"centered fit, tau={tau_c:.0f}h")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{l}\nraw rho1={r_raw['rho1']:.3f}, centered rho1={r_cen['rho1']:.3f}")
    ax.set_xlabel("lag k (steps, dt=%.2fh)" % DT)
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean cosine similarity")
fig.suptitle("Phase 3 Task 2B -- per-layer centroid ACF: raw vs grand-mean-centered "
              "(lags 1-16, original-adjacent pairs only)", y=1.04)
fig.tight_layout()
fig_path = FIGURES / "phase3_task2b_centered_acf.png"
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig_path}")

(LOGS / "task2b_console_log.txt").write_text("\n".join(log_lines))
log("\n=== DONE (Task 2B) ===")
