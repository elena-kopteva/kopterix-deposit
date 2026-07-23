"""
Kopterix Phase 3, Task 2 -- four independent analyses:
  A. Cross-run layer-ordering consistency test (pre-registered April->May)
  B. Orthogonality null for layer residuals (is near-pi/2 just the HD null?)
  C. Memory timescale estimation (tau in hours) -- the headline analysis
  D. Changepoint detection (PELT/rbf) on detrended phi_t and D_t

Conventions inherited from Task 1 / Phase 2:
  - Window [2026-04-01, 2026-06-01) UTC, half-open.
  - Seed 42 (single rng stream per stochastic section, documented inline).
  - kopterix_state.csv: Timestamp (ISO+tz), MeanEmbedding (384-d), LayerCentroids
    (dict surface/mid/residue, each 384-d), UnigramCounts (top-200 + __total__).
  - real_clean.csv: per-run observations (phi_t, D_t, centroid_dist_*), Timestamp_UTC
    has a trailing " UTC" stripped before parsing.
  - Cosine distance = 1 - cos(a,b); NaN if either norm < 1e-12 (matches addendum).
  - Raw inputs are READ-ONLY; all outputs are new files in phase3/.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import ruptures as rpt

BASE    = Path(__file__).resolve().parents[1]
TABLES  = BASE / "phase3" / "tables"
FIGURES = BASE / "phase3" / "figures"
LOGS    = BASE / "phase3" / "logs"
for d in (TABLES, FIGURES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
SEED  = 42
LAYERS = ["surface", "mid", "residue"]

log_lines = []
def log(msg):
    print(msg); log_lines.append(str(msg))

log("=== Kopterix Phase 3 Task 2 ===")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
state = pd.read_csv(BASE / "kopterix_state.csv")
state["ts"] = pd.to_datetime(state["Timestamp"], utc=True)
state = state.sort_values("ts").reset_index(drop=True)
state["month"] = state["ts"].dt.strftime("%Y-%m")

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

def parse_unigram(s):
    if pd.isna(s): return None
    try:
        d = json.loads(s); d.pop("__total__", None); return d
    except Exception:
        return None

state["mean_emb"] = state["MeanEmbedding"].apply(parse_vec)
state["lc"]       = state["LayerCentroids"].apply(parse_layers)
state["unigram"]  = state["UnigramCounts"].apply(parse_unigram)

in_win = (state["ts"] >= WINDOW_START) & (state["ts"] < WINDOW_END)
sw = state[in_win].copy().reset_index(drop=True)               # state in-window
lc_ok = sw["lc"].notna() & sw["mean_emb"].notna()
slc = sw[lc_ok].copy().reset_index(drop=True)                  # valid layer rows
log(f"state rows in window: {len(sw)}; with all-3-valid layers + mean_emb: {len(slc)}")
log(f"  by month: {slc['month'].value_counts().to_dict()}")

real = pd.read_csv(BASE / "analysis_intermediate" / "real_clean.csv")
real["ts"] = pd.to_datetime(real["Timestamp_UTC"].str.replace(" UTC", "", regex=False), utc=True)
real = real.sort_values("ts").reset_index(drop=True)
assert (real["ts"] >= WINDOW_START).all() and (real["ts"] < WINDOW_END).all()
log(f"real_clean rows in window: {len(real)}")

def cos_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))

def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return np.nan
    c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(c))

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: cross-run layer-ordering consistency
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== ANALYSIS A: layer-ordering consistency ===")
ORD_LABELS = {  # ascending order of (d_SM, d_MR, d_SR) -> readable label
}
rows = []
for _, r in slc.iterrows():
    d = r["lc"]
    dSM = cos_dist(d["surface"], d["mid"])
    dMR = cos_dist(d["mid"], d["residue"])
    dSR = cos_dist(d["surface"], d["residue"])
    trip = {"SM": dSM, "MR": dMR, "SR": dSR}
    order = sorted(trip, key=lambda k: trip[k])           # ascending
    label = "<".join(order)                                # e.g. "SR<SM<MR"
    rows.append({"ts": r["ts"], "month": r["month"],
                 "d_SM": dSM, "d_MR": dMR, "d_SR": dSR, "ordering": label})
ord_df = pd.DataFrame(rows)

ALL_ORDERS = ["<".join(p) for p in __import__("itertools").permutations(["SM","MR","SR"])]
def freq_table(sub):
    vc = sub["ordering"].value_counts()
    return {o: int(vc.get(o, 0)) for o in ALL_ORDERS}

overall_freq = freq_table(ord_df)
apr_df = ord_df[ord_df["month"] == "2026-04"]
may_df = ord_df[ord_df["month"] == "2026-05"]
apr_freq = freq_table(apr_df)
may_freq = freq_table(may_df)

freq_rows = []
for o in ALL_ORDERS:
    freq_rows.append({"ordering": o,
                      "n_overall": overall_freq[o], "n_april": apr_freq[o], "n_may": may_freq[o],
                      "frac_overall": overall_freq[o]/len(ord_df),
                      "frac_april": apr_freq[o]/len(apr_df) if len(apr_df) else np.nan,
                      "frac_may": may_freq[o]/len(may_df) if len(may_df) else np.nan})
freq_out = pd.DataFrame(freq_rows).sort_values("n_overall", ascending=False)
log("Ordering frequencies (overall):")
for _, fr in freq_out.iterrows():
    log(f"  {fr['ordering']:9s} overall={fr['n_overall']:3d} ({fr['frac_overall']:.3f})  "
        f"apr={fr['n_april']:3d}  may={fr['n_may']:3d}")

# Pre-registration: modal ordering from APRIL ONLY
april_modal = max(apr_freq, key=apr_freq.get)
log(f"\nPre-registered modal ordering (APRIL only): {april_modal} "
    f"(april n={apr_freq[april_modal]}/{len(apr_df)})")

# Test on MAY: one-sided exact binomial vs null p0 = 1/6
p0 = 1.0/6.0
may_k = may_freq[april_modal]; may_n = len(may_df)
bt = stats.binomtest(may_k, may_n, p0, alternative="greater")
log(f"MAY test of April modal ordering '{april_modal}': k={may_k}/{may_n}, "
    f"observed frac={may_k/may_n:.4f}, null p0=1/6={p0:.4f}")
log(f"  one-sided exact binomial (greater) p = {bt.pvalue:.4g}")

# Secondary: chi-square GoF of full MAY ordering distribution vs uniform 1/6
may_counts = np.array([may_freq[o] for o in ALL_ORDERS], dtype=float)
exp_counts = np.full(6, may_n/6.0)
chi2, chi2_p = stats.chisquare(may_counts, exp_counts)
log(f"Chi-square GoF (May full distribution vs uniform 1/6): chi2={chi2:.4f}, "
    f"dof=5, p={chi2_p:.4g}")

# Sensitivity: every 4th run (~1/day) to reduce autocorrelation dependence
sub4 = ord_df.iloc[::4].reset_index(drop=True)
apr4 = sub4[sub4["month"]=="2026-04"]; may4 = sub4[sub4["month"]=="2026-05"]
apr4_freq = freq_table(apr4); may4_freq = freq_table(may4)
april_modal_4 = max(apr4_freq, key=apr4_freq.get)
may4_k = may4_freq[april_modal]; may4_n = len(may4)      # test SAME pre-registered ordering
bt4 = stats.binomtest(may4_k, may4_n, p0, alternative="greater")
may4_counts = np.array([may4_freq[o] for o in ALL_ORDERS], dtype=float)
chi2_4, chi2_4p = stats.chisquare(may4_counts, np.full(6, may4_n/6.0))
log(f"\nSensitivity (every 4th run, ~1/day): full n={len(sub4)}, "
    f"april n={len(apr4)} (modal={april_modal_4}), may n={may4_n}")
log(f"  MAY pre-registered '{april_modal}': k={may4_k}/{may4_n}, "
    f"binomial(greater) p={bt4.pvalue:.4g}; chi2={chi2_4:.3f} p={chi2_4p:.4g}")

# write A outputs
freq_path = TABLES / "phase3_layer_ordering.csv"
with open(freq_path, "w") as f:
    f.write("# Cross-run layer-ordering frequencies. ordering = ascending rank of the "
            "three inter-layer cosine distances d_SM,d_MR,d_SR (e.g. 'SR<SM<MR' means "
            "d_SR<d_SM<d_MR). One row per state run with all 3 valid layer centroids "
            f"in window [2026-04-01,2026-06-01) UTC; n={len(ord_df)} "
            f"({len(apr_df)} April + {len(may_df)} May).\n")
    freq_out.to_csv(f, index=False)
ord_df.to_csv(TABLES / "phase3_layer_ordering_perrun.csv", index=False)
log(f"Wrote {freq_path} and per-run orderings")

testA = pd.DataFrame([
    {"test":"April-modal-ordering","value":april_modal,"april_k":apr_freq[april_modal],
     "april_n":len(apr_df),"may_k":may_k,"may_n":may_n,"null_p0":p0,
     "stat":np.nan,"pvalue":bt.pvalue,
     "detail":"one-sided exact binomial (May modal occurrence > 1/6)"},
    {"test":"May-chisquare-GoF-uniform","value":"all-6-orderings","april_k":np.nan,
     "april_n":np.nan,"may_k":may_n,"may_n":may_n,"null_p0":p0,
     "stat":chi2,"pvalue":chi2_p,"detail":"chi-square GoF May distribution vs uniform 1/6, dof=5"},
    {"test":"April-modal-ordering (every-4th-run sens.)","value":april_modal,
     "april_k":apr4_freq[april_modal],"april_n":len(apr4),"may_k":may4_k,"may_n":may4_n,
     "null_p0":p0,"stat":np.nan,"pvalue":bt4.pvalue,
     "detail":"autocorrelation sensitivity: every 4th run (~1/day); SAME pre-registered ordering"},
    {"test":"May-chisquare-GoF (every-4th-run sens.)","value":"all-6-orderings",
     "april_k":np.nan,"april_n":np.nan,"may_k":may4_n,"may_n":may4_n,"null_p0":p0,
     "stat":chi2_4,"pvalue":chi2_4p,"detail":"chi-square GoF May (every 4th run) vs uniform, dof=5"},
])
with open(TABLES / "phase3_layer_ordering_test.csv", "w") as f:
    f.write("# Pre-registered test: modal ordering identified on APRIL rows only, tested "
            "on MAY rows only. Primary = one-sided exact binomial vs null 1/6. Secondary "
            "= chi-square GoF of full May distribution vs uniform. CAVEAT: consecutive "
            "runs are autocorrelated so effective N < row count; every-4th-run rows give "
            "a lower-dependence sensitivity check (~1 run/day).\n")
    testA.to_csv(f, index=False)
log("Wrote phase3_layer_ordering_test.csv")

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: orthogonality null for layer residuals
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== ANALYSIS B: residual orthogonality null ===")
PAIRS = [("surface","mid","SM"), ("mid","residue","MR"), ("surface","residue","SR")]
res_list = []   # per timestamp: dict of residual vectors + norms
for _, r in slc.iterrows():
    me = r["mean_emb"]; d = r["lc"]
    resid = {l: d[l] - me for l in LAYERS}
    res_list.append(resid)
n_ts = len(res_list)

# observed angles
obs_ang = {tag: [] for _,_,tag in PAIRS}
for resid in res_list:
    for a,b,tag in PAIRS:
        obs_ang[tag].append(angle_between(resid[a], resid[b]))
obs_ang = {k: np.array(v) for k,v in obs_ang.items()}
obs_all = np.concatenate([obs_ang[t] for _,_,t in PAIRS])
log(f"Observed residual angles (rad), n={n_ts} timestamps x 3 pairs:")
for _,_,tag in PAIRS:
    log(f"  {tag}: mean={np.nanmean(obs_ang[tag]):.4f} std={np.nanstd(obs_ang[tag]):.4f}")
log(f"  pi/2 = {np.pi/2:.4f}")

# residual norms per timestamp per layer (for Null 1 rescaling)
norms = np.array([[np.linalg.norm(resid[l]) for l in LAYERS] for resid in res_list])  # n_ts x 3

rng = np.random.default_rng(SEED)
N_NULL = 200
DIM = 384

# Null 1: Gaussian random vectors rescaled to observed residual norms at each ts
null1 = {tag: [] for _,_,tag in PAIRS}
for _ in range(N_NULL):
    for i in range(n_ts):
        g = rng.standard_normal((3, DIM))
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * norms[i][:, None]
        rv = {LAYERS[j]: g[j] for j in range(3)}
        for a,b,tag in PAIRS:
            null1[tag].append(angle_between(rv[a], rv[b]))
null1 = {k: np.array(v) for k,v in null1.items()}
null1_all = np.concatenate([null1[t] for _,_,t in PAIRS])

# Null 2: permute layer labels across timestamps (each layer's residual vectors are
# reassigned to random timestamps within that layer), 200 permutations.
resid_by_layer = {l: np.array([res_list[i][l] for i in range(n_ts)]) for l in LAYERS}
null2 = {tag: [] for _,_,tag in PAIRS}
for _ in range(N_NULL):
    perm = {l: rng.permutation(n_ts) for l in LAYERS}
    for i in range(n_ts):
        rv = {l: resid_by_layer[l][perm[l][i]] for l in LAYERS}
        for a,b,tag in PAIRS:
            null2[tag].append(angle_between(rv[a], rv[b]))
null2 = {k: np.array(v) for k,v in null2.items()}
null2_all = np.concatenate([null2[t] for _,_,t in PAIRS])

def summ(x):
    x = x[~np.isnan(x)]
    return np.mean(x), np.std(x)

brows = []
for label, nullarr in [("Null1_gaussian_rescaled", null1), ("Null2_layer_label_perm", null2)]:
    for a,b,tag in PAIRS + [("","","ALL")]:
        if tag == "ALL":
            o = obs_all; nu = (null1_all if "Null1" in label else null2_all)
        else:
            o = obs_ang[tag]; nu = nullarr[tag]
        om, osd = summ(o); nm, nsd = summ(nu)
        ks, ksp = stats.ks_2samp(o[~np.isnan(o)], nu[~np.isnan(nu)])
        brows.append({"null": label, "pair": tag, "n_obs": int(np.sum(~np.isnan(o))),
                      "n_null": int(np.sum(~np.isnan(nu))),
                      "obs_mean": om, "obs_std": osd, "null_mean": nm, "null_std": nsd,
                      "mean_diff_obs_minus_null": om-nm, "std_ratio_obs_over_null": osd/nsd,
                      "ks_stat": ks, "ks_p": ksp})
        log(f"  [{label}] {tag}: obs mean={om:.4f} std={osd:.4f} | null mean={nm:.4f} "
            f"std={nsd:.4f} | dmean={om-nm:+.4f} std_ratio={osd/nsd:.3f} KS={ks:.3f} p={ksp:.3g}")
bdf = pd.DataFrame(brows)

# Decision: tighter (std_ratio<1) or displaced (mean shifted) vs the HD null?
all_rows = bdf[bdf["pair"]=="ALL"]
tighter = (all_rows["std_ratio_obs_over_null"] < 0.9).any()
displaced = (all_rows["mean_diff_obs_minus_null"].abs() > 0.05).any()
ks_sig = (all_rows["ks_p"] < 0.05).all()
parts = []
for _, rr in all_rows.iterrows():
    if rr["std_ratio_obs_over_null"] < 0.9:
        parts.append(f"TIGHTER vs {rr['null']} (std ratio {rr['std_ratio_obs_over_null']:.2f})")
    if abs(rr["mean_diff_obs_minus_null"]) > 0.05:
        parts.append(f"DISPLACED vs {rr['null']} (mean {rr['mean_diff_obs_minus_null']:+.3f} rad)")
if tighter or displaced:
    conclusion = ("observed angles deviate from the null in: " + "; ".join(parts)
                  + f" (KS p<0.05 for all-pairs vs both nulls: {ks_sig}).")
else:
    conclusion = "observed angles are indistinguishable from the high-dimensional null"
log(f"\nCONCLUSION (B): {conclusion}")

bpath = TABLES / "phase3_residual_angle_null.csv"
with open(bpath, "w") as f:
    f.write("# Residual orthogonality null. Residuals r_l(t)=c_l(t)-MeanEmbedding(t). "
            f"Observed pairwise angles (rad) over n={n_ts} timestamps with 3 valid layers, "
            "vs two nulls (200 reps each, seed 42): Null1 = Gaussian random 384-d vectors "
            "rescaled to each timestamp's observed residual norms; Null2 = layer labels "
            "permuted across timestamps (vectors kept, reassigned within layer). Question: "
            "is the observed distribution TIGHTER or DISPLACED vs the random-vector null, "
            "NOT whether angles are near pi/2 (pi/2=1.5708).\n")
    f.write(f"# CONCLUSION: {conclusion}\n")
    bdf.to_csv(f, index=False)
log(f"Wrote {bpath}")

# stash observed/null arrays for figure
np.savez(LOGS / "B_angle_arrays.npz",
         obs_all=obs_all, null1_all=null1_all, null2_all=null2_all,
         **{f"obs_{t}":obs_ang[t] for _,_,t in PAIRS})

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS C: memory timescale estimation
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== ANALYSIS C: memory timescale ===")
DT = 6.0  # hours, regular grid spacing
TOL = pd.Timedelta("3h")

def resample_grid(ts, vals):
    """Nearest-neighbor onto regular 6h grid within 3h tolerance; else NaN."""
    ts = pd.DatetimeIndex(ts); vals = np.asarray(vals, float)
    ok = ~np.isnan(vals)
    ts, vals = ts[ok], vals[ok]
    order = np.argsort(ts.values); ts, vals = ts[order], vals[order]
    t0, t1 = ts[0], ts[-1]
    n_grid = int(np.floor((t1 - t0).total_seconds()/3600.0/DT)) + 1
    grid = t0 + pd.to_timedelta(np.arange(n_grid)*DT, unit="h")
    out = np.full(n_grid, np.nan)
    ts_ns = ts.view("int64").astype(float)
    for i, g in enumerate(grid):
        gi = float(pd.Timestamp(g).value)
        j = int(np.argmin(np.abs(ts_ns - gi)))
        if abs(ts_ns[j]-gi) <= TOL.value:
            out[i] = vals[j]
    return grid, out

def nan_acf(x, maxlag):
    """Pairwise-complete autocorrelation (Pearson) at lags 0..maxlag."""
    x = np.asarray(x, float)
    rho = np.full(maxlag+1, np.nan); rho[0] = 1.0
    for k in range(1, maxlag+1):
        a, b = x[:-k], x[k:]
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() >= 5 and np.std(a[m])>0 and np.std(b[m])>0:
            rho[k] = np.corrcoef(a[m], b[m])[0,1]
    return rho

def fit_tau_exp(rho, dt, lags=range(1,17)):
    ks = np.array([k for k in lags if k < len(rho) and not np.isnan(rho[k])])
    ys = np.array([rho[k] for k in ks])
    if len(ks) < 3: return np.nan, np.nan
    def model(k, tau): return np.exp(-k*dt/tau)
    try:
        popt,_ = curve_fit(model, ks, ys, p0=[24.0],
                           bounds=(0.1, 1e5), maxfev=10000)
        resid = ys - model(ks, popt[0])
        ss = np.sum(resid**2); sstot = np.sum((ys-ys.mean())**2)
        r2 = 1 - ss/sstot if sstot>0 else np.nan
        return float(popt[0]), float(r2)
    except Exception:
        return np.nan, np.nan

def tau_ar1(rho1, dt):
    if rho1 is None or np.isnan(rho1) or rho1 <= 0: return np.nan
    return float(-dt/np.log(rho1))

def linear_detrend(x):
    x = np.asarray(x, float); idx = np.arange(len(x)); m = ~np.isnan(x)
    if m.sum() < 3: return x.copy()
    b1, b0 = np.polyfit(idx[m], x[m], 1)
    return x - (b0 + b1*idx)

def adjacency_block_boot_tau(x, dt, block=24, n_boot=5000, rng=None, lags=range(1,17)):
    """Adjacency-preserving NON-CIRCULAR moving-block bootstrap CI for exp-fit tau
    and AR(1) tau.

    CORRECTED aggregate-scalar method (supersedes the former block=8 CIRCULAR
    bootstrap, which could not preserve dependence at lags 8-16 and manufactured
    artificial lagged pairs across sampled block boundaries). For each replicate:
      * draw ceil(n/block) contiguous ORIGINAL blocks x[s:s+block] with replacement,
        each start s uniform on 0..n-block INCLUSIVE -- NO wrap-around;
      * at each lag k, gather lagged pairs (blk[i], blk[i+k]) that lie WITHIN the
        same sampled original block only -- never across block boundaries;
      * pool the finite first members and finite second members from all eligible
        within-block pairs and take ONE pairwise-complete Pearson correlation per
        lag (block-level correlations are NOT averaged);
      * fit rho(k)=exp(-k*dt/tau) over lags 1-16 and take AR(1) tau from lag 1;
      * discard only failed / non-finite estimates.
    """
    if rng is None: rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    n = len(x); maxlag = max(lags)
    n_blocks = int(np.ceil(n/block))
    max_start = n - block            # inclusive upper bound for a block start
    if max_start < 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    taus_exp, taus_ar1 = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)  # 0..n-block inclusive
        blocks = [x[s:s+block] for s in starts]
        rho = np.full(maxlag + 1, np.nan); rho[0] = 1.0
        for k in range(1, maxlag + 1):
            a_parts, b_parts = [], []
            for blk in blocks:
                if len(blk) > k:                     # within-block pairs only
                    a_parts.append(blk[:-k]); b_parts.append(blk[k:])
            if a_parts:
                a = np.concatenate(a_parts); b = np.concatenate(b_parts)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= 5 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                    rho[k] = np.corrcoef(a[m], b[m])[0, 1]
        te,_ = fit_tau_exp(rho, dt, lags)
        ta = tau_ar1(rho[1], dt)
        if not np.isnan(te): taus_exp.append(te)
        if not np.isnan(ta): taus_ar1.append(ta)
    def ci(a):
        a = np.array(a)
        return (np.nanpercentile(a,2.5), np.nanpercentile(a,97.5)) if len(a)>10 else (np.nan,np.nan)
    return ci(taus_exp), ci(taus_ar1)

mem_rows = []
def estimate_series(name, ts, vals, detrend_also=True):
    grid, g = resample_grid(ts, vals)
    n_fill = int(np.sum(~np.isnan(g))); n_miss = int(np.sum(np.isnan(g)))
    log(f"\n[{name}] grid pts={len(g)} filled={n_fill} missing={n_miss} "
        f"(dt={DT}h, tol=3h)")
    out = []
    for detr, tag in ([(False,"raw")] + ([(True,"detrended")] if detrend_also else [])):
        series = linear_detrend(g) if detr else g
        rho = nan_acf(series, 28)
        te, r2 = fit_tau_exp(rho, DT)
        ta = tau_ar1(rho[1], DT)
        rng_b = np.random.default_rng(SEED+ (1 if detr else 0))
        (ci_e, ci_a) = adjacency_block_boot_tau(series, DT, block=24, n_boot=5000, rng=rng_b)
        log(f"  {tag}: rho1={rho[1]:.4f} tau_exp={te:.2f}h (R2={r2:.3f}) "
            f"CI[{ci_e[0]:.2f},{ci_e[1]:.2f}] | tau_ar1={ta:.2f}h CI[{ci_a[0]:.2f},{ci_a[1]:.2f}]")
        for method, tau, ci in [("exp_fit_lags1-16", te, ci_e), ("AR1", ta, ci_a)]:
            out.append({"series": name, "detrended": detr, "method": method,
                        "dt_hours": DT, "n_grid": len(g), "n_filled": n_fill,
                        "n_missing": n_miss, "rho1": rho[1], "tau_hours": tau,
                        "ci95_lo": ci[0], "ci95_hi": ci[1], "exp_R2": r2 if "exp" in method else np.nan,
                        "tau_vs_24h": tau/24.0 if not np.isnan(tau) else np.nan,
                        "exceeds_24h": bool(tau>24) if not np.isnan(tau) else False})
    return grid, g, out

# C1-C3: phi_t
grid_phi, g_phi, o = estimate_series("phi_t", real["ts"], real["phi_t"].values); mem_rows += o
# C4: D_t
grid_D, g_D, o = estimate_series("D_t", real["ts"], real["D_t"].values); mem_rows += o

# C4: consecutive-step lexical cosine distance series (computed from state unigrams)
def cosine_dist_counts(d1, d2):
    vocab = sorted(set(d1.keys()) | set(d2.keys()))
    if not vocab: return np.nan
    v1 = np.array([d1.get(w,0) for w in vocab], float)
    v2 = np.array([d2.get(w,0) for w in vocab], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-12 or n2<1e-12: return np.nan
    return float(1.0 - np.dot(v1,v2)/(n1*n2))

su = sw[sw["unigram"].notna()].copy().reset_index(drop=True)
lex_ts, lex_val = [], []
for i in range(len(su)-1):
    lex_val.append(cosine_dist_counts(su.iloc[i]["unigram"], su.iloc[i+1]["unigram"]))
    lex_ts.append(su.iloc[i+1]["ts"])         # assign to the 'to' run timestamp
grid_lex, g_lex, o = estimate_series("lexical_cos_step", pd.DatetimeIndex(lex_ts),
                                     np.array(lex_val)); mem_rows += o

# C5: per-layer memory -- cos sim between c_l(t) and c_l(t+k), original-adjacent pairs only
log("\n[per-layer memory] cos-sim decay, k=1..16 over consecutive valid rows "
    "(no bridging gaps>12h)")
slc_sorted = slc.sort_values("ts").reset_index(drop=True)
gaps = slc_sorted["ts"].diff().dt.total_seconds().values/3600.0  # hours, gaps[0]=nan
median_gap = float(np.nanmedian(gaps[1:]))
BIG = 12.0  # hours: do not bridge gaps larger than this
def layer_decay_tau(layer):
    vecs = np.array([slc_sorted.iloc[i]["lc"][layer] for i in range(len(slc_sorted))])
    n = len(vecs); sims = {}
    for k in range(1, 17):
        vals = []
        for i in range(n-k):
            seg = gaps[i+1:i+k+1]                 # the k consecutive gaps spanned
            if np.any(np.isnan(seg)) or np.any(seg > BIG):
                continue
            vals.append(1 - cos_dist(vecs[i], vecs[i+k]))  # cosine similarity
        sims[k] = (np.mean(vals) if vals else np.nan, len(vals))
    ks = np.array([k for k in range(1,17) if not np.isnan(sims[k][0])])
    ys = np.array([sims[k][0] for k in ks])
    dt_step = median_gap
    # fit sim(k) = A*exp(-k*dt/tau) + C
    def model(k, A, tau, C): return A*np.exp(-k*dt_step/tau) + C
    tau = np.nan; r2 = np.nan
    try:
        p0 = [ys[0]-ys[-1], 24.0, ys[-1]]
        popt,_ = curve_fit(model, ks, ys, p0=p0,
                           bounds=([0,0.1,-1],[2,1e5,1]), maxfev=20000)
        tau = float(popt[1])
        resid = ys-model(ks,*popt); ss=np.sum(resid**2); st=np.sum((ys-ys.mean())**2)
        r2 = 1-ss/st if st>0 else np.nan
    except Exception as e:
        log(f"    {layer}: fit failed {e}")
    return tau, r2, dt_step, sims, ks, ys

layer_taus = {}
# bootstrap CI for per-layer tau via block bootstrap over the row sequence
def layer_boot_ci(layer, n_boot=500):
    vecs = np.array([slc_sorted.iloc[i]["lc"][layer] for i in range(len(slc_sorted))])
    n=len(vecs); rng_l=np.random.default_rng(SEED+hash(layer)%1000)
    block=8; n_blocks=int(np.ceil(n/block)); taus=[]
    dt_step=median_gap; idx_all=np.arange(n)
    for _ in range(n_boot):
        starts=rng_l.integers(0,n,size=n_blocks)
        idx=np.concatenate([np.arange(s,s+block)%n for s in starts])[:n]
        vv=vecs[idx]
        sims=[]; ks=[]
        for k in range(1,17):
            vals=[1-cos_dist(vv[i],vv[i+k]) for i in range(n-k)]
            sims.append(np.mean(vals)); ks.append(k)
        ks=np.array(ks); ys=np.array(sims)
        def model(k,A,tau,C): return A*np.exp(-k*dt_step/tau)+C
        try:
            popt,_=curve_fit(model,ks,ys,p0=[ys[0]-ys[-1],24.0,ys[-1]],
                             bounds=([0,0.1,-1],[2,1e5,1]),maxfev=20000)
            taus.append(popt[1])
        except Exception: pass
    taus=np.array(taus)
    return (np.nanpercentile(taus,2.5),np.nanpercentile(taus,97.5)) if len(taus)>10 else (np.nan,np.nan)

layer_sim_store = {}
for layer in LAYERS:
    tau, r2, dt_step, sims, ks, ys = layer_decay_tau(layer)
    ci = layer_boot_ci(layer)
    layer_taus[layer] = (tau, ci, dt_step)
    layer_sim_store[layer] = (ks, ys)
    log(f"  {layer}: tau={tau:.2f}h (R2={r2:.3f}) CI[{ci[0]:.2f},{ci[1]:.2f}] "
        f"dt_step(median gap)={dt_step:.3f}h  vs24h={'EXCEEDS' if tau>24 else 'below'}")
    mem_rows.append({"series": f"layer_{layer}_cosine_decay", "detrended": False,
                     "method": "exp_fit_AexpC_k1-16", "dt_hours": dt_step,
                     "n_grid": len(slc_sorted), "n_filled": len(slc_sorted), "n_missing": 0,
                     "rho1": ys[0] if len(ys) else np.nan, "tau_hours": tau,
                     "ci95_lo": ci[0], "ci95_hi": ci[1], "exp_R2": r2,
                     "tau_vs_24h": tau/24.0 if not np.isnan(tau) else np.nan,
                     "exceeds_24h": bool(tau>24)})

tau_surf = layer_taus["surface"][0]; tau_res = layer_taus["residue"][0]
log(f"\n  tau_residue ({tau_res:.2f}h) > tau_surface ({tau_surf:.2f}h)? "
    f"{tau_res > tau_surf}")

mem_df = pd.DataFrame(mem_rows)
mpath = TABLES / "phase3_memory_timescales.csv"
with open(mpath, "w") as f:
    f.write("# Memory timescale tau (hours). phi_t/D_t/lexical_cos_step resampled to a "
            "regular 6h grid (nearest obs within 3h tolerance; gaps beyond tolerance left "
            "NaN, never interpolated). ACF computed pairwise-complete to lag 28; tau from "
            "(i) NLS exp fit rho(k)=exp(-k*dt/tau) on lags 1-16 and (ii) AR(1) "
            "tau=-dt/ln(rho1). 95% CI for the aggregate scalar rows (phi_t, D_t, "
            "lexical_cos_step) = adjacency-preserving NON-CIRCULAR moving-block bootstrap, "
            "block=24 grid pts (6 days), 5000 resamples, raw seed 42 / detrended seed 43, "
            "percentile 2.5/97.5, exp fit over lags 1-16; lagged pairs are formed only "
            "WITHIN a sampled original block (no cross-boundary pairs, no wrap-around). "
            "Detrended rows use linear-detrended series (the "
            "defensible estimate, given phi_t's strong negative trend). Per-layer rows fit "
            "sim(k)=A*exp(-k*dt/tau)+C on consecutive valid rows (no bridging gaps>12h); "
            "dt = median consecutive gap. tau_vs_24h = tau/24; exceeds_24h flags persistence "
            "outliving the deepest 24h sampled feed-age window (NOT a proof of relay).\n")
    mem_df.to_csv(f, index=False)
log(f"Wrote {mpath}")

# stash phi/D grids and layer sims for figures
np.savez(LOGS / "C_series.npz",
         grid_phi=grid_phi.view("int64"), g_phi=g_phi,
         grid_D=grid_D.view("int64"), g_D=g_D,
         grid_lex=grid_lex.view("int64"), g_lex=g_lex,
         **{f"sim_{l}_k":layer_sim_store[l][0] for l in LAYERS},
         **{f"sim_{l}_y":layer_sim_store[l][1] for l in LAYERS})

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS D: changepoint detection
# ═════════════════════════════════════════════════════════════════════════════
log("\n=== ANALYSIS D: changepoint detection (PELT/rbf) ===")
def detrend_series(ts, vals):
    df = pd.DataFrame({"ts": ts, "v": vals}).dropna().sort_values("ts").reset_index(drop=True)
    idx = np.arange(len(df))
    b1,b0 = np.polyfit(idx, df["v"].values, 1)
    df["v_detr"] = df["v"].values - (b0+b1*idx)
    return df

def pelt_bic(df, name):
    sig = df["v_detr"].values.reshape(-1,1)
    n = len(sig)
    algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(sig)
    # sweep penalties (geometric), record K and total cost
    pens = np.geomspace(0.2, 200, 60)
    seen = {}
    for pen in pens:
        bkps = algo.predict(pen=pen)
        K = len(bkps)-1
        if K not in seen:
            cost = algo.cost.sum_of_costs(bkps)
            bic = n*np.log(cost/n + 1e-12) + (K+1)*np.log(n)
            seen[K] = {"pen": pen, "cost": cost, "bic": bic, "bkps": bkps}
    bestK = min(seen, key=lambda k: seen[k]["bic"])
    pen_star = seen[bestK]["pen"]
    log(f"[{name}] n={n} BIC-selected K={bestK} changepoints at pen~{pen_star:.3g}")
    # sensitivity over 3x penalty range
    sens = {}
    for mult, ml in [(1/np.sqrt(3),"pen/sqrt3"),(1.0,"pen"),(np.sqrt(3),"pen*sqrt3")]:
        bkps = algo.predict(pen=pen_star*mult)
        cps = [df.iloc[b-1]["ts"] for b in bkps[:-1]]  # b is end index (exclusive); cp at b-1
        sens[ml] = (pen_star*mult, len(cps), cps)
        log(f"    {ml} (pen={pen_star*mult:.3g}): {len(cps)} cps")
    return pen_star, bestK, seen[bestK]["bkps"], df, sens

df_phi = detrend_series(real["ts"], real["phi_t"].values)
df_D   = detrend_series(real["ts"], real["D_t"].values)
pen_phi, K_phi, bkps_phi, dfp, sens_phi = pelt_bic(df_phi, "phi_t(detrended)")
pen_D,   K_D,   bkps_D,   dfd, sens_D   = pelt_bic(df_D,   "D_t(detrended)")

cp_rows = []
def record_cps(name, df, bkps, pen, sens):
    cps_main = [df.iloc[b-1]["ts"] for b in bkps[:-1]]
    for cp in cps_main:
        cp_rows.append({"series": name, "changepoint_utc": str(cp),
                        "penalty_BIC": pen, "n_cp_at_BIC": len(cps_main),
                        "n_cp_pen_div_sqrt3": sens["pen/sqrt3"][1],
                        "n_cp_pen": sens["pen"][1],
                        "n_cp_pen_mul_sqrt3": sens["pen*sqrt3"][1]})
    return cps_main
cps_phi = record_cps("phi_t_detrended", dfp, bkps_phi, pen_phi, sens_phi)
cps_D   = record_cps("D_t_detrended",   dfd, bkps_D,   pen_D,   sens_D)
cp_df = pd.DataFrame(cp_rows)

# robust-z outliers on D_t (median/MAD, thresh 3.5) -> reproduce "10 outliers, 7 consecutive"
dvals = real[["ts","D_t"]].dropna().sort_values("ts").reset_index(drop=True)
med = dvals["D_t"].median(); mad = stats.median_abs_deviation(dvals["D_t"], scale="normal")
dvals["rz"] = (dvals["D_t"]-med)/mad
dvals["outlier"] = dvals["rz"].abs() > 3.5
out_idx = np.where(dvals["outlier"].values)[0]
# consecutive clusters
clusters = []
if len(out_idx):
    cur=[out_idx[0]]
    for j in out_idx[1:]:
        if j==cur[-1]+1: cur.append(j)
        else: clusters.append(cur); cur=[j]
    clusters.append(cur)
n_consec = sum(len(c) for c in clusters if len(c)>=2)
log(f"\nD_t robust-z (|z|>3.5): {int(dvals['outlier'].sum())} outliers, "
    f"{sum(1 for c in clusters if len(c)>=2)} consecutive runs covering {n_consec} of them")
cluster_spans = [(str(dvals.iloc[c[0]]["ts"]), str(dvals.iloc[c[-1]]["ts"]), len(c))
                 for c in clusters]
# does each changepoint coincide with a D_t outlier cluster (within 12h)?
def near_cluster(cp):
    cp = pd.Timestamp(cp)
    for c in clusters:
        t0=pd.Timestamp(dvals.iloc[c[0]]["ts"]); t1=pd.Timestamp(dvals.iloc[c[-1]]["ts"])
        if (t0-pd.Timedelta("12h")) <= cp <= (t1+pd.Timedelta("12h")):
            return f"{t0}..{t1} (len {len(c)})"
    return ""
cp_df["near_Dt_outlier_cluster"] = cp_df["changepoint_utc"].apply(near_cluster)

with open(TABLES / "phase3_changepoints.csv", "w") as f:
    f.write("# PELT changepoints (ruptures, model='rbf', min_size=3). Penalty selected by "
            "BIC-style sweep over K; sensitivity reported over a 3x penalty range "
            "(pen/sqrt3, pen, pen*sqrt3). Run on LINEARLY DETRENDED phi_t and D_t "
            "(observed series, NaN dropped, time-ordered). changepoint_utc = timestamp of "
            "the last run of each detected segment. near_Dt_outlier_cluster flags coincidence "
            "(within 12h) with a robust-z (|z|>3.5, median/MAD) D_t outlier cluster.\n")
    cp_df.to_csv(f, index=False)
log("Wrote phase3_changepoints.csv")

# ── D3: markdown log cross-reference ──
log("\n[D cross-ref] parsing monthly observation logs")
obs_entries = []
for mf in ["kopterix_log_2026-04.md", "kopterix_log_2026-05.md"]:
    p = BASE / mf
    if not p.exists(): continue
    text = p.read_text(encoding="utf-8")
    blocks = re.split(r"\n## obs :: ", text)
    for blk in blocks[1:]:
        mh = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*UTC", blk)
        if not mh: continue
        ts = pd.Timestamp(mh.group(1), tz="UTC")
        mood = re.search(r"\*\*surface mood:\*\*\s*(.+)", blk)
        flag = re.search(r"\*\*anomaly flag:\*\*\s*(\w+)", blk)
        obs_entries.append({"ts": ts,
                            "surface_mood": mood.group(1).strip() if mood else "",
                            "anomaly_flag": flag.group(1).strip() if flag else ""})
obs_log = pd.DataFrame(obs_entries).sort_values("ts").reset_index(drop=True)
log(f"  parsed {len(obs_log)} observation-log entries")

xref_rows = []
all_cps = [("phi_t_detrended", cp) for cp in cps_phi] + [("D_t_detrended", cp) for cp in cps_D]
for series, cp in all_cps:
    cp = pd.Timestamp(cp)
    win = obs_log[(obs_log["ts"]>=cp-pd.Timedelta("12h")) & (obs_log["ts"]<=cp+pd.Timedelta("12h"))]
    if len(win)==0:
        xref_rows.append({"series":series,"changepoint_utc":str(cp),"log_ts":"",
                          "delta_hours":np.nan,"surface_mood":"(no log entry within +/-12h)",
                          "anomaly_flag":""})
    for _, w in win.iterrows():
        xref_rows.append({"series":series,"changepoint_utc":str(cp),"log_ts":str(w["ts"]),
                          "delta_hours":(w["ts"]-cp).total_seconds()/3600.0,
                          "surface_mood":w["surface_mood"],"anomaly_flag":w["anomaly_flag"]})
xref_df = pd.DataFrame(xref_rows)
with open(TABLES / "phase3_changepoint_crossref.csv", "w") as f:
    f.write("# Alignment (NOT interpretation) of detected changepoints with monthly "
            "observation-log entries within +/-12h. Quotes the log's 'surface mood' and "
            "'anomaly flag' fields verbatim. Also cross-references D_t robust-z outlier "
            "clusters (see phase3_changepoints.csv near_Dt_outlier_cluster column).\n")
    f.write(f"# D_t robust-z outlier clusters (|z|>3.5): {cluster_spans}\n")
    xref_df.to_csv(f, index=False)
log("Wrote phase3_changepoint_crossref.csv")

# stash for figures
np.savez(LOGS / "D_series.npz",
         phi_ts=dfp["ts"].view("int64").values, phi_detr=dfp["v_detr"].values,
         D_ts=dfd["ts"].view("int64").values, D_detr=dfd["v_detr"].values,
         cps_phi=np.array([pd.Timestamp(c).value for c in cps_phi]),
         cps_D=np.array([pd.Timestamp(c).value for c in cps_D]),
         dt_out_ts=dvals[dvals["outlier"]]["ts"].view("int64").values)

(LOGS / "task2_console_log.txt").write_text("\n".join(log_lines))
log("\n=== DONE (analyses A-D) ===")
