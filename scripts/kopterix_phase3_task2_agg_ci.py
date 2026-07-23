"""
Kopterix Phase 3 Task 2 -- Aggregate scalar confidence-interval repair.

SINGLE PURPOSE. This script repairs ONLY the aggregate scalar 95% confidence
intervals in phase3/tables/phase3_memory_timescales.csv for the three aggregate
run-level series:
    * phi_t              (semantic homogeneity)
    * D_t                (run-to-run drift)
    * lexical_cos_step   (consecutive lexical cosine-step distance)

WHY: the original circ_block_boot_tau() in kopterix_phase3_task2.py used circular
blocks of only 8 grid points. A block of length 8 cannot preserve the original
dependence at lags 8-16 used by the exponential fit, and concatenating the short
circular blocks manufactures artificial lagged pairs across sampled block
boundaries. Both effects shift the bootstrap decay-time distribution downward and
pushed several point estimates outside their own percentile intervals.

THE FIX: an adjacency-preserving, NON-CIRCULAR moving-block bootstrap with block
length 24 grid points (six days on the 6h grid). For every replicate we draw
contiguous original blocks of length 24, and at each lag we form lagged pairs ONLY
between members that lie inside the SAME sampled original block -- never across
block boundaries, and never wrapping the series end to its start. The pooled
within-block first/second members give ONE pairwise-complete Pearson correlation
per lag; the unchanged exponential model rho(k)=exp(-k*dt/tau) is fitted over
lags 1-16 and the unchanged AR(1) tau=-dt/ln(rho1) is taken from lag 1.

SCOPE / SAFETY:
    * The aggregate POINT ESTIMATES, R^2, rho1, grid counts and tau_vs_24h are
      recomputed exactly as in kopterix_phase3_task2.py and are NOT modified; only
      the ci95_lo / ci95_hi fields of the 12 aggregate rows are overwritten.
    * The three per-layer rows (layer_*_cosine_decay) are NOT touched: they keep
      the later adjacency-preserving correction written by kopterix_phase3_task2_D.py.
    * No Task 2B / 2C / 2D output is written. No figure is regenerated. No raw
      input file is modified. Paths are resolved from a portable root probe:
      the root is the directory containing phase3/tables/phase3_memory_timescales.csv.

Seed convention (unchanged aggregate convention): raw series -> seed 42,
detrended series -> seed 43. 5000 resamples. Percentile 95% CI from the 2.5th and
97.5th percentiles.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── paths (relative to this script; no session-absolute paths) ───────────────
# Portable root resolution: ROOT is the directory that contains
# phase3/tables/phase3_memory_timescales.csv. That is the project root when this
# script runs from phase3/, and the deposit root when it runs from
# zenodo_deposit/scripts/. BASE is retained as the phase3 output base.
HERE    = Path(__file__).resolve().parent
_ROOT_MARKER = Path("phase3") / "tables" / "phase3_memory_timescales.csv"


def _resolve_root():
    for _cand in (HERE, HERE.parent):
        if (_cand / _ROOT_MARKER).exists():
            return _cand
    return HERE.parent


ROOT    = _resolve_root()
BASE    = ROOT / "phase3"
TABLES  = BASE / "tables"
REAL_CSV = ROOT / "analysis_intermediate" / "real_clean.csv"
STATE_CSV = ROOT / "kopterix_state.csv"
MEM_CSV  = TABLES / "phase3_memory_timescales.csv"
DIAG_CSV = TABLES / "phase3_aggregate_memory_bootstrap_diagnostics.csv"

WINDOW_START = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
SEED_RAW = 42
SEED_DETR = 43
LAYERS = ["surface", "mid", "residue"]

# corrected-bootstrap constants
DT = 6.0                       # hours, regular grid spacing
TOL = pd.Timedelta("3h")
BLOCK = 24                     # grid points  == 6 days on the 6h grid
BLOCK_HOURS = BLOCK * DT       # 144 h
N_BOOT = 5000
LAGS = range(1, 17)            # exponential fit over lags 1-16 (unchanged)
BOOT_VARIANT = "adjacency_preserving_non_circular_moving_block"
INTERVAL_TYPE = "percentile_2.5_97.5"
AGG_SERIES = ["phi_t", "D_t", "lexical_cos_step"]

log_lines = []
def log(m):
    print(m); log_lines.append(str(m))

log("=== Kopterix Phase 3 Task 2 -- aggregate scalar CI repair (block=24, non-circular) ===")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions -- copied verbatim from kopterix_phase3_task2.py so the point
# estimates reproduce exactly. (Only the bootstrap function is the corrected one.)
# ─────────────────────────────────────────────────────────────────────────────
def parse_unigram(s):
    if pd.isna(s): return None
    try:
        d = json.loads(s); d.pop("__total__", None); return d
    except Exception:
        return None

def _epoch_ns(idx):
    """int64 nanoseconds-since-epoch for a DatetimeIndex, robust across pandas
    versions. The original kopterix_phase3_task2.py used DatetimeIndex.view("int64")
    which silently breaks on non-nanosecond-resolution indexes under pandas >= 3.0
    (every nearest-neighbour tolerance test then fails and the whole grid becomes
    NaN). Forcing ns resolution and using asi8 is numerically identical to the
    original nanosecond int view on the ns-resolution data used throughout Phase 3.
    (REPORTED CHANGE: required only to reproduce the original point estimates under
    the installed pandas; does not alter any value.)"""
    idx = pd.DatetimeIndex(idx)
    try:
        idx = idx.as_unit("ns")
    except (AttributeError, ValueError):
        pass
    try:
        return idx.asi8.astype(float)
    except AttributeError:
        return idx.view("int64").astype(float)

def resample_grid(ts, vals):
    """Nearest-neighbor onto regular 6h grid within 3h tolerance; else NaN."""
    ts = pd.DatetimeIndex(ts); vals = np.asarray(vals, float)
    ok = ~np.isnan(vals)
    ts, vals = ts[ok], vals[ok]
    order = np.argsort(ts.values); ts, vals = ts[order], vals[order]
    t0, t1 = ts[0], ts[-1]
    n_grid = int(np.floor((t1 - t0).total_seconds()/3600.0/DT)) + 1
    grid = pd.DatetimeIndex(t0 + pd.to_timedelta(np.arange(n_grid)*DT, unit="h"))
    out = np.full(n_grid, np.nan)
    ts_ns = _epoch_ns(ts)
    grid_ns = _epoch_ns(grid)
    tol_ns = float(TOL.value)
    for i in range(n_grid):
        gi = grid_ns[i]
        j = int(np.argmin(np.abs(ts_ns - gi)))
        if abs(ts_ns[j]-gi) <= tol_ns:
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

def fit_tau_exp(rho, dt, lags=LAGS):
    """Unchanged exp model rho(k)=exp(-k*dt/tau), NLS over lags 1-16, tau in (0.1,1e5)."""
    ks = np.array([k for k in lags if k < len(rho) and not np.isnan(rho[k])])
    ys = np.array([rho[k] for k in ks])
    if len(ks) < 3: return np.nan, np.nan
    def model(k, tau): return np.exp(-k*dt/tau)
    try:
        popt,_ = curve_fit(model, ks, ys, p0=[24.0], bounds=(0.1, 1e5), maxfev=10000)
        resid = ys - model(ks, popt[0])
        ss = np.sum(resid**2); sstot = np.sum((ys-ys.mean())**2)
        r2 = 1 - ss/sstot if sstot>0 else np.nan
        return float(popt[0]), float(r2)
    except Exception:
        return np.nan, np.nan

def tau_ar1(rho1, dt):
    """Unchanged AR(1) conversion from the lag-1 correlation."""
    if rho1 is None or np.isnan(rho1) or rho1 <= 0: return np.nan
    return float(-dt/np.log(rho1))

def linear_detrend(x):
    x = np.asarray(x, float); idx = np.arange(len(x)); m = ~np.isnan(x)
    if m.sum() < 3: return x.copy()
    b1, b0 = np.polyfit(idx[m], x[m], 1)
    return x - (b0 + b1*idx)

def cosine_dist_counts(d1, d2):
    vocab = sorted(set(d1.keys()) | set(d2.keys()))
    if not vocab: return np.nan
    v1 = np.array([d1.get(w,0) for w in vocab], float)
    v2 = np.array([d2.get(w,0) for w in vocab], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-12 or n2<1e-12: return np.nan
    return float(1.0 - np.dot(v1,v2)/(n1*n2))

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED BOOTSTRAP: adjacency-preserving, non-circular moving blocks (block=24)
# ─────────────────────────────────────────────────────────────────────────────
def adjacency_block_boot_tau(x, dt, block=BLOCK, n_boot=N_BOOT, rng=None, lags=LAGS):
    """Adjacency-preserving NON-CIRCULAR moving-block bootstrap CI for exp-fit tau
    and AR(1) tau.

    For each replicate:
      1. draw ceil(n/block) contiguous original blocks x[s:s+block] with
         replacement, each start s uniform on 0..n-block INCLUSIVE (no wrap-around);
      2. at each lag k, gather lagged pairs (blk[i], blk[i+k]) that lie WITHIN the
         same sampled original block only -- never across block boundaries;
      3. pool the finite first members and finite second members from all eligible
         within-block pairs and take ONE pairwise-complete Pearson correlation per
         lag (block-level correlations are NOT averaged);
      4. fit rho(k)=exp(-k*dt/tau) over lags 1-16 and take AR(1) tau from lag 1;
      5. discard only failed / non-finite estimates.
    """
    if rng is None: rng = np.random.default_rng(SEED_RAW)
    x = np.asarray(x, float)
    n = len(x)
    maxlag = max(lags)
    n_blocks = int(np.ceil(n / block))       # enough blocks for ~original length
    max_start = n - block                     # inclusive upper bound for a start
    if max_start < 0:
        return (np.array([]), np.array([]), n_boot, n_boot)
    taus_exp, taus_ar1 = [], []
    fail_exp = fail_ar1 = 0
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)  # 0..n-block inclusive
        blocks = [x[s:s+block] for s in starts]
        rho = np.full(maxlag + 1, np.nan); rho[0] = 1.0
        for k in range(1, maxlag + 1):
            a_parts, b_parts = [], []
            for blk in blocks:
                if len(blk) > k:                    # within-block pairs only
                    a_parts.append(blk[:-k])
                    b_parts.append(blk[k:])
            if a_parts:
                a = np.concatenate(a_parts)
                b = np.concatenate(b_parts)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= 5 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                    rho[k] = np.corrcoef(a[m], b[m])[0, 1]
        te, _ = fit_tau_exp(rho, dt, lags)
        ta = tau_ar1(rho[1], dt)
        if np.isfinite(te): taus_exp.append(te)
        else: fail_exp += 1
        if np.isfinite(ta): taus_ar1.append(ta)
        else: fail_ar1 += 1
    return np.array(taus_exp), np.array(taus_ar1), fail_exp, fail_ar1

def pct_ci(a):
    a = np.asarray(a, float)
    if len(a) > 10:
        return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    return (np.nan, np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD raw inputs (read-only) and REBUILD the three aggregate series
# ─────────────────────────────────────────────────────────────────────────────
state = pd.read_csv(STATE_CSV)
state["ts"] = pd.to_datetime(state["Timestamp"], utc=True)
state = state.sort_values("ts").reset_index(drop=True)
state["unigram"] = state["UnigramCounts"].apply(parse_unigram)
in_win = (state["ts"] >= WINDOW_START) & (state["ts"] < WINDOW_END)
sw = state[in_win].copy().reset_index(drop=True)

real = pd.read_csv(REAL_CSV)
real["ts"] = pd.to_datetime(real["Timestamp_UTC"].str.replace(" UTC", "", regex=False), utc=True)
real = real.sort_values("ts").reset_index(drop=True)
assert (real["ts"] >= WINDOW_START).all() and (real["ts"] < WINDOW_END).all(), \
    "real_clean.csv rows fall outside the [2026-04-01, 2026-06-01) UTC window"

# lexical consecutive-step cosine distance, assigned to the DESTINATION run timestamp
su = sw[sw["unigram"].notna()].copy().reset_index(drop=True)
lex_ts, lex_val = [], []
for i in range(len(su) - 1):
    lex_val.append(cosine_dist_counts(su.iloc[i]["unigram"], su.iloc[i+1]["unigram"]))
    lex_ts.append(su.iloc[i+1]["ts"])

grid_phi, g_phi = resample_grid(real["ts"], real["phi_t"].values)
grid_D,   g_D   = resample_grid(real["ts"], real["D_t"].values)
grid_lex, g_lex = resample_grid(pd.DatetimeIndex(lex_ts), np.array(lex_val))
series_grids = {"phi_t": g_phi, "D_t": g_D, "lexical_cos_step": g_lex}

# ─────────────────────────────────────────────────────────────────────────────
# Point estimates (unchanged) + corrected bootstrap intervals
# ─────────────────────────────────────────────────────────────────────────────
# results[(series, detrended_bool, method_kind)] = dict(...)
results = {}
diag_rows = []
for name in AGG_SERIES:
    g = series_grids[name]
    for detr in (False, True):
        series = linear_detrend(g) if detr else g
        rho = nan_acf(series, 28)
        te, r2 = fit_tau_exp(rho, DT)
        ta = tau_ar1(rho[1], DT)
        seed = SEED_DETR if detr else SEED_RAW
        rng_b = np.random.default_rng(seed)
        boot_exp, boot_ar1, fail_exp, fail_ar1 = adjacency_block_boot_tau(
            series, DT, block=BLOCK, n_boot=N_BOOT, rng=rng_b, lags=LAGS)
        ci_e = pct_ci(boot_exp)
        ci_a = pct_ci(boot_ar1)
        med_e = float(np.nanmedian(boot_exp)) if len(boot_exp) else np.nan
        med_a = float(np.nanmedian(boot_ar1)) if len(boot_ar1) else np.nan

        results[(name, detr, "exp")] = dict(point=te, ci=ci_e)
        results[(name, detr, "ar1")] = dict(point=ta, ci=ci_a)

        log(f"[{name} {'detr' if detr else 'raw '}] seed={seed} "
            f"exp tau={te:.4f} R2={r2:.4f} CI[{ci_e[0]:.4f},{ci_e[1]:.4f}] "
            f"(n_ok={len(boot_exp)}, med={med_e:.4f}) | "
            f"ar1 tau={ta:.4f} CI[{ci_a[0]:.4f},{ci_a[1]:.4f}] "
            f"(n_ok={len(boot_ar1)}, med={med_a:.4f})")

        # exponential diagnostics row
        pe = te
        inside_e = bool(np.isfinite(pe) and np.isfinite(ci_e[0]) and np.isfinite(ci_e[1])
                        and ci_e[0] <= pe <= ci_e[1])
        msg_e = "ok"
        if name == "phi_t" and not detr:
            msg_e = ("raw exponential fit has negative R^2 (phi_t's negative trend keeps the "
                     "raw ACF from returning to zero); retained only to show why detrending "
                     "is required; report table displays n/a for this interval")
        diag_rows.append(dict(
            series=name, detrended=detr, method="exp_fit_lags1-16",
            point_estimate_hours=pe, ci95_lo=ci_e[0], ci95_hi=ci_e[1],
            bootstrap_median_hours=med_e,
            n_successful_bootstrap_estimates=len(boot_exp),
            point_inside_ci=inside_e,
            block_length_grid_points=BLOCK, block_length_hours=BLOCK_HOURS,
            n_resamples=N_BOOT, seed=seed, bootstrap_variant=BOOT_VARIANT,
            interval_type=INTERVAL_TYPE, fit_lag_start=1, fit_lag_end=16,
            point_estimate_R2=r2,
            n_failed_bootstrap_estimates=fail_exp,
            diagnostic_message=msg_e))

        # AR(1) diagnostics row
        inside_a = bool(np.isfinite(ta) and np.isfinite(ci_a[0]) and np.isfinite(ci_a[1])
                        and ci_a[0] <= ta <= ci_a[1])
        diag_rows.append(dict(
            series=name, detrended=detr, method="AR1",
            point_estimate_hours=ta, ci95_lo=ci_a[0], ci95_hi=ci_a[1],
            bootstrap_median_hours=med_a,
            n_successful_bootstrap_estimates=len(boot_ar1),
            point_inside_ci=inside_a,
            block_length_grid_points=BLOCK, block_length_hours=BLOCK_HOURS,
            n_resamples=N_BOOT, seed=seed, bootstrap_variant=BOOT_VARIANT,
            interval_type=INTERVAL_TYPE, fit_lag_start=1, fit_lag_end=1,
            point_estimate_R2=np.nan,
            n_failed_bootstrap_estimates=fail_ar1,
            diagnostic_message="ok"))

# ─────────────────────────────────────────────────────────────────────────────
# Update ONLY ci95_lo / ci95_hi of the 12 aggregate rows in the memory CSV,
# leaving every other field (and every per-layer row) byte-identical.
# ─────────────────────────────────────────────────────────────────────────────
raw_text = MEM_CSV.read_text()
raw_lines = raw_text.splitlines()
comment_lines = [l for l in raw_lines if l.startswith("#")]
data_lines = [l for l in raw_lines if l and not l.startswith("#")]
header = data_lines[0]
cols = header.split(",")
IX = {c: i for i, c in enumerate(cols)}
i_series, i_detr, i_method = IX["series"], IX["detrended"], IX["method"]
i_lo, i_hi = IX["ci95_lo"], IX["ci95_hi"]

def method_kind(m):
    return "exp" if "exp" in m else ("ar1" if m == "AR1" else None)

comparison = []   # (series, detrended, method, old_lo, old_hi, new_lo, new_hi)
new_data_lines = [header]
for line in data_lines[1:]:
    f = line.split(",")
    s = f[i_series]
    if s in AGG_SERIES:
        detr = (f[i_detr] == "True")
        mk = method_kind(f[i_method])
        key = (s, detr, mk)
        if key in results:
            new_lo, new_hi = results[key]["ci"]
            old_lo, old_hi = f[i_lo], f[i_hi]
            comparison.append((s, f[i_detr], f[i_method], old_lo, old_hi,
                               repr(new_lo), repr(new_hi)))
            f[i_lo] = repr(float(new_lo))
            f[i_hi] = repr(float(new_hi))
    new_data_lines.append(",".join(f))

# old-vs-new comparison (printed + saved) BEFORE overwriting
log("\n=== OLD vs NEW aggregate 95% CI (block=8 circular  ->  block=24 non-circular) ===")
log(f"{'series':17s} {'detr':5s} {'method':22s} {'old_lo':>9s} {'old_hi':>9s} "
    f"{'new_lo':>9s} {'new_hi':>9s}")
cmp_records = []
for s, d, m, olo, ohi, nlo, nhi in comparison:
    log(f"{s:17s} {d:5s} {m:22s} {float(olo):9.3f} {float(ohi):9.3f} "
        f"{float(nlo):9.3f} {float(nhi):9.3f}")
    cmp_records.append(dict(series=s, detrended=d, method=m,
                            old_ci95_lo=float(olo), old_ci95_hi=float(ohi),
                            new_ci95_lo=float(nlo), new_ci95_hi=float(nhi)))
pd.DataFrame(cmp_records).to_csv(TABLES / "phase3_aggregate_ci_old_vs_new.csv",
                                 index=False, lineterminator="\n")
log(f"Saved old-vs-new comparison -> {TABLES / 'phase3_aggregate_ci_old_vs_new.csv'}")

# rebuild the comment header: aggregate method scoped explicitly to the 3 series,
# and a separate statement that the per-layer rows keep the task2_D.py correction.
new_header = [
    ("# Memory timescale tau (hours). phi_t/D_t/lexical_cos_step resampled to a regular 6h "
     "grid (nearest obs within 3h tolerance; gaps beyond tolerance left NaN, never "
     "interpolated). ACF computed pairwise-complete to lag 28; tau from (i) NLS exp fit "
     "rho(k)=exp(-k*dt/tau) on lags 1-16 and (ii) AR(1) tau=-dt/ln(rho1). Detrended rows use "
     "linear-detrended series (the defensible estimate, given phi_t's strong negative trend). "
     "Per-layer rows fit sim(k)=A*exp(-k*dt/tau)+C on consecutive valid rows (no bridging "
     "gaps>12h); dt = median consecutive gap. tau_vs_24h = tau/24; exceeds_24h flags "
     "persistence outliving the deepest 24h sampled feed-age window (NOT a proof of relay)."),
    ("# AGGREGATE 95% CI METHOD (corrected): rows whose series is phi_t, D_t or "
     "lexical_cos_step report 95% CI from an ADJACENCY-PRESERVING moving-block bootstrap -- "
     "NON-CIRCULAR contiguous blocks of 24 grid points (six days on the 6h grid), 5000 "
     "resamples, raw seed 42 and detrended seed 43, percentile 95% intervals from the 2.5th "
     "and 97.5th percentiles, exponential fitting over lags 1-16. Lagged pairs are formed "
     "only WITHIN a sampled original block; no pairs cross sampled block boundaries and no "
     "block wraps the series end to its start. This supersedes the earlier block=8, 1000-"
     "resample circular bootstrap. This aggregate CI method applies ONLY to the phi_t, D_t "
     "and lexical_cos_step rows, NOT to the per-layer layer_*_cosine_decay rows. Written by "
     "kopterix_phase3_task2_agg_ci.py."),
    ("# NOTE (per-layer CI corrected): per-layer 95% CI now from a moving-block bootstrap "
     "with block length 16 runs that PRESERVES original row adjacency (lag-k cosine "
     "similarities only ever use original-adjacent pairs within a block); the earlier "
     "draft's CI scrambled adjacency and is superseded."),
    ("# The per-layer rows (series = layer_surface_cosine_decay, layer_mid_cosine_decay, "
     "layer_residue_cosine_decay) preserve the later adjacency-preserving correction written "
     "by kopterix_phase3_task2_D.py and are NOT modified by this aggregate CI repair."),
]

# Write bytes with explicit "\n" endings (never let the OS translate to CRLF) and
# do it atomically via a temp file + replace so a killed process cannot leave a
# half-written table behind.
_mem_text = "\n".join(new_header) + "\n" + "\n".join(new_data_lines) + "\n"
_mem_tmp = MEM_CSV.with_suffix(".csv.tmp")
_mem_tmp.write_bytes(_mem_text.encode("utf-8"))
_mem_tmp.replace(MEM_CSV)
log(f"\nUpdated aggregate ci95_lo/ci95_hi in {MEM_CSV} (per-layer rows untouched)")

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics CSV
# ─────────────────────────────────────────────────────────────────────────────
diag_cols = ["series", "detrended", "method", "point_estimate_hours", "ci95_lo", "ci95_hi",
             "bootstrap_median_hours", "n_successful_bootstrap_estimates", "point_inside_ci",
             "block_length_grid_points", "block_length_hours", "n_resamples", "seed",
             "bootstrap_variant", "interval_type", "fit_lag_start", "fit_lag_end",
             "point_estimate_R2", "n_failed_bootstrap_estimates", "diagnostic_message"]
diag_df = pd.DataFrame(diag_rows)[diag_cols]
_diag_hdr = ("# Aggregate memory-timescale bootstrap diagnostics for the phi_t, D_t and "
             "lexical_cos_step scalars. One row per (series, detrended, method). CI from the "
             "adjacency-preserving, NON-CIRCULAR moving-block bootstrap: contiguous blocks of "
             "24 grid points (6 days on the 6h grid), 5000 resamples, raw seed 42 / detrended "
             "seed 43, percentile 2.5/97.5. Lagged pairs are within-block only. point_inside_ci "
             "reports whether the (unchanged) point estimate lies within its corrected "
             "interval. point_estimate_R2 is populated for exponential rows only. This file "
             "does NOT cover the per-layer layer_*_cosine_decay rows.\n")
_diag_text = _diag_hdr + diag_df.to_csv(index=False, lineterminator="\n")
DIAG_CSV.write_bytes(_diag_text.encode("utf-8"))
log(f"Wrote diagnostics -> {DIAG_CSV}")

# report any interpretable point estimate still outside its corrected interval
outside = [r for r in diag_rows
           if not r["point_inside_ci"] and np.isfinite(r["point_estimate_hours"])
           and not (r["series"] == "phi_t" and r["detrended"] is False and "exp" in r["method"])]
log("\n=== Point estimates still OUTSIDE their corrected interval (interpretable rows) ===")
if outside:
    for r in outside:
        log(f"  OUTSIDE: {r['series']} detr={r['detrended']} {r['method']} "
            f"point={r['point_estimate_hours']:.4f} CI[{r['ci95_lo']:.4f},{r['ci95_hi']:.4f}]")
else:
    log("  (none -- every interpretable aggregate point estimate lies inside its corrected CI)")

# raw phi_t exp is reported separately (it is the deliberately-unreliable reference)
rphi = results[("phi_t", False, "exp")]
log(f"\n[raw phi_t exp] point={rphi['point']:.4f} (negative R^2) "
    f"CI[{rphi['ci'][0]:.4f},{rphi['ci'][1]:.4f}] retained in CSV; report table shows n/a.")

(BASE / "logs" / "task2_agg_ci_log.txt").write_bytes(("\n".join(log_lines) + "\n").encode("utf-8"))
log("\n=== DONE: aggregate scalar CI repair ===")
# completion sentinel (used only to poll background execution; harmless to leave)
(BASE / "logs" / "task2_agg_ci_DONE.flag").write_bytes(b"OK\n")
