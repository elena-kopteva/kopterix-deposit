"""
Kopterix Phase 3 -- Metric repair: rarefied entropy H_rare and the n_total confound.

Repairs the lexical entropy metric H_t, which a reviewer found correlates with
sample size n_total at r ~ 0.92 (i.e. it currently measures sample size, not
lexical diversity). The repair: rarefaction. For each run, draw fixed-size
multinomial subsamples from the per-run unigram distribution (UnigramCounts in
kopterix_state.csv) and compute Shannon entropy of the subsample. This removes
the dependence of entropy on the underlying sample size.

CRITICAL CAVEAT (repeated in every output): UnigramCounts stores only the
TOP-200 tokens per snapshot (plus a __total__ key, excluded from all entropy
computations here). H_rare is therefore the rarefied entropy of a TRUNCATED
top-200 distribution, not the full lexical distribution. This is an instrument
diagnostic, not a platform-level claim about lexical diversity.

Reproducibility: a single numpy Generator is seeded with 42 at the start of the
run. Rows are processed in chronological (Timestamp_UTC) order. For each row,
50 multinomial draws of size B are taken first, followed by 50 draws of size B2
(only for rows with >= B2 total top-200 tokens), all from the same rng stream,
in that order. Re-running this script reproduces identical results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE    = Path(__file__).resolve().parents[1]
TABLES  = BASE / "phase3" / "tables"
FIGURES = BASE / "phase3" / "figures"
LOGS    = BASE / "phase3" / "logs"

WINDOW_START = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")

N_REPS = 50
SEED   = 42
B2     = 5000

CAVEAT = (
    "CAVEAT: UnigramCounts stores only the TOP-200 tokens per snapshot "
    "(__total__ key excluded from all entropy computations). H_rare is the "
    "rarefied entropy of this TRUNCATED top-200 distribution, not the full "
    "lexical distribution. Instrument diagnostic, not a platform-level claim."
)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=== Kopterix Phase 3: rarefied entropy repair ===")
log(CAVEAT)

# ── 1. Load real_clean.csv (already excludes the quarantined April H_t=4.2 row) ──
real = pd.read_csv(BASE / "analysis_intermediate" / "real_clean.csv")
real["ts_utc"] = pd.to_datetime(
    real["Timestamp_UTC"].str.replace(" UTC", "", regex=False), utc=True)
real = real.sort_values("ts_utc").reset_index(drop=True)

assert (real["ts_utc"] >= WINDOW_START).all() and (real["ts_utc"] < WINDOW_END).all(), \
    "real_clean.csv contains rows outside the [2026-04-01, 2026-06-01) window"

valid = real[real["H_t_numeric"].notna()].copy().reset_index(drop=True)
log(f"\nLoaded analysis_intermediate/real_clean.csv: {len(real)} rows in window")
log(f"Valid rows for entropy repair (H_t_numeric notna; April H_t=4.2 quarantine "
    f"already excluded upstream in real_clean.csv): {len(valid)}")
log(f"  April: {(valid['month']=='2026-04').sum()}  May: {(valid['month']=='2026-05').sum()}")
n_regime = int(valid["n_total_regime"].sum())
log(f"  n_total_regime (low-n, n_total<=300, May, flagged not excluded): {n_regime}")

# ── 2. Load and parse kopterix_state.csv ──
state = pd.read_csv(BASE / "kopterix_state.csv")
state["ts_utc"] = pd.to_datetime(state["Timestamp"], utc=True)
state = state.sort_values("ts_utc").reset_index(drop=True)

def parse_unigram(s):
    d = json.loads(s)
    d.pop("__total__", None)
    return d

state["unigram"] = state["UnigramCounts"].apply(parse_unigram)
state["total_tokens_top200"] = state["unigram"].apply(lambda d: sum(d.values()))
log(f"\nParsed kopterix_state.csv: {len(state)} rows, "
    f"total_tokens_top200 range [{state['total_tokens_top200'].min()}, "
    f"{state['total_tokens_top200'].max()}]")

# ── 3. Match valid observations to state rows (nearest, 5-min tolerance) ──
obs_s  = valid[["ts_utc"]].assign(obs_idx=valid.index).sort_values("ts_utc")
stat_s = state[["ts_utc"]].assign(state_idx=state.index).sort_values("ts_utc")
matched = pd.merge_asof(
    obs_s, stat_s.rename(columns={"ts_utc": "ts_state"}),
    left_on="ts_utc", right_on="ts_state", direction="nearest",
    tolerance=pd.Timedelta("5min"))

n_unmatched = int(matched["state_idx"].isna().sum())
log(f"\nObs-to-state match (5min tolerance): {len(valid)-n_unmatched}/{len(valid)} matched, "
    f"{n_unmatched} unmatched")
if n_unmatched > 0:
    raise SystemExit("FATAL: unmatched valid rows -- investigate before proceeding")

state_idx = matched.set_index("obs_idx")["state_idx"].astype(int)
valid["state_idx"]           = state_idx.values
valid["state_ts"]            = state.loc[valid["state_idx"], "ts_utc"].values
valid["unigram"]              = state.loc[valid["state_idx"], "unigram"].values
valid["total_tokens_top200"] = state.loc[valid["state_idx"], "total_tokens_top200"].values

# ── 4. Determine token budgets ──
B = int(valid["total_tokens_top200"].min())
b2_eligible = valid["total_tokens_top200"] >= B2
log(f"\nToken budget B (min total_tokens_top200 across {len(valid)} valid rows) = {B}")
log(f"Secondary fixed budget B2 = {B2}: {int(b2_eligible.sum())}/{len(valid)} rows eligible "
    f"(>= {B2} top-200 tokens)")
min_row = valid.loc[valid["total_tokens_top200"].idxmin()]
log(f"  Row driving B: {min_row['Timestamp_UTC']} (n_total={min_row['n_total']}, "
    f"n_total_regime={min_row['n_total_regime']}, total_tokens_top200={min_row['total_tokens_top200']})")

# ── 5. Rarefaction ──
def shannon_entropy_bits(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

def rarefy(unigram_dict, budget, rng, n_reps=N_REPS):
    tokens = list(unigram_dict.keys())
    counts = np.array([unigram_dict[t] for t in tokens], dtype=np.float64)
    total = counts.sum()
    if total < budget:
        return np.nan, np.nan
    p = counts / total
    ents = np.empty(n_reps)
    for i in range(n_reps):
        draw = rng.multinomial(budget, p)
        ents[i] = shannon_entropy_bits(draw)
    return float(ents.mean()), float(ents.std(ddof=1))

rng = np.random.default_rng(SEED)
H_rare, H_rare_std, H_rare_B2, H_rare_B2_std = [], [], [], []
for _, row in valid.iterrows():
    m, s = rarefy(row["unigram"], B, rng)
    H_rare.append(m); H_rare_std.append(s)
    if row["total_tokens_top200"] >= B2:
        m2, s2 = rarefy(row["unigram"], B2, rng)
    else:
        m2, s2 = np.nan, np.nan
    H_rare_B2.append(m2); H_rare_B2_std.append(s2)

valid["H_rare"]        = H_rare
valid["H_rare_std"]    = H_rare_std
valid["H_rare_B2"]     = H_rare_B2
valid["H_rare_B2_std"] = H_rare_B2_std
valid["budget_B"]  = B
valid["budget_B2"] = B2
valid["B2_eligible"] = b2_eligible.values

log(f"\nH_rare (B={B}) summary: mean={valid['H_rare'].mean():.4f} "
    f"std={valid['H_rare'].std():.4f} "
    f"min={valid['H_rare'].min():.4f} max={valid['H_rare'].max():.4f}")
log(f"H_rare_B2 (B2={B2}, n={int(b2_eligible.sum())}) summary: "
    f"mean={valid['H_rare_B2'].mean():.4f} std={valid['H_rare_B2'].std():.4f}")

# ── 6. Output: phase3_rarefied_entropy.csv ──
out_cols = ["Timestamp_UTC", "month", "n_total", "n_total_regime",
            "total_tokens_top200", "budget_B", "H_rare", "H_rare_std",
            "B2_eligible", "budget_B2", "H_rare_B2", "H_rare_B2_std",
            "H_t", "H_t_numeric"]
entropy_out = valid[out_cols].copy()

entropy_path = TABLES / "phase3_rarefied_entropy.csv"
with open(entropy_path, "w") as f:
    f.write(f"# {CAVEAT}\n")
    f.write(f"# B = {B} (min total_tokens_top200 across {len(valid)} valid rows); "
            f"B2 = {B2} (used only where total_tokens_top200 >= B2, "
            f"{int(b2_eligible.sum())}/{len(valid)} rows)\n")
    f.write(f"# H_rare/H_rare_B2: mean Shannon entropy (bits) over {N_REPS} multinomial "
            f"subsamples (seed={SEED}); H_rare_std/H_rare_B2_std: std (ddof=1) over those "
            f"{N_REPS} draws\n")
    entropy_out.to_csv(f, index=False)
log(f"\nWrote {entropy_path} ({len(entropy_out)} rows)")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS B: does the repair work?
# ══════════════════════════════════════════════════════════════════════════
log("\n=== Analysis B: does the repair work? ===")

apr = valid[valid["month"] == "2026-04"]
may = valid[valid["month"] == "2026-05"]

# ── B1. Pearson / Spearman: H_rare vs n_total ──
log("\n--- B1. H_rare vs n_total correlations ---")
corr_rows = []

def corr_block(df, scope, hcol, budget_label, n_label):
    n_total_vals = df["n_total"].dropna()
    sub = df.loc[n_total_vals.index, hcol]
    n_pairs = len(n_total_vals)
    if n_total_vals.nunique() <= 1:
        note = (f"undefined (n_total constant at {n_total_vals.iloc[0]:.0f} "
                f"for all {n_pairs} non-NaN {scope} rows)")
        return {"scope": scope, "budget": budget_label, "n_pairs": n_pairs,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "note": note}
    pr, pp = stats.pearsonr(n_total_vals.values, sub.values)
    sr, sp = stats.spearmanr(n_total_vals.values, sub.values)
    note = f"pearson r={pr:.4f} p={pp:.3g}; spearman r={sr:.4f} p={sp:.3g}"
    return {"scope": scope, "budget": budget_label, "n_pairs": n_pairs,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp, "note": note}

# Primary: H_rare at budget B (all rows have H_rare)
corr_rows.append(corr_block(apr, "April-only", "H_rare", f"B={B}", "n_total"))
corr_rows.append(corr_block(may, "May-only", "H_rare", f"B={B}", "n_total"))
corr_rows.append(corr_block(valid, "Two-month", "H_rare", f"B={B}", "n_total"))

# Sensitivity: H_rare_B2 at budget B2 (subset of rows)
valid_b2 = valid[valid["B2_eligible"]]
apr_b2 = valid_b2[valid_b2["month"] == "2026-04"]
may_b2 = valid_b2[valid_b2["month"] == "2026-05"]
corr_rows.append(corr_block(apr_b2, "April-only (B2 subset, budget sensitivity)",
                             "H_rare_B2", f"B2={B2}", "n_total"))
corr_rows.append(corr_block(may_b2, "May-only (B2 subset, budget sensitivity)",
                             "H_rare_B2", f"B2={B2}", "n_total"))
corr_rows.append(corr_block(valid_b2, "Two-month (B2 subset, budget sensitivity)",
                             "H_rare_B2", f"B2={B2}", "n_total"))

# For reference: raw H_t vs n_total, recomputed on the same valid set (should match
# the previously reported r~0.92)
corr_rows.append(corr_block(apr, "April-only (raw H_t, reference)", "H_t_numeric",
                             "n/a (raw)", "n_total"))
corr_rows.append(corr_block(may, "May-only (raw H_t, reference)", "H_t_numeric",
                             "n/a (raw)", "n_total"))
corr_rows.append(corr_block(valid, "Two-month (raw H_t, reference)", "H_t_numeric",
                             "n/a (raw)", "n_total"))

corr_df = pd.DataFrame(corr_rows)
for r in corr_rows:
    log(f"  {r['scope']:45s} [{r['budget']:6s}] n={r['n_pairs']:3d}  {r['note']}")

corr_path = TABLES / "phase3_hrare_ntotal_corr.csv"
with open(corr_path, "w") as f:
    f.write(f"# {CAVEAT}\n")
    f.write(f"# H_rare vs n_total correlations (Pearson and Spearman). Primary rows use "
            f"H_rare at budget B={B} (all 214 rows). 'B2 subset' rows use H_rare_B2 at "
            f"budget B2={B2} ({int(b2_eligible.sum())} rows with >= {B2} top-200 tokens) "
            f"as a budget-sensitivity check. 'raw H_t' rows reproduce the original "
            f"(unrepaired) correlation for reference. April n_total is constant at 750 "
            f"for all 104 April rows -> Pearson/Spearman undefined for April-only.\n")
    corr_df.to_csv(f, index=False)
log(f"\nWrote {corr_path} ({len(corr_df)} rows)")

# ── B2. April-vs-May comparison on H_rare (MW + KS), BH-corrected alongside the
#       other cross-month tests (rerun from analysis_intermediate/real_clean.csv,
#       same metrics/columns as tables/phase2_april_may_stability.csv) ──
log("\n--- B2. April-vs-May comparison on H_rare (with BH correction across all "
    "cross-month tests) ---")

OTHER_METRICS = ["post_rate_est", "H_t_numeric", "phi_t", "D_t", "n_total",
                 "centroid_dist_surface_mid", "centroid_dist_mid_residue",
                 "centroid_dist_surface_residue"]

cm_rows = []
for m in OTHER_METRICS:
    a = real[real["month"] == "2026-04"][m].dropna()
    b = real[real["month"] == "2026-05"][m].dropna()
    mw_u, mw_p = stats.mannwhitneyu(a.values, b.values, alternative="two-sided")
    ks_s, ks_p = stats.ks_2samp(a.values, b.values)
    pooled_std = pd.concat([a, b]).std()
    d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else np.nan
    cm_rows.append({
        "metric": m, "n_april": len(a), "n_may": len(b),
        "mean_april": a.mean(), "mean_may": b.mean(),
        "mean_shift_may_minus_april": b.mean() - a.mean(),
        "cohens_d": d, "mw_pval": mw_p, "ks_pval": ks_p,
    })

# H_rare: April vs May (B-budget, all 214 valid rows have H_rare)
a_h = apr["H_rare"].dropna()
b_h = may["H_rare"].dropna()
mw_u_h, mw_p_h = stats.mannwhitneyu(a_h.values, b_h.values, alternative="two-sided")
ks_s_h, ks_p_h = stats.ks_2samp(a_h.values, b_h.values)
pooled_std_h = pd.concat([a_h, b_h]).std()
d_h = (b_h.mean() - a_h.mean()) / pooled_std_h if pooled_std_h > 0 else np.nan
cm_rows.append({
    "metric": f"H_rare (B={B})", "n_april": len(a_h), "n_may": len(b_h),
    "mean_april": a_h.mean(), "mean_may": b_h.mean(),
    "mean_shift_may_minus_april": b_h.mean() - a_h.mean(),
    "cohens_d": d_h, "mw_pval": mw_p_h, "ks_pval": ks_p_h,
})

cm_df = pd.DataFrame(cm_rows)
# BH correction across all cross-month MW p-values, jointly with the new H_rare test
mw_q = stats.false_discovery_control(cm_df["mw_pval"].values, method="bh")
cm_df["mw_q_bh"] = mw_q
cm_df["flag_mw_sig_0.05"] = cm_df["mw_pval"] < 0.05
cm_df["flag_mw_q_sig_0.05"] = cm_df["mw_q_bh"] < 0.05

raw_ht_mw_p = cm_df.loc[cm_df["metric"] == "H_t_numeric", "mw_pval"].iloc[0]
hrare_mw_p  = mw_p_h
hrare_mw_q  = cm_df.loc[cm_df["metric"].str.startswith("H_rare"), "mw_q_bh"].iloc[0]

PREVIOUS_RAW_UNIGRAM_MW_P = 1.6868e-31
log(f"  H_t_numeric (raw observation metric) April-vs-May MW p (rerun on real_clean) "
    f"= {raw_ht_mw_p:.3g}")
log(f"  Previously reported cross-month unigram entropy MW p (raw, single test, "
    f"computed directly on full top-200 entropy of UnigramCounts, unadjusted) "
    f"= {PREVIOUS_RAW_UNIGRAM_MW_P:.3g}")
log(f"  H_rare (B={B}) April-vs-May: MW p = {hrare_mw_p:.3g}, "
    f"KS p = {ks_p_h:.3g}, BH q (MW) = {hrare_mw_q:.3g}")
survives = bool(hrare_mw_q < 0.05)
log(f"  Cross-month entropy difference {'SURVIVES' if survives else 'DOES NOT SURVIVE'} "
    f"rarefaction at q<0.05 (BH-corrected across {len(cm_df)} cross-month tests).")

cm_path = TABLES / "phase3_hrare_april_may.csv"
with open(cm_path, "w") as f:
    f.write(f"# {CAVEAT}\n")
    f.write(f"# April-vs-May Mann-Whitney U and Kolmogorov-Smirnov tests, rerun from "
            f"analysis_intermediate/real_clean.csv for the {len(OTHER_METRICS)} metrics "
            f"in tables/phase2_april_may_stability.csv, plus H_rare (budget B={B}) added "
            f"as a {len(cm_df)}th test. mw_q_bh is the Benjamini-Hochberg-adjusted MW "
            f"p-value across all {len(cm_df)} tests jointly.\n")
    f.write(f"# Previously reported cross-month unigram entropy MW p (raw, unadjusted, "
            f"single test, on full top-200 entropy of UnigramCounts) = "
            f"{PREVIOUS_RAW_UNIGRAM_MW_P:.4e}. After rarefaction, H_rare (B={B}) "
            f"April-vs-May MW p = {hrare_mw_p:.4e}, BH q = {hrare_mw_q:.4e} -> "
            f"{'SURVIVES' if survives else 'DOES NOT SURVIVE'} at q<0.05.\n")
    cm_df.to_csv(f, index=False)
log(f"\nWrote {cm_path} ({len(cm_df)} rows)")

# ── B3. Low-n_total sensitivity on H_rare ──
log("\n--- B3. Low-n_total sensitivity on H_rare ---")
n_all   = len(valid)
mean_all   = valid["H_rare"].mean()
median_all = valid["H_rare"].median()
std_all    = valid["H_rare"].std()

excl = valid[~valid["n_total_regime"]]
n_excl   = len(excl)
mean_excl   = excl["H_rare"].mean()
median_excl = excl["H_rare"].median()
std_excl    = excl["H_rare"].std()

mean_diff = mean_excl - mean_all
pct_of_std = abs(mean_diff) / std_all * 100

sens_df = pd.DataFrame([{
    "metric": f"H_rare (B={B})",
    "n_all": n_all, "mean_all": mean_all, "median_all": median_all, "std_all": std_all,
    "n_excl_regime": n_excl, "mean_excl": mean_excl, "median_excl": median_excl,
    "std_excl": std_excl,
    "mean_diff_excl_minus_all": mean_diff,
    "pct_of_std": pct_of_std,
    "n_regime_rows": n_regime,
}])

log(f"  H_rare mean (all {n_all}) = {mean_all:.4f}, std = {std_all:.4f}")
log(f"  H_rare mean (excl {n_regime} low-n rows, n={n_excl}) = {mean_excl:.4f}")
log(f"  mean diff = {mean_diff:.4f} ({pct_of_std:.2f}% of two-month std)")
log(f"  For reference, raw H_t_numeric low-n sensitivity was 15.16% of std "
    f"(tables/phase2_low_n_sensitivity.csv)")

sens_path = TABLES / "phase3_hrare_low_n_sensitivity.csv"
with open(sens_path, "w") as f:
    f.write(f"# {CAVEAT}\n")
    f.write(f"# Sensitivity of H_rare (budget B={B}, all {n_all} valid rows) to the "
            f"{n_regime} n_total_regime rows (n_total<=300, all May, flagged not "
            f"excluded from the primary analysis). pct_of_std = "
            f"|mean_excl - mean_all| / std_all * 100.\n")
    sens_df.to_csv(f, index=False)
log(f"\nWrote {sens_path}")
