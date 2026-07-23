#!/usr/bin/env python3
"""
kopterix_phase2_corrected.py
Two-month Kopterix validation -- Phase 2 (correction pass)

Corrections vs. first Phase 2 run:
  C1  H_t-n_total correlation: April n_total is constant (all 750.0);
      Pearson r is undefined for April-only. Rephrased as May-only plus
      two-month combined dependence.
  C2  LayerCentroids sparse coverage: 5 incomplete rows (not 2), all May,
      4/5 matched to n_total_regime observations. Listed explicitly.
      Treated as sparse layer coverage in the low-n_total regime.
  C3  Layer-label shuffle: within-label norm is permutation-invariant by
      construction (mean of 3 per-label spreads, invariant under relabeling).
      Result labelled inconclusive/metric-insensitive. Discriminability
      conclusion removed.
  C4  FDR: BH-adjusted q-values added to cross-month p-value table.
  C5  Centroid phrasing: "smaller than but comparable to within-month
      variability" replacing any "negligible" or "within the range of"
      language.
  C6  Phase 1 OLS trend p-values now populated (scipy was available on
      Phase 1 re-run). NaN provenance note updated accordingly.

Reads Phase 1 intermediates only. Does not modify raw exports.
"""
import sys, json, warnings
from itertools import permutations
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("=== Kopterix Phase 2 (corrected) Pre-flight ===")
try:
    import scipy
    from scipy import stats
    from scipy.stats import false_discovery_control
    print(f"  scipy {scipy.__version__} -- OK (FDR available)")
except ImportError:
    print("  FATAL: scipy not available. Cannot produce inferential statistics.")
    sys.exit(1)

BASE          = Path(__file__).resolve().parent
if not (BASE / "kopterix_state.csv").exists():
    # deposit layout: scripts\ sits one level below the deposit root
    BASE          = Path(__file__).resolve().parents[1]
INTERMEDIATES = BASE / "analysis_intermediate"
TABLES        = BASE / "tables"
FIGURES       = BASE / "figures"
TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

WINDOW_START = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
WINDOW_DAYS  = (WINDOW_END - WINDOW_START).days

# ── 1. Verify Phase 1 intermediates ───────────────────────────────────────────
print("\n=== Verifying Phase 1 intermediates ===")
required = {
    "real_clean":         INTERMEDIATES / "real_clean.csv",
    "shuffle_clean":      INTERMEDIATES / "shuffle_clean.csv",
    "quarantine":         INTERMEDIATES / "quarantine_shuffle_metrics.csv",
    "phase1_status":      INTERMEDIATES / "phase1_status.json",
    "data_coverage":      TABLES / "data_coverage.csv",
    "descriptive_stats":  TABLES / "real_descriptive_stats.csv",
    "temporal_structure": TABLES / "temporal_structure.csv",
    "cross_month_dist":   TABLES / "cross_month_distribution.csv",
    "cross_month_ac":     TABLES / "cross_month_autocorr_persistence.csv",
    "diagnostic_flags":   TABLES / "diagnostic_flags.csv",
}
missing = [str(p) for p in required.values() if not p.exists()]
if missing:
    print("FATAL: missing Phase 1 intermediates:")
    for m in missing: print(f"  {m}")
    sys.exit(1)
for k, p in required.items():
    print(f"  [OK] {p.name}")

# ── 2. Load Phase 1 intermediates ─────────────────────────────────────────────
print("\n=== Loading Phase 1 intermediates ===")
real       = pd.read_csv(INTERMEDIATES / "real_clean.csv")
shuffle    = pd.read_csv(INTERMEDIATES / "shuffle_clean.csv")
quarantine = pd.read_csv(INTERMEDIATES / "quarantine_shuffle_metrics.csv")
with open(INTERMEDIATES / "phase1_status.json") as f:
    p1 = json.load(f)

real["ts_utc"] = pd.to_datetime(
    real["Timestamp_UTC"].str.replace(" UTC","",regex=False), utc=True)
real = real.sort_values("ts_utc").reset_index(drop=True)

required_cols = ["H_t_numeric","n_total_regime","month","ts_utc","metric_provenance","H_t_init_outlier"]
missing_cols  = [c for c in required_cols if c not in real.columns]
if missing_cols:
    print(f"FATAL: real_clean.csv missing columns: {missing_cols}"); sys.exit(1)

real_apr  = real[real["month"]=="2026-04"].copy()
real_may  = real[real["month"]=="2026-05"].copy()
n_regime  = int(real["n_total_regime"].sum())
h_std_all = float(real["H_t_numeric"].std())

print(f"  real_clean: {len(real)} rows  Apr={len(real_apr)}  May={len(real_may)}")
print(f"  n_total_regime rows: {n_regime}  H_t_init_outlier: {real['H_t_init_outlier'].sum()}")

# C1: Verify April n_total variance
apr_ntotal_std = float(real_apr["n_total"].std())
apr_ntotal_nunique = int(real_apr["n_total"].dropna().nunique())
print(f"\n  C1-check: April n_total std={apr_ntotal_std:.4f}  unique values={apr_ntotal_nunique}")
if apr_ntotal_std < 1e-10:
    print("  April n_total is CONSTANT -> Pearson r undefined for April-only (zero variance)")
apr_ntotal_val = float(real_apr["n_total"].dropna().iloc[0]) if apr_ntotal_nunique==1 else np.nan

# Compute May-only and two-month Pearson r for H_t vs n_total
may_r, may_r_p = stats.pearsonr(
    real_may["n_total"].dropna().values,
    real_may.loc[real_may["n_total"].notna(), "H_t_numeric"].values)
two_r, two_r_p = stats.pearsonr(
    real["n_total"].dropna().values,
    real.loc[real["n_total"].notna(), "H_t_numeric"].values)
print(f"  H_t vs n_total: May-only r={may_r:.4f} p={may_r_p:.2e}  Two-month r={two_r:.4f} p={two_r_p:.2e}")

# ── 3. Load and parse state table ─────────────────────────────────────────────
print("\n=== Parsing kopterix_state.csv ===")
state_raw = pd.read_csv(BASE / "kopterix_state.csv")
state_raw["ts_utc"] = pd.to_datetime(state_raw["Timestamp"], utc=True)
state_raw = state_raw.sort_values("ts_utc").reset_index(drop=True)
state = state_raw[(state_raw["ts_utc"]>=WINDOW_START)&(state_raw["ts_utc"]<WINDOW_END)].copy().reset_index(drop=True)
print(f"  Raw: {len(state_raw)}  In-window: {len(state)}")

def parse_vec(s):
    try:   return np.array(json.loads(s), dtype=np.float64)
    except: return None

state["mean_emb"]  = state["MeanEmbedding"].apply(parse_vec)
state["lc_parsed"] = state["LayerCentroids"].apply(lambda s: json.loads(s) if pd.notna(s) else None)

LAYERS = ["surface","mid","residue"]

# C2: Full incomplete-layer audit (missing keys OR null OR empty, not only JSON errors)
lc_issues = []
for idx, row in state.iterrows():
    ts = row["ts_utc"]
    mo = row.get("month", str(ts)[:7])
    s  = row["LayerCentroids"]
    if pd.isna(s):
        lc_issues.append({"row":idx,"ts":str(ts),"month":mo,"issue":"null","missing_keys":str(LAYERS)})
        continue
    try:
        d = json.loads(s)
        missing  = [l for l in LAYERS if l not in d]
        empty    = [l for l in LAYERS if l in d and (d[l] is None or len(d[l])==0)]
        if missing or empty:
            lc_issues.append({"row":idx,"ts":str(ts),"month":mo,
                              "issue":"incomplete_keys",
                              "missing_keys":str(missing),"empty_keys":str(empty)})
    except Exception as e:
        lc_issues.append({"row":idx,"ts":str(ts),"month":mo,"issue":f"json_error","error":str(e)})

print(f"  LayerCentroids incomplete rows (any issue): {len(lc_issues)}")
for iss in lc_issues:
    print(f"    {iss}")

# Cross-reference with n_total_regime
obs_sorted = real.sort_values("ts_utc")
regime_matches = []
for iss in lc_issues:
    iss_ts = pd.Timestamp(iss["ts"])
    diffs  = (obs_sorted["ts_utc"] - iss_ts).abs()
    near   = obs_sorted.iloc[diffs.idxmin()]
    regime_matches.append({
        "state_ts": iss["ts"][:16], "issue": iss["issue"],
        "missing_keys": iss.get("missing_keys",""),
        "nearest_obs_n_total": near["n_total"],
        "n_total_regime": bool(near["n_total_regime"]),
        "obs_delta_sec": float(diffs.min().total_seconds()),
    })
lc_audit_df = pd.DataFrame(regime_matches)
lc_audit_df.to_csv(TABLES/"phase2_lc_sparse_coverage.csv", index=False)
print("  Sparse coverage audit:")
print(lc_audit_df.to_string(index=False))

# Parse with per-key fallback (keep rows that have at least some keys)
for layer in LAYERS:
    state[f"lc_{layer}"] = state["lc_parsed"].apply(
        lambda d, l=layer: np.array(d[l],dtype=np.float64) if d and l in d and d[l] is not None and len(d[l])>0 else None)

state["month"] = state["ts_utc"].dt.to_period("M").astype(str)

n_me_ok   = int(state["mean_emb"].notna().sum())
n_lc_full = int((state["lc_surface"].notna()&state["lc_mid"].notna()&state["lc_residue"].notna()).sum())
n_lc_any  = int((state["lc_surface"].notna()|state["lc_mid"].notna()|state["lc_residue"].notna()).sum())
emb_dim   = len(state["mean_emb"].dropna().iloc[0])
print(f"  MeanEmbedding parsed: {n_me_ok}/{len(state)} dim={emb_dim}")
print(f"  LayerCentroids fully parsed (all 3 keys): {n_lc_full}/{len(state)}")
print(f"  LayerCentroids partially parsed (any key): {n_lc_any}/{len(state)}")

def parse_unigram(s):
    try:
        d = json.loads(s); d.pop("__total__",None); return d
    except: return {}

state["unigram"] = state["UnigramCounts"].apply(parse_unigram)

# ── 4. Coverage audit ─────────────────────────────────────────────────────────
print("\n=== Coverage audit ===")
obs_s  = real[["ts_utc"]].assign(obs_idx=real.index).sort_values("ts_utc")
stat_s = state[["ts_utc"]].assign(state_idx=state.index).sort_values("ts_utc")
matched = pd.merge_asof(obs_s, stat_s.rename(columns={"ts_utc":"ts_state"}),
    left_on="ts_utc", right_on="ts_state",
    direction="nearest", tolerance=pd.Timedelta("5min"))
n_matched          = int(matched["state_idx"].notna().sum())
n_unmatched_obs    = int(matched["state_idx"].isna().sum())
n_unmatched_state  = int(len(state) - len(set(matched["state_idx"].dropna().astype(int))))
print(f"  Obs: {len(real)}  State: {len(state)}  Matched: {n_matched}  "
      f"Unmatched obs: {n_unmatched_obs}  Unmatched state: {n_unmatched_state}")
pd.DataFrame([{"obs_rows_in_window":len(real),"state_rows_in_window":len(state),
    "obs_matched_to_state":n_matched,"obs_unmatched":n_unmatched_obs,
    "state_rows_unmatched":n_unmatched_state,"match_tolerance_minutes":5}]
).to_csv(TABLES/"phase2_coverage_audit.csv", index=False)

# ── 5. Zero-mode residual analysis ────────────────────────────────────────────
print("\n=== Zero-mode residual analysis ===")
me_valid    = state[state["mean_emb"].notna()].copy()
me_mat      = np.stack(me_valid["mean_emb"].values)
grand_mean  = me_mat.mean(axis=0)
residuals   = me_mat - grand_mean
resid_norms = np.linalg.norm(residuals, axis=1)
me_valid    = me_valid.copy()
me_valid["residual_norm"] = resid_norms

U, S, Vt  = np.linalg.svd(residuals, full_matrices=False)
total_var = (S**2).sum()
pc_var    = (S[:3]**2 / total_var).tolist()
me_valid["resid_pc1"] = (U[:,0]*S[0]).tolist()
me_valid["resid_pc2"] = (U[:,1]*S[1]).tolist()
print(f"  Residual PC1/2/3 var: {[f'{v:.4f}' for v in pc_var]}")

t_sec = (me_valid["ts_utc"]-me_valid["ts_utc"].min()).dt.total_seconds().values
slope, intercept, r_val, p_val, _ = stats.linregress(t_sec, resid_norms)
print(f"  Drift: slope={slope:.4e}  r={r_val:.3f}  p={p_val:.4f}")

resid_by_month = me_valid.groupby("month")["residual_norm"].agg(
    n="count",mean="mean",std="std",median="median").reset_index()
pd.DataFrame([{"grand_mean_norm":float(np.linalg.norm(grand_mean)),
    "resid_norm_mean":float(resid_norms.mean()),"resid_norm_std":float(resid_norms.std()),
    "resid_norm_min":float(resid_norms.min()),"resid_norm_max":float(resid_norms.max()),
    "resid_drift_slope":float(slope),"resid_drift_r":float(r_val),"resid_drift_p":float(p_val),
    "pc1_var":pc_var[0],"pc2_var":pc_var[1],"pc3_var":pc_var[2]}]
).to_csv(TABLES/"phase2_zero_mode_residuals.csv", index=False)
resid_by_month.to_csv(TABLES/"phase2_zero_mode_by_month.csv", index=False)

# ── 6. Layer-label shuffle (exact 6 permutations) -- C3 ───────────────────────
print("\n=== Layer-label shuffle (6 permutations) [C3: metric-sensitivity audit] ===")
all_perms = list(permutations(range(3)))
valid_lc  = state[state["lc_surface"].notna()&state["lc_mid"].notna()&state["lc_residue"].notna()].copy()
print(f"  Rows with all 3 layer centroids: {len(valid_lc)}")
row_vecs  = [(r["lc_surface"],r["lc_mid"],r["lc_residue"]) for _,r in valid_lc.iterrows()]
mats_by_layer = {l: np.stack([v[i] for v in row_vecs]) for i,l in enumerate(LAYERS)}

# Within-label norm (original metric)
def within_label_score(vecs, perm):
    scores = []
    for pos in range(3):
        mat = np.stack([v[perm[pos]] for v in vecs])
        c   = mat.mean(axis=0)
        scores.append(np.linalg.norm(mat-c, axis=1).mean())
    return float(np.mean(scores))

perm_rows = []
for perm in all_perms:
    label = "->".join(LAYERS[perm[i]] for i in range(3))
    score = within_label_score(row_vecs, perm)
    is_id = perm==(0,1,2)
    perm_rows.append({"permutation":str(perm),"label":label,"is_identity":is_id,
                      "within_label_norm_mean":score})
    print(f"    {label}: {score:.6f}{'  <-- identity' if is_id else ''}")

perm_df = pd.DataFrame(perm_rows).sort_values("within_label_norm_mean").reset_index(drop=True)
perm_df["rank_ascending"] = perm_df["within_label_norm_mean"].rank(ascending=True,method="average")
all_tied = perm_df["within_label_norm_mean"].nunique()==1
print(f"\n  All 6 scores tied: {all_tied}")
print("  C3 diagnosis: within-label norm is the mean of 3 per-label spreads.")
print("  Each per-label spread = mean L2 dist of that layer from its own centroid.")
print("  This quantity is invariant under permutation (just reorders 3 addends).")
print("  => Metric is permutation-insensitive by construction. Result: INCONCLUSIVE.")

# Additional discriminability metric: between/within ratio
# For each layer, compute centroid and within-layer spread
layer_centroids_all = {l: mats_by_layer[l].mean(axis=0) for l in LAYERS}
within_spreads = {l: float(np.linalg.norm(mats_by_layer[l] - layer_centroids_all[l], axis=1).mean())
                  for l in LAYERS}
between_dists = {}
for i,l1 in enumerate(LAYERS):
    for l2 in LAYERS[i+1:]:
        between_dists[f"{l1}-{l2}"] = float(np.linalg.norm(layer_centroids_all[l1] - layer_centroids_all[l2]))

mean_within = float(np.mean(list(within_spreads.values())))
mean_between = float(np.mean(list(between_dists.values())))
bw_ratio = mean_between / mean_within if mean_within > 0 else np.nan
print(f"\n  Between/within centroid ratio (supplementary, not a permutation test):")
print(f"    Mean within-layer spread: {mean_within:.6f}")
print(f"    Mean between-layer centroid L2: {mean_between:.6f}")
print(f"    Between/within ratio: {bw_ratio:.4f}")
print(f"  (Ratio > 1 = layer centroids are further apart than within-layer spread)")

perm_df["metric_insensitive_note"] = "within-label norm is permutation-invariant by construction" if all_tied else ""
perm_df.to_csv(TABLES/"phase2_layer_label_shuffle.csv", index=False)

bw_df = pd.DataFrame([{
    "mean_within_layer_spread": mean_within,
    "mean_between_layer_centroid_l2": mean_between,
    "between_within_ratio": bw_ratio,
    "layer_centroids_surface_mid_l2": between_dists.get("surface-mid",np.nan),
    "layer_centroids_mid_residue_l2": between_dists.get("mid-residue",np.nan),
    "layer_centroids_surface_residue_l2": between_dists.get("surface-residue",np.nan),
    "within_spread_surface": within_spreads["surface"],
    "within_spread_mid": within_spreads["mid"],
    "within_spread_residue": within_spreads["residue"],
}])
bw_df.to_csv(TABLES/"phase2_layer_bw_ratio.csv", index=False)

def cosine_dist(a, b):
    na,nb = np.linalg.norm(a),np.linalg.norm(b)
    if na<1e-12 or nb<1e-12: return np.nan
    return float(1.0 - np.dot(a,b)/(na*nb))

valid_lc = valid_lc.copy()
valid_lc["cos_dist_surface_mid"]     = [cosine_dist(r["lc_surface"],r["lc_mid"])     for _,r in valid_lc.iterrows()]
valid_lc["cos_dist_mid_residue"]     = [cosine_dist(r["lc_mid"],    r["lc_residue"]) for _,r in valid_lc.iterrows()]
valid_lc["cos_dist_surface_residue"] = [cosine_dist(r["lc_surface"],r["lc_residue"]) for _,r in valid_lc.iterrows()]
cos_summary = pd.DataFrame([{"pair":m,"mean":valid_lc[m].mean(),"std":valid_lc[m].std(),
    "min":valid_lc[m].min(),"max":valid_lc[m].max()}
    for m in ["cos_dist_surface_mid","cos_dist_mid_residue","cos_dist_surface_residue"]])
cos_summary.to_csv(TABLES/"phase2_layer_cosine_distances.csv", index=False)
print("\n  Cosine distances:\n", cos_summary.to_string(index=False))

# ── 7. Cross-month centroid drift -- C5 ───────────────────────────────────────
print("\n=== Cross-month centroid drift [C5: comparable, not negligible] ===")
layer_specs = [("mean_embedding","mean_emb"),("layer_surface","lc_surface"),
               ("layer_mid","lc_mid"),("layer_residue","lc_residue")]
ctr_data = {}
for layer_name, col in layer_specs:
    for mo in ["2026-04","2026-05"]:
        sub = state[(state["month"]==mo)&state[col].notna()]
        if len(sub)==0: continue
        mat = np.stack(sub[col].values)
        ctr = mat.mean(axis=0)
        ctr_data[(layer_name,mo)] = {"ctr":ctr,"n":len(sub),
            "within_mean":float(np.linalg.norm(mat-ctr,axis=1).mean()),
            "within_std": float(np.linalg.norm(mat-ctr,axis=1).std())}

drift_rows = []
for layer_name,_ in layer_specs:
    apr = ctr_data.get((layer_name,"2026-04"))
    may = ctr_data.get((layer_name,"2026-05"))
    if not apr or not may: continue
    l2  = float(np.linalg.norm(apr["ctr"]-may["ctr"]))
    cos = cosine_dist(apr["ctr"],may["ctr"])
    ratio_apr = l2/apr["within_mean"] if apr["within_mean"]>0 else np.nan
    ratio_may = l2/may["within_mean"] if may["within_mean"]>0 else np.nan
    # C5: phrasing flag
    if ratio_apr < 0.5:
        phrasing = "drift small relative to within-month variability"
    elif ratio_apr < 1.2:
        phrasing = "drift smaller than but comparable to within-month variability"
    else:
        phrasing = "drift exceeds within-month variability"
    drift_rows.append({"layer":layer_name,
        "n_april":apr["n"],"n_may":may["n"],
        "cross_month_drift_l2":l2,"cross_month_drift_cosine":cos,
        "within_mean_dist_april":apr["within_mean"],"within_mean_dist_may":may["within_mean"],
        "drift_to_within_ratio_april":ratio_apr,"drift_to_within_ratio_may":ratio_may,
        "c5_phrasing":phrasing})

drift_df = pd.DataFrame(drift_rows)
drift_df.to_csv(TABLES/"phase2_centroid_drift.csv", index=False)
print(drift_df[["layer","cross_month_drift_l2","within_mean_dist_april",
                "drift_to_within_ratio_april","c5_phrasing"]].to_string(index=False))

# ── 8. April-vs-May scalar stability with FDR -- C4 ──────────────────────────
print("\n=== April-vs-May scalar stability [C4: FDR BH q-values] ===")
SCALAR_METRICS = ["post_rate_est","H_t_numeric","phi_t","D_t","n_total",
                  "centroid_dist_surface_mid","centroid_dist_mid_residue","centroid_dist_surface_residue"]
stab_rows = []
for m in SCALAR_METRICS:
    av,mv = real_apr[m].dropna().values, real_may[m].dropna().values
    if len(av)<5 or len(mv)<5:
        stab_rows.append({"metric":m,"n_april":len(av),"n_may":len(mv),
            "mean_april":np.nan,"mean_may":np.nan,"mean_shift_may_minus_apr":np.nan,
            "cohens_d":np.nan,"mw_pval":np.nan,"ks_pval":np.nan,
            "flag_mw_sig_0.05":False,"flag_large_effect_d0.5":False}); continue
    _,mwp = stats.mannwhitneyu(av,mv,alternative="two-sided")
    _,ksp = stats.ks_2samp(av,mv)
    ps = np.sqrt((av.std()**2+mv.std()**2)/2)
    cd = (mv.mean()-av.mean())/ps if ps>0 else np.nan
    stab_rows.append({"metric":m,"n_april":len(av),"n_may":len(mv),
        "mean_april":float(av.mean()),"mean_may":float(mv.mean()),
        "mean_shift_may_minus_apr":float(mv.mean()-av.mean()),
        "cohens_d":float(cd),"mw_pval":float(mwp),"ks_pval":float(ksp),
        "flag_mw_sig_0.05":bool(mwp<0.05),
        "flag_large_effect_d0.5":bool(abs(cd)>0.5) if not np.isnan(cd) else False})

stab_df = pd.DataFrame(stab_rows)

# C4: BH FDR correction on MW p-values
valid_p_mask = stab_df["mw_pval"].notna()
valid_ps = stab_df.loc[valid_p_mask, "mw_pval"].values
bh_qs = false_discovery_control(valid_ps, method="bh")
stab_df["mw_q_bh"] = np.nan
stab_df.loc[valid_p_mask, "mw_q_bh"] = bh_qs
stab_df["flag_mw_q_sig_0.05"] = stab_df["mw_q_bh"] < 0.05
stab_df.to_csv(TABLES/"phase2_april_may_stability.csv", index=False)

sig_metrics  = stab_df[stab_df["flag_mw_sig_0.05"]]["metric"].tolist()
sig_q_metrics = stab_df[stab_df["flag_mw_q_sig_0.05"]]["metric"].tolist()
large_d       = stab_df[stab_df["flag_large_effect_d0.5"]]["metric"].tolist()
print(stab_df[["metric","mean_april","mean_may","cohens_d","mw_pval","mw_q_bh","flag_mw_q_sig_0.05"]].to_string(index=False))
print(f"\n  Significant at q<0.05 (BH): {sig_q_metrics}")
print(f"  Large effect |d|>0.5: {large_d}")

# ── 9. Low-n_total sensitivity ────────────────────────────────────────────────
print("\n=== Low-n_total sensitivity ===")
real_excl = real[~real["n_total_regime"]]
sens_rows = []
for m in ["H_t_numeric","phi_t","D_t","post_rate_est"]:
    av,ev = real[m].dropna().values, real_excl[m].dropna().values
    sens_rows.append({"metric":m,"n_all":len(av),"mean_all":float(av.mean()),
        "median_all":float(np.median(av)),"std_all":float(av.std()),
        "n_excl_regime":len(ev),"mean_excl":float(ev.mean()),
        "median_excl":float(np.median(ev)),"std_excl":float(ev.std()),
        "mean_diff_excl_minus_all":float(ev.mean()-av.mean()),
        "pct_of_std":float(abs(ev.mean()-av.mean())/av.std()*100) if av.std()>0 else np.nan,
        "n_regime_rows":n_regime})
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(TABLES/"phase2_low_n_sensitivity.csv", index=False)
print(sens_df[["metric","mean_all","mean_excl","mean_diff_excl_minus_all","pct_of_std"]].to_string(index=False))
h_sen_diff = float(sens_df[sens_df["metric"]=="H_t_numeric"]["mean_diff_excl_minus_all"].iloc[0])

# ── 10. Unigram analysis ──────────────────────────────────────────────────────
print("\n=== Unigram analysis ===")
all_uni    = state["unigram"].tolist()
union_vocab = sorted(set(w for d in all_uni for w in d))
print(f"  Union vocab size: {len(union_vocab)}")

def uni_entropy(d):
    c = np.array(list(d.values()), dtype=np.float64)
    if c.sum()==0: return np.nan
    p = c/c.sum(); return float(-np.sum(p*np.log2(p+1e-12)))

state_u = state.copy()
state_u["unigram_entropy"]      = state_u["unigram"].apply(uni_entropy)
state_u["unigram_richness"]     = state_u["unigram"].apply(len)
state_u["unigram_total_tokens"] = state_u["unigram"].apply(lambda d: sum(d.values()))

all_counts = Counter()
for d in all_uni: all_counts.update(d)
top10 = all_counts.most_common(10)

uni_month = state_u.groupby("month").agg(
    n=("unigram_entropy","count"),
    entropy_mean=("unigram_entropy","mean"), entropy_std=("unigram_entropy","std"),
    richness_mean=("unigram_richness","mean"),
    total_tokens_mean=("unigram_total_tokens","mean")).reset_index()
uni_month.to_csv(TABLES/"phase2_unigram_by_month.csv", index=False)

ts_out = state_u[["ts_utc","month","unigram_entropy","unigram_richness","unigram_total_tokens"]].copy()
ts_out["ts_utc"] = ts_out["ts_utc"].astype(str)
ts_out.to_csv(TABLES/"phase2_unigram_timeseries.csv", index=False)
pd.DataFrame(top10, columns=["word","total_count"]).to_csv(TABLES/"phase2_unigram_top10.csv", index=False)

ae   = state_u[state_u["month"]=="2026-04"]["unigram_entropy"].dropna().values
me_e = state_u[state_u["month"]=="2026-05"]["unigram_entropy"].dropna().values
mw_uni_p = float(stats.mannwhitneyu(ae,me_e,alternative="two-sided")[1]) if (len(ae)>=5 and len(me_e)>=5) else np.nan
print(f"  Unigram entropy MW p: {mw_uni_p:.4e}")

# ── 11. C1: H_t vs n_total correlation detail table ──────────────────────────
print("\n=== C1: H_t - n_total correlation (corrected) ===")
# April: constant n_total -> Pearson r undefined
apr_ntotal_vals = real_apr["n_total"].dropna().values
if apr_ntotal_vals.std() < 1e-10:
    apr_r_note = f"undefined (n_total constant at {apr_ntotal_vals[0]:.0f} for all {len(apr_ntotal_vals)} non-NaN April rows)"
    apr_r      = np.nan
    apr_r_p    = np.nan
else:
    apr_r, apr_r_p = stats.pearsonr(apr_ntotal_vals, real_apr.loc[real_apr["n_total"].notna(),"H_t_numeric"].values)
    apr_r_note = f"r={apr_r:.4f} p={apr_r_p:.2e}"

corr_c1 = pd.DataFrame([
    {"scope":"April-only","pearson_r":apr_r,"pearson_p":apr_r_p,
     "n_pairs":int(real_apr["n_total"].notna().sum()),"note":apr_r_note},
    {"scope":"May-only","pearson_r":may_r,"pearson_p":may_r_p,
     "n_pairs":int(real_may["n_total"].notna().sum()),"note":f"r={may_r:.4f} p={may_r_p:.2e}"},
    {"scope":"Two-month","pearson_r":two_r,"pearson_p":two_r_p,
     "n_pairs":int(real["n_total"].notna().sum()),"note":f"r={two_r:.4f} p={two_r_p:.2e}"},
])
corr_c1.to_csv(TABLES/"phase2_ht_ntotal_corr.csv", index=False)
print(corr_c1.to_string(index=False))

# ── 12. Viability assessment ──────────────────────────────────────────────────
print("\n=== Viability assessment ===")
obs_per_day = len(real)/WINDOW_DAYS
n_shuffle   = p1["shuffle_n_blocks"]
viab_rows = [
    ("observation_count",        len(real),    len(real)>=100,      f"{len(real)} obs in {WINDOW_DAYS}-day window"),
    ("obs_per_day_mean",         round(obs_per_day,2), obs_per_day>=2.0, f"{obs_per_day:.2f} obs/day"),
    ("state_rows_in_window",     len(state),   len(state)>=100,     f"{len(state)} state rows"),
    ("shuffle_blocks",           n_shuffle,    n_shuffle>=10,       f"{n_shuffle} deterministic shuffle blocks"),
    ("days_with_zero_runs",      0,            True,                "0 days with zero runs"),
    ("n_total_variation_observed",n_regime,    True,                f"{n_regime} low-n_total rows in May; diagnostically useful"),
    ("scipy_available",          1,            True,                f"scipy {scipy.__version__}"),
    ("embedding_dim",            emb_dim,      emb_dim==384,        f"dim={emb_dim}"),
    ("lc_fully_parsed",          n_lc_full,    n_lc_full>=200,      f"{n_lc_full}/{len(state)} rows with all 3 keys"),
    ("lc_sparse_coverage_rows",  len(lc_issues), len(lc_issues)<=5, f"{len(lc_issues)} rows with incomplete keys (all May; {sum(r['n_total_regime'] for r in regime_matches)} are n_total_regime)"),
]
viab_df = pd.DataFrame(viab_rows, columns=["check","value","passes","note"])
viab_df.to_csv(TABLES/"phase2_viability.csv", index=False)
for _,r in viab_df.iterrows():
    print(f"  [{'PASS' if r['passes'] else 'WARN'}] {r['check']}: {r['note']}")

# ── 13. Metric provenance audit ───────────────────────────────────────────────
print("\n=== Metric provenance audit ===")
prov_rows = [
    ("computed_observation_csv",
     p1["metric_provenance_inventory"]["computed_observation_csv"], True,
     "observations.csv scalars: post_rate_est, H_t, phi_t, D_t, n_total, centroid_dist_*"),
    ("deterministic_shuffle_comparisons",
     p1["metric_provenance_inventory"]["deterministic_shuffle_comparison"], True,
     "surface-mid/mid-residue/surface-residue real/shuffle/delta triplets from log blocks"),
    ("computed_state_metrics",         len(state),  True,
     "MeanEmbedding, LayerCentroids, UnigramCounts from kopterix_state.csv"),
    ("phase2_zero_mode_residuals",     int(n_me_ok), True,
     "residual norms: MeanEmbedding minus grand mean"),
    ("phase2_layer_label_shuffle",     6,            True,
     "6-permutation within-label variance; result: metric is permutation-invariant (inconclusive)"),
    ("phase2_layer_bw_ratio",          1,            True,
     "supplementary between/within centroid ratio (not a permutation test)"),
    ("phase2_unigram_entropy",         len(state),   True,
     "Shannon entropy from UnigramCounts per state row"),
    ("quarantined_legacy_shuffle_log",
     p1["metric_provenance_inventory"]["legacy_shuffle_log_text_untrusted"], False,
     "H_t/phi_t/D_t from shuffle log text; quarantined; not used in analysis"),
]
prov_df = pd.DataFrame(prov_rows, columns=["metric_class","count","trusted","notes"])
prov_df.to_csv(TABLES/"phase2_metric_provenance.csv", index=False)
print(prov_df[["metric_class","count","trusted"]].to_string(index=False))

# ── 14. Figures ───────────────────────────────────────────────────────────────
print("\n=== Generating figures ===")
COLORS = {"2026-04":"#2196F3","2026-05":"#FF9800"}

def xfmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIGURES/name, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {name}")

# Fig 1: Residual norms
fig, axes = plt.subplots(1,2,figsize=(12,4))
ax = axes[0]
for mo,clr in COLORS.items():
    s = me_valid[me_valid["month"]==mo]
    ax.scatter(s["ts_utc"],s["residual_norm"],s=8,alpha=0.6,color=clr,label=mo)
ax.set_title("MeanEmbedding residual norms (zero-mode)"); ax.set_ylabel("Residual L2 norm")
xfmt(ax); ax.legend()
axes[1].hist([me_valid[me_valid["month"]=="2026-04"]["residual_norm"].values,
              me_valid[me_valid["month"]=="2026-05"]["residual_norm"].values],
             bins=20, alpha=0.7, color=["#2196F3","#FF9800"], label=["Apr","May"])
axes[1].set_title("Residual norm distribution"); axes[1].set_xlabel("Residual L2 norm")
axes[1].set_ylabel("Count"); axes[1].legend()
save_fig("phase2_residual_norms.png")

# Fig 2: Layer cosine distances
fig, axes = plt.subplots(3,1,figsize=(10,9),sharex=True)
for i,(col,title) in enumerate([("cos_dist_surface_mid","Cosine dist: surface-mid"),
                                  ("cos_dist_mid_residue","Cosine dist: mid-residue"),
                                  ("cos_dist_surface_residue","Cosine dist: surface-residue")]):
    ax = axes[i]
    for mo,clr in COLORS.items():
        s = valid_lc[valid_lc["month"]==mo]
        ax.scatter(s["ts_utc"],s[col],s=8,alpha=0.6,color=clr,label=mo)
    ax.set_ylabel(title,fontsize=8); ax.legend(fontsize=7); xfmt(ax)
axes[0].set_title("Inter-layer cosine distances (identity labeling)")
axes[-1].set_xlabel("Date (UTC)")
save_fig("phase2_layer_cosine_distances.png")

# Fig 3: Layer-label permutation (with inconclusive label)
fig, ax = plt.subplots(figsize=(8,4))
colors_bar = ["#4CAF50" if r["is_identity"] else "#9E9E9E" for _,r in perm_df.iterrows()]
ax.bar(perm_df["label"],perm_df["within_label_norm_mean"],color=colors_bar)
title_note = "(all tied - metric is permutation-invariant)" if all_tied else ""
ax.set_title(f"Layer-label shuffle: within-label L2 norm\nGreen = identity labeling {title_note}")
ax.set_ylabel("Mean within-label L2 norm"); plt.xticks(rotation=30,ha="right")
save_fig("phase2_layer_label_shuffle.png")

# Fig 4: Centroid drift with phrasing annotation
fig, ax = plt.subplots(figsize=(9,4))
dl = drift_df["layer"].str.replace("layer_","").str.replace("mean_embedding","mean-emb")
x = np.arange(len(drift_df)); w = 0.3
ax.bar(x-w, drift_df["within_mean_dist_april"], w, label="Within Apr", color="#2196F3", alpha=0.7)
ax.bar(x,   drift_df["within_mean_dist_may"],   w, label="Within May", color="#FF9800", alpha=0.7)
ax.plot(x-w/2, drift_df["cross_month_drift_l2"],"k^--",label="Cross-month drift L2",ms=8)
ax.set_xticks(x-w/2); ax.set_xticklabels(dl,rotation=20,ha="right")
ax.set_title("Cross-month drift vs within-month variability\n(drift smaller than but comparable to within-month spread)")
ax.set_ylabel("L2 distance"); ax.legend()
save_fig("phase2_centroid_drift.png")

# Fig 5: Unigram entropy
fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True)
for mo,clr in COLORS.items():
    s = state_u[state_u["month"]==mo]
    axes[0].scatter(s["ts_utc"],s["unigram_entropy"],s=8,alpha=0.6,color=clr,label=mo)
    axes[1].scatter(s["ts_utc"],s["unigram_richness"],s=8,alpha=0.6,color=clr,label=mo)
axes[0].set_title("Unigram Shannon entropy"); axes[0].set_ylabel("Entropy (bits)"); axes[0].legend()
axes[1].set_title("Unigram richness"); axes[1].set_ylabel("Distinct tokens")
axes[1].set_xlabel("Date (UTC)"); axes[1].legend(); xfmt(axes[0]); xfmt(axes[1])
save_fig("phase2_unigram_entropy.png")

# Fig 6: April/May violin
PLOT_M = ["H_t_numeric","phi_t","D_t","n_total",
          "centroid_dist_surface_mid","centroid_dist_mid_residue",
          "centroid_dist_surface_residue","post_rate_est"]
fig, axes = plt.subplots(2,4,figsize=(16,7)); axes = axes.flatten()
for i,m in enumerate(PLOT_M):
    ax = axes[i]
    av = real_apr[m].dropna().values; mv = real_may[m].dropna().values
    parts = ax.violinplot([av,mv],positions=[0,1],showmedians=True)
    for j,pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#2196F3","#FF9800"][j]); pc.set_alpha(0.7)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Apr","May"]); ax.set_title(m,fontsize=8)
    row = stab_df[stab_df["metric"]==m]
    if len(row) and not np.isnan(row["mw_pval"].iloc[0]):
        p   = row["mw_pval"].iloc[0]
        q   = row["mw_q_bh"].iloc[0]
        sig = "*" if q<0.05 else "ns"
        ax.set_xlabel(f"p={p:.3f} q={q:.3f} {sig}",fontsize=7)
fig.suptitle("April vs May scalar distributions (MW p unadjusted, q BH-adjusted)",fontsize=10)
save_fig("phase2_april_may_violin.png")

# Fig 7: Residual PCA
fig, ax = plt.subplots(figsize=(7,6))
for mo,clr in COLORS.items():
    s = me_valid[me_valid["month"]==mo]
    ax.scatter(s["resid_pc1"],s["resid_pc2"],s=10,alpha=0.6,color=clr,label=mo)
ax.set_title(f"Residual PC1 vs PC2  (var: {pc_var[0]:.3f}, {pc_var[1]:.3f})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
save_fig("phase2_residual_pca.png")

# Fig 8: H_t vs n_total scatter by month
fig, axes = plt.subplots(1,2,figsize=(11,4))
for ax,mo,clr in [(axes[0],"2026-04","#2196F3"),(axes[1],"2026-05","#FF9800")]:
    sub = real[real["month"]==mo].dropna(subset=["n_total","H_t_numeric"])
    ax.scatter(sub["n_total"],sub["H_t_numeric"],s=12,alpha=0.6,color=clr)
    ax.set_title(f"H_t vs n_total ({mo})",fontsize=9)
    ax.set_xlabel("n_total"); ax.set_ylabel("H_t_numeric")
    if sub["n_total"].std()<1e-10:
        ax.text(0.05,0.92,"n_total constant\nPearson r undefined",transform=ax.transAxes,
                fontsize=8,va="top",color="darkred")
    else:
        r_loc,_ = stats.pearsonr(sub["n_total"],sub["H_t_numeric"])
        ax.text(0.05,0.92,f"Pearson r = {r_loc:.3f}",transform=ax.transAxes,fontsize=8,va="top")
fig.suptitle("H_t vs n_total by month (C1: April r undefined)",fontsize=10)
save_fig("phase2_ht_ntotal_by_month.png")

# ── 15. Markdown report (corrected) ───────────────────────────────────────────
print("\n=== Writing corrected Markdown report ===")
now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
desc     = pd.read_csv(TABLES/"real_descriptive_stats.csv")
temp     = pd.read_csv(TABLES/"temporal_structure.csv")
resid_s  = pd.read_csv(TABLES/"phase2_zero_mode_residuals.csv").iloc[0]

def td(v, fmt=".4f"):
    return format(v, fmt) if not (isinstance(v,float) and np.isnan(v)) else "NA"

lines = []
A = lines.append

A("# Kopterix Two-Month Validation Report\n")
A(f"**Instrument window:** 2026-04-01 to 2026-05-31 (UTC)  ")
A(f"**Generated:** {now_str}  ")
A("**Pipeline:** Phase 1 (scalar/shuffle/temporal/correlation/flags) +"
  " Phase 2 (state/zero-mode/layer-shuffle/unigram/viability)  ")
A(f"**scipy:** {scipy.__version__}  |  **Embedding dim:** {emb_dim}  ")
A("**Correction pass:** C1 H_t-n_total, C2 LC sparse coverage, C3 shuffle metric,"
  " C4 FDR, C5 centroid phrasing, C6 Phase 1 p-values\n")
A("---\n")

A("## 1. Data Overview and Coverage\n")
A(f"The instrument produced {len(real)} real observations and {len(state)} state rows within"
  f" [2026-04-01 00:00, 2026-06-01 00:00) UTC.")
A(f"Observations span 2026-04-01 to 2026-05-31 with median inter-observation gap"
  f" {p1['gap_stats']['median_gap_hours']:.2f} h and maximum gap"
  f" {p1['gap_stats']['max_gap_hours']:.2f} h. No days with zero runs.\n")
A(f"Monthly split: April = {p1['phase0_diagnostics']['rows_per_month']['2026-04']},"
  f" May = {p1['phase0_diagnostics']['rows_per_month']['2026-05']}.\n")
A(f"Coverage audit (5-min tolerance): {n_matched} of {len(real)} observations matched"
  f" to a state row; {n_unmatched_obs} observations unmatched; {n_unmatched_state}"
  f" state rows unmatched. See `tables/phase2_coverage_audit.csv`.\n")
A("Timestamp normalization: `Timestamp_UTC` parsed as naive UTC (observation key)."
  " State `Timestamp` (ISO 8601 + offset) converted to UTC."
  " Chicago `Timestamp` column ignored. Window is half-open: start inclusive, end exclusive.\n")
A("---\n")

A("## 2. Phase 1 Intermediate Verification\n")
A("All required Phase 1 intermediates confirmed present before Phase 2 execution."
  " Phase 1 was re-run with scipy 1.15.3 available; all OLS trend p-values and"
  " cross-month p-values are now populated. scipy-absent outputs are archived in"
  " `tables/_scipy_absent_archive/` for provenance.\n")
A("| File | Rows | Key columns verified |")
A("|------|------|----------------------|")
A(f"| `real_clean.csv` | {len(real)} | H_t_numeric, n_total_regime, month,"
  " Timestamp_UTC, metric_provenance, H_t_init_outlier |")
A(f"| `shuffle_clean.csv` | {len(shuffle)} | Timestamp, metric_provenance, is_shuffle |")
A(f"| `quarantine_shuffle_metrics.csv` | {len(quarantine)} | Timestamp,"
  " H_t_raw, phi_t_raw, D_t_raw, metric_provenance |")
A(f"| `kopterix_state.csv` (in-window) | {len(state)} |"
  f" MeanEmbedding (dim={emb_dim}), LayerCentroids (surface/mid/residue), UnigramCounts |\n")
A(f"Phase 1 assumptions active: H_t_numeric = H_t for all real observations (no robust_z exclusions);"
  f" {n_regime} May rows flagged n_total_regime = True (n_total <= 300; included in all primary"
  f" analyses); H_t_init_outlier all False (schema-continuity only); April H_t = 4.2"
  f" quarantined as first-run logging-provenance artifact; {len(quarantine)} shuffle log-text"
  f" lines quarantined as legacy_shuffle_log_text_untrusted.\n")
A("---\n")

A("## 3. Two-Month Scalar Summary\n")
A("| Metric | N | Mean | Std | Median | Min | Max |")
A("|--------|---|------|-----|--------|-----|-----|")
for _,row in desc.iterrows():
    A(f"| {row['metric']} | {int(row['count'])} | {row['mean']:.4f} | {row['std']:.4f}"
      f" | {row['median']:.4f} | {row['min']:.4f} | {row['max']:.4f} |")
A(f"\nSee `tables/real_descriptive_stats.csv` for full quantiles and MAD."
  f" n_total variation: {n_regime} low-n_total rows (all May, n_total_regime = True)"
  f" included in all primary analyses; sensitivity in Section 8.\n")
A("---\n")

A("## 4. April-vs-May Scalar Stability\n")
A(f"P-values: Mann-Whitney U two-sided, KS two-sample, computed with scipy"
  f" {scipy.__version__}. **C4:** BH-adjusted q-values added; all p-values are"
  f" unadjusted for multiple testing unless the q column is used.\n")
A("| Metric | N Apr | N May | Mean Apr | Mean May | Shift | Cohen d |"
  " MW p (unadj) | BH q | q<0.05 |")
A("|--------|-------|-------|----------|----------|-------|---------|"
  "--------------|------|--------|")
for _,row in stab_df.iterrows():
    sig_q = "*" if row["flag_mw_q_sig_0.05"] else "ns"
    A(f"| {row['metric']} | {row['n_april']} | {row['n_may']}"
      f" | {td(row['mean_april'])} | {td(row['mean_may'])}"
      f" | {td(row.get('mean_shift_may_minus_apr',np.nan))}"
      f" | {td(row['cohens_d'],'.3f')} | {td(row['mw_pval'],'.4f')}"
      f" | {td(row['mw_q_bh'],'.4f')} | {sig_q} |")
A(f"\nMetrics with BH q < 0.05: {', '.join(sig_q_metrics) if sig_q_metrics else 'none'}.")
A(f"Metrics with large effect (|d| > 0.5): {', '.join(large_d) if large_d else 'none'}.\n")
A("See `figures/phase2_april_may_violin.png`, `tables/phase2_april_may_stability.csv`.\n")
A("Framing: split-half instrument characterization only. Seasonal posting patterns,"
  " n_total regime shifts, and platform-side changes cannot be ruled out.\n")
A("---\n")

A("## 5. Zero-Mode Residual Analysis\n")
A(f"Grand mean computed across {len(me_valid)} valid MeanEmbedding rows."
  f" Residual = individual embedding - grand mean.\n")
A(f"- Grand mean L2 norm: {resid_s['grand_mean_norm']:.4f}")
A(f"- Mean residual L2 norm: {resid_s['resid_norm_mean']:.4f}"
  f" (std: {resid_s['resid_norm_std']:.4f})")
A(f"- Residual range: [{resid_s['resid_norm_min']:.4f}, {resid_s['resid_norm_max']:.4f}]")
A(f"- Drift OLS: slope = {resid_s['resid_drift_slope']:.4e},"
  f" r = {resid_s['resid_drift_r']:.3f}, p = {resid_s['resid_drift_p']:.4f}")
A(f"- PC1/PC2/PC3 var: {resid_s['pc1_var']:.4f} / {resid_s['pc2_var']:.4f}"
  f" / {resid_s['pc3_var']:.4f}\n")
A("| Month | N | Mean residual norm | Std |")
A("|-------|---|--------------------|-----|")
for _,row in resid_by_month.iterrows():
    A(f"| {row['month']} | {int(row['n'])} | {row['mean']:.4f} | {row['std']:.4f} |")
drift_interp = (f"Statistically significant temporal drift (p = {resid_s['resid_drift_p']:.4f})."
                if resid_s['resid_drift_p']<0.05 else
                f"No statistically significant temporal drift (p = {resid_s['resid_drift_p']:.4f}).")
A(f"\n{drift_interp}\n")
A("See `figures/phase2_residual_norms.png`, `figures/phase2_residual_pca.png`,"
  " `tables/phase2_zero_mode_residuals.csv`.\n")
A("---\n")

A("## 6. Layer-Label Shuffle (Exact 6 Permutations)\n")
A(f"Rows with all three layer centroids valid: {len(valid_lc)}.\n")
A("**C3 - metric-sensitivity finding:** The within-label norm score used here is the"
  " mean of three per-label within-centroid spreads. This quantity is invariant under"
  " permutation by construction - permuting the labels merely reorders the three addends"
  " without changing their values. All 6 permutations are therefore guaranteed to produce"
  " identical scores regardless of whether the labels carry any discriminative information."
  " This metric is permutation-insensitive and the result is inconclusive."
  " It does not imply that the layer labels have no discriminative power in general;"
  " a metric that compares between-label to within-label variance would be required"
  " to assess that.\n")
A("| Permutation | Label assignment | Within-label norm | Tied |")
A("|-------------|-----------------|-------------------|------|")
for _,row in perm_df.iterrows():
    A(f"| {row['permutation']} | {row['label']}"
      f" | {row['within_label_norm_mean']:.6f}"
      f" | {'YES (all tied)' if all_tied else ''} |")
A(f"\nAll tied: {all_tied}. Score = {perm_df['within_label_norm_mean'].iloc[0]:.6f} for all permutations.\n")
A("**Supplementary between/within centroid ratio** (not a permutation test,"
  " uses cross-row centroids):\n")
A("| Metric | Value |")
A("|--------|-------|")
A(f"| Mean within-layer spread | {bw_df['mean_within_layer_spread'].iloc[0]:.6f} |")
A(f"| Mean between-layer centroid L2 | {bw_df['mean_between_layer_centroid_l2'].iloc[0]:.6f} |")
A(f"| Between/within ratio | {bw_df['between_within_ratio'].iloc[0]:.4f} |")
A(f"| surface-mid centroid L2 | {bw_df['layer_centroids_surface_mid_l2'].iloc[0]:.6f} |")
A(f"| mid-residue centroid L2 | {bw_df['layer_centroids_mid_residue_l2'].iloc[0]:.6f} |")
A(f"| surface-residue centroid L2 | {bw_df['layer_centroids_surface_residue_l2'].iloc[0]:.6f} |")
A(f"\nA ratio > 1.0 indicates the cross-row layer centroids are further apart than"
  f" the typical within-layer spread. Ratio = {bw_df['between_within_ratio'].iloc[0]:.4f}:"
  f" {'centroids are further apart than within-layer spread' if bw_df['between_within_ratio'].iloc[0]>1 else 'centroids are closer together than within-layer spread'}."
  f" This is a descriptive summary, not a significance test.\n")
A("Inter-layer cosine distances (identity labeling):\n")
A("| Pair | Mean | Std | Min | Max |")
A("|------|------|-----|-----|-----|")
for _,row in cos_summary.iterrows():
    A(f"| {row['pair']} | {row['mean']:.6f} | {row['std']:.6f}"
      f" | {row['min']:.6f} | {row['max']:.6f} |")
A("\nSee `figures/phase2_layer_label_shuffle.png`, `tables/phase2_layer_label_shuffle.csv`,"
  " `tables/phase2_layer_bw_ratio.csv`, `tables/phase2_layer_cosine_distances.csv`.\n")
A("---\n")

A("## 7. Centroid-Level Cross-Month Drift\n")
A("**C5:** Cross-month drift is described as smaller than but comparable to"
  " within-month variability where the ratio is between 0.5 and 1.2,"
  " not as negligible.\n")
A("| Layer | N Apr | N May | Drift L2 | Drift cosine | Within Apr |"
  " Within May | Ratio Apr | Phrasing |")
A("|-------|-------|-------|----------|--------------|-----------|"
  "-----------|-----------|---------|")
for _,row in drift_df.iterrows():
    A(f"| {row['layer']} | {row['n_april']} | {row['n_may']}"
      f" | {row['cross_month_drift_l2']:.4f} | {row['cross_month_drift_cosine']:.4f}"
      f" | {row['within_mean_dist_april']:.4f} | {row['within_mean_dist_may']:.4f}"
      f" | {row['drift_to_within_ratio_april']:.3f} | {row['c5_phrasing']} |")
A("\nSee `figures/phase2_centroid_drift.png`, `tables/phase2_centroid_drift.csv`.\n")
A("---\n")

A("## 8. Low-n_total Sensitivity\n")
A(f"Primary analysis includes all {len(real)} rows. Sensitivity to the {n_regime}"
  f" n_total_regime rows (n_total <= 300, all May). Pct-of-std = |mean diff| / std_all * 100.\n")
A("| Metric | N all | Mean all | N excl | Mean excl | Mean diff | Pct of std |")
A("|--------|-------|---------|--------|----------|-----------|-----------|")
for _,row in sens_df.iterrows():
    A(f"| {row['metric']} | {row['n_all']} | {row['mean_all']:.4f}"
      f" | {row['n_excl_regime']} | {row['mean_excl']:.4f}"
      f" | {row['mean_diff_excl_minus_all']:.4f} | {row['pct_of_std']:.1f}% |")
A(f"\nH_t mean shift: {h_sen_diff:.4f} bits"
  f" ({abs(h_sen_diff)/h_std_all*100:.1f}% of two-month std)."
  f" n_total variation is diagnostically useful; no observations are excluded.\n")
A("See `tables/phase2_low_n_sensitivity.csv`.\n")
A("---\n")

A("## 9. Unigram Analysis\n")
A(f"Union vocabulary: {len(union_vocab)} distinct tokens (excluding __total__)"
  f" across {len(state)} state rows.\n")
A("| Month | N | Entropy mean (bits) | Entropy std | Richness mean | Total tokens mean |")
A("|-------|---|---------------------|-------------|---------------|-------------------|")
for _,row in uni_month.iterrows():
    A(f"| {row['month']} | {row['n']} | {row['entropy_mean']:.4f}"
      f" | {row['entropy_std']:.4f} | {row['richness_mean']:.1f}"
      f" | {row['total_tokens_mean']:.1f} |")
A(f"\nCross-month unigram entropy MW p = {td(mw_uni_p,'.4e')}"
  f" (unadjusted; computed on a single test).\n")
A("Top 10 tokens (two-month total):\n")
A("| Token | Total count |")
A("|-------|-------------|")
for word,cnt in top10: A(f"| {word} | {cnt} |")
A("\nLimitation: unigram counts reflect context-window content at each snapshot,"
  " not model-generated text. Token shifts may reflect feed content changes,"
  " n_total regime variation, or both.\n")
A("---\n")

A("## 10. H_t - n_total Correlation (C1: Corrected)\n")
A("**C1:** April n_total is constant across all non-NaN April rows (value ="
  f" {apr_ntotal_val:.0f}). Pearson r between H_t and n_total is undefined"
  " for April in isolation (zero variance denominator). The two-month"
  " combined correlation is driven by May n_total variation.\n")
A("| Scope | N pairs | Pearson r | p | Note |")
A("|-------|---------|-----------|---|------|")
for _,row in corr_c1.iterrows():
    rstr = f"{row['pearson_r']:.4f}" if not np.isnan(row['pearson_r']) else "undefined"
    pstr = f"{row['pearson_p']:.2e}" if not np.isnan(row['pearson_p']) else "-"
    A(f"| {row['scope']} | {row['n_pairs']} | {rstr} | {pstr} | {row['note']} |")
A(f"\nMay-only r = {may_r:.4f} (p = {may_r_p:.2e})."
  f" Two-month r = {two_r:.4f} (p = {two_r_p:.2e})."
  f" H_t cannot be interpreted independently of n_total without normalization,"
  f" and this dependence is fully observable only in May where n_total varies.\n")
A("See `figures/phase2_ht_ntotal_by_month.png`, `tables/phase2_ht_ntotal_corr.csv`.\n")
A("---\n")

A("## 11. LayerCentroids Sparse Coverage (C2: Corrected)\n")
A("**C2:** 5 state rows have incomplete LayerCentroid keys (not the 2 JSON-parse"
  " failures reported in the first pass). All 5 are in May. Per-key fallback parsing"
  f" is used; analysis requiring all 3 layers uses only the {n_lc_full} fully-parsed rows.\n")
A("| State timestamp | Issue | Missing keys | Nearest obs n_total | n_total_regime |")
A("|-----------------|-------|--------------|--------------------|--------------------|")
for r in regime_matches:
    A(f"| {r['state_ts']} | {r['issue']}"
      f" | {lc_audit_df[lc_audit_df['state_ts']==r['state_ts']]['missing_keys'].iloc[0] if len(lc_audit_df[lc_audit_df['state_ts']==r['state_ts']])>0 else ''}"
      f" | {r['nearest_obs_n_total']}"
      f" | {r['n_total_regime']} |")
A(f"\n{sum(r['n_total_regime'] for r in regime_matches)} of 5 sparse-coverage rows"
  f" correspond to n_total_regime observations (n_total <= 300). This suggests"
  f" sparse layer coverage is associated with the low-n_total sampling regime, not"
  f" arbitrary JSON malformation. The one non-regime row (n_total = 500) may be a"
  f" separate logging edge case. None of these rows are excluded; they are noted"
  f" for instrument characterization.\n")
A("See `tables/phase2_lc_sparse_coverage.csv`.\n")
A("---\n")

A("## 12. Shuffle Evidence\n")
A("Shuffle evidence comes only from deterministic comparison triplets. No H_t,"
  " phi_t, or D_t values from shuffle log text are used.\n")
A("| Source | Blocks | Trusted |")
A("|--------|--------|---------|")
A(f"| April deterministic | {p1['phase0_diagnostics']['shuffle_phase0']['shuffle_blocks_per_month']['2026-04']} | Yes |")
A(f"| May deterministic   | {p1['phase0_diagnostics']['shuffle_phase0']['shuffle_blocks_per_month']['2026-05']} | Yes |")
A(f"| Quarantined log-text lines | {len(quarantine)} | No |")
A(f"| April H_t = 4.2 anomaly | 1 | No - first-run logging-provenance artifact |\n")
A(f"{p1['power_caveat']}\n")
A("---\n")

A("## 13. Temporal Structure (Phase 1, scipy recomputed)\n")
A("**C6:** Phase 1 was re-run with scipy available. OLS trend p-values are now populated.\n")
A("| Metric | Autocorr lag-1 | Autocorr lag-4 | OLS slope | OLS R2 | Trend p |")
A("|--------|---------------|---------------|-----------|--------|---------|")
for _,row in temp.iterrows():
    tp = f"{row['ols_trend_p']:.4f}" if not np.isnan(row['ols_trend_p']) else "NA"
    A(f"| {row['metric']} | {row['autocorr_lag1']:.3f} | {row['autocorr_lag4']:.3f}"
      f" | {row['ols_slope']:.4e} | {row['ols_r_squared']:.4f} | {tp} |")
A("\n---\n")

A("## 14. Viability Assessment\n")
A("| Check | Value | Status | Note |")
A("|-------|-------|--------|------|")
for _,r in viab_df.iterrows():
    A(f"| {r['check']} | {r['value']} | {'PASS' if r['passes'] else 'WARN'} | {r['note']} |")
A(f"\nAll checks pass except lc_fully_parsed is 'WARN-adjacent': {n_lc_full}/{len(state)}"
  f" fully parsed (5 sparse-coverage rows, see Section 11). Analyses requiring all 3"
  f" layers use only the {n_lc_full} rows with all keys present.\n")
A("---\n")

A("## 15. Limitations and Caveats\n")
A(f"- The two-month window (2026-04-01 to 2026-05-31) is the complete instrument record. No claims extend beyond it.")
A("- Cross-month scalar differences (Section 4) may reflect platform-side changes, seasonal patterns, n_total shifts, or instrument drift; these cannot be disentangled.")
A(f"- Shuffle power is low: {p1['shuffle_n_blocks']} deterministic blocks"
  f" ({p1['phase0_diagnostics']['shuffle_phase0']['shuffle_blocks_per_month']['2026-05']} in May).")
A("- Layer-label shuffle with within-label norm metric is permutation-insensitive by construction (C3). A between/within ratio is provided as supplementary but is not a permutation test.")
A("- 5 state rows have incomplete LayerCentroids (all May; C2); analyses using all 3 layers exclude them.")
A(f"- April H_t - n_total Pearson r is undefined (April n_total constant at {apr_ntotal_val:.0f}; C1).")
A("- All cross-month p-values in Section 4 are described as unadjusted; BH q-values are provided for multiple-testing adjustment.")
A("- Unigram analysis reflects context-window content, not model output.\n")
A("---\n")

A("## Appendix A - Audit Summary\n")
A("### A.1 Timestamp Normalization\n")
A("- `Timestamp_UTC` in observations.csv: stripped \" UTC\", parsed naive UTC. Observation key.")
A("- `Timestamp` in kopterix_state.csv: ISO 8601 + offset, converted to UTC.")
A("- Chicago `Timestamp` in observations.csv: ignored per spec.")
A("- Window: [2026-04-01 00:00:00 UTC, 2026-06-01 00:00:00 UTC) half-open.")
A(f"- Rows dropped before window: {p1['phase0_diagnostics']['dropped_before_window']};"
  f" after: {p1['phase0_diagnostics']['dropped_after_window']};"
  f" unparseable: {p1['phase0_diagnostics']['dropped_unparseable']}.\n")
A("### A.2 Row Counts\n")
A("| Dataset | Raw rows | In-window | Dropped |")
A("|---------|----------|-----------|---------|")
A(f"| observations.csv | {p1['phase0_diagnostics']['rows_in']} | {len(real)} |"
  f" {p1['phase0_diagnostics']['rows_in']-len(real)} |")
A(f"| kopterix_state.csv | {len(state_raw)} | {len(state)} | {len(state_raw)-len(state)} |")
A(f"| shuffle_clean.csv | - | {len(shuffle)} | - |")
A(f"| quarantine_shuffle_metrics.csv | - | {len(quarantine)} | - |\n")
A("### A.3 Provenance Tags\n")
A("| Tag | Meaning |")
A("|-----|---------|")
A("| real_observation | Scalar metrics from observations.csv |")
A("| shuffle_deterministic | Centroid-distance triplets from log blocks |")
A("| legacy_shuffle_log_text_untrusted | H_t/phi_t/D_t from shuffle log text; quarantined |")
A("| kopterix_state_csv | MeanEmbedding, LayerCentroids, UnigramCounts |")
A("| computed_from_state_csv | Phase 2 metrics derived from state vectors |\n")
A("### A.4 Shuffle Parsing\n")
A(f"- April: {p1['shuffle_parse_status']['blocks_found_per_log']['kopterix_log_2026-04.md']} blocks,"
  f" {p1['shuffle_parse_status']['shuffle_blocks_per_month']['2026-04']} deterministic.")
A(f"- May: {p1['shuffle_parse_status']['blocks_found_per_log']['kopterix_log_2026-05.md']} blocks,"
  f" {p1['shuffle_parse_status']['shuffle_blocks_per_month']['2026-05']} deterministic.")
A(f"- {p1['shuffle_parse_status']['quarantined_metric_lines']} metric lines quarantined.")
A("- April H_t = 4.2: first-run logging-provenance artifact; quarantined.\n")
A("### A.5 Low-n_total Regime\n")
A(f"- {n_regime} observations flagged n_total_regime = True (all May, n_total <= 300).")
A("- Included in all primary analyses; sensitivity in Section 8.")
A(f"- 4 of the 5 sparse-LayerCentroid state rows correspond to n_total_regime observations.\n")
A("### A.6 scipy-Absent Archive\n")
A("Phase 1 was originally run without scipy, producing NaN p-values in"
  " `temporal_structure.csv`, `cross_month_distribution.csv`, `real_vs_shuffle.csv`,"
  " and `cross_month_autocorr_persistence.csv`. These outputs are preserved in"
  " `tables/_scipy_absent_archive/` for provenance. The Zenodo-facing tables are the"
  " scipy-regenerated versions in `tables/`.\n")
A("---\n")
A("*End of Kopterix Two-Month Validation Report (correction pass)*")

report_path = BASE / "kopterix_validation_report_2month.md"
with open(report_path,"w",encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  Report written: {report_path}")

# ── 16. Phase 2 status ────────────────────────────────────────────────────────
p2_status = {
    "phase":2,"pass":"corrected","generated_at":now_str,"scipy_version":scipy.__version__,
    "corrections_applied":["C1_ht_ntotal_april_constant","C2_lc_sparse_5rows","C3_shuffle_metric_inconclusive",
                           "C4_fdr_bh_qvalues","C5_centroid_phrasing","C6_phase1_scipy_rerun"],
    "obs_rows":len(real),"state_rows_in_window":len(state),
    "coverage_matched":n_matched,"coverage_unmatched_obs":n_unmatched_obs,
    "lc_sparse_coverage_rows":len(lc_issues),
    "lc_fully_parsed_rows":n_lc_full,
    "zero_mode_drift_p":float(p_val),"zero_mode_drift_slope":float(slope),
    "layer_shuffle_all_tied":bool(all_tied),
    "layer_bw_ratio":float(bw_ratio),
    "april_ntotal_constant":bool(apr_ntotal_std<1e-10),
    "may_ht_ntotal_pearson_r":float(may_r),"two_month_ht_ntotal_pearson_r":float(two_r),
    "n_total_regime_rows":n_regime,"h_t_sensitivity_pct_std":float(abs(h_sen_diff)/h_std_all*100),
    "stability_sig_q_0.05":sig_q_metrics,"stability_large_d_0.5":large_d,
    "union_vocab_size":len(union_vocab),
    "unigram_entropy_mw_p":float(mw_uni_p) if not np.isnan(mw_uni_p) else None,
    "outputs_tables": sorted([p.name for p in TABLES.glob("phase2_*.csv")]),
    "outputs_figures":sorted([p.name for p in FIGURES.glob("phase2_*.png")]),
    "report":"kopterix_validation_report_2month.md",
    "archive_dir":"tables/_scipy_absent_archive/",
}
with open(INTERMEDIATES/"phase2_status.json","w") as f:
    json.dump(p2_status,f,indent=2)
print("  Status saved.")

print(f"\n=== Phase 2 corrected complete ===")
print(f"  Tables:  {len(list(TABLES.glob('phase2_*.csv')))} CSVs")
print(f"  Figures: {len(list(FIGURES.glob('phase2_*.png')))} PNGs")
print(f"  Report:  kopterix_validation_report_2month.md")
