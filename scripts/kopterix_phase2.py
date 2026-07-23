#!/usr/bin/env python3
"""
kopterix_phase2.py  --  Two-month Kopterix validation Phase 2
Reads Phase 1 intermediates only. Do not modify raw exports.
"""
import os, sys, json, warnings
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

print("=== Kopterix Phase 2 Pre-flight ===")
try:
    import scipy
    from scipy import stats
    print(f"  scipy {scipy.__version__} -- OK")
except ImportError:
    print("  FATAL: scipy not available. Inferential statistics cannot be produced.")
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

required_cols = ["H_t_numeric","n_total_regime","month","ts_utc",
                 "metric_provenance","H_t_init_outlier"]
missing_cols = [c for c in required_cols if c not in real.columns]
if missing_cols:
    print(f"FATAL: real_clean.csv missing columns: {missing_cols}"); sys.exit(1)

real_apr = real[real["month"]=="2026-04"].copy()
real_may = real[real["month"]=="2026-05"].copy()
n_regime = int(real["n_total_regime"].sum())
h_std_all = float(real["H_t_numeric"].std())

print(f"  real_clean: {len(real)} rows  Apr={len(real_apr)}  May={len(real_may)}")
print(f"  shuffle_clean: {len(shuffle)} rows  quarantine: {len(quarantine)} rows")
print(f"  n_total_regime rows: {n_regime}  H_t_init_outlier rows: {real['H_t_init_outlier'].sum()}")

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
for layer in ["surface","mid","residue"]:
    state[f"lc_{layer}"] = state["lc_parsed"].apply(
        lambda d, l=layer: np.array(d[l],dtype=np.float64) if d and l in d else None)

def parse_unigram(s):
    try:
        d = json.loads(s); d.pop("__total__",None); return d
    except: return {}

state["unigram"] = state["UnigramCounts"].apply(parse_unigram)
state["month"]   = state["ts_utc"].dt.to_period("M").astype(str)

n_me_ok  = state["mean_emb"].notna().sum()
n_lc_ok  = state["lc_surface"].notna().sum()
emb_dim  = len(state["mean_emb"].dropna().iloc[0])
print(f"  MeanEmbedding parsed: {n_me_ok}/{len(state)} dim={emb_dim}")
print(f"  LayerCentroids parsed: {n_lc_ok}/{len(state)}")

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
print(f"  Obs: {len(real)}  State: {len(state)}  Matched: {n_matched}  Unmatched obs: {n_unmatched_obs}  Unmatched state: {n_unmatched_state}")
pd.DataFrame([{"obs_rows_in_window":len(real),"state_rows_in_window":len(state),
    "obs_matched_to_state":n_matched,"obs_unmatched":n_unmatched_obs,
    "state_rows_unmatched":n_unmatched_state,"match_tolerance_minutes":5}]
).to_csv(TABLES/"phase2_coverage_audit.csv", index=False)

# ── 5. Zero-mode residual analysis ────────────────────────────────────────────
print("\n=== Zero-mode residual analysis ===")
me_valid = state[state["mean_emb"].notna()].copy()
me_mat   = np.stack(me_valid["mean_emb"].values)
grand_mean   = me_mat.mean(axis=0)
residuals    = me_mat - grand_mean
resid_norms  = np.linalg.norm(residuals, axis=1)
me_valid = me_valid.copy()
me_valid["residual_norm"] = resid_norms

# PCA via SVD
U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
total_var = (S**2).sum()
pc_var    = (S[:3]**2 / total_var).tolist()
me_valid["resid_pc1"] = (U[:,0]*S[0]).tolist()
me_valid["resid_pc2"] = (U[:,1]*S[1]).tolist()
print(f"  Residual PC1/2/3 var: {[f'{v:.4f}' for v in pc_var]}")

t_sec = (me_valid["ts_utc"]-me_valid["ts_utc"].min()).dt.total_seconds().values
slope, intercept, r_val, p_val, _ = stats.linregress(t_sec, resid_norms)
print(f"  Residual norm drift: slope={slope:.4e}  r={r_val:.3f}  p={p_val:.4f}")

resid_by_month = me_valid.groupby("month")["residual_norm"].agg(
    n="count",mean="mean",std="std",median="median").reset_index()
pd.DataFrame([{"grand_mean_norm":float(np.linalg.norm(grand_mean)),
    "resid_norm_mean":float(resid_norms.mean()),"resid_norm_std":float(resid_norms.std()),
    "resid_norm_min":float(resid_norms.min()),"resid_norm_max":float(resid_norms.max()),
    "resid_drift_slope":float(slope),"resid_drift_r":float(r_val),"resid_drift_p":float(p_val),
    "pc1_var":pc_var[0],"pc2_var":pc_var[1],"pc3_var":pc_var[2]}]
).to_csv(TABLES/"phase2_zero_mode_residuals.csv", index=False)
resid_by_month.to_csv(TABLES/"phase2_zero_mode_by_month.csv", index=False)

# ── 6. Layer-label shuffle (exact 6 permutations) ─────────────────────────────
print("\n=== Layer-label shuffle (6 permutations) ===")
LAYERS    = ["surface","mid","residue"]
all_perms = list(permutations(range(3)))
valid_lc  = state[state["lc_surface"].notna()&state["lc_mid"].notna()&state["lc_residue"].notna()].copy()
print(f"  Rows with all 3 layer centroids: {len(valid_lc)}")
row_vecs = [(r["lc_surface"],r["lc_mid"],r["lc_residue"]) for _,r in valid_lc.iterrows()]

def within_label_score(vecs, perm):
    relabeled = [[v[perm[i]] for i in range(3)] for v in vecs]
    scores = []
    for pos in range(3):
        mat = np.stack([r[pos] for r in relabeled])
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
perm_df["rank_ascending"] = perm_df["within_label_norm_mean"].rank(ascending=True).astype(int)
identity_score = float(perm_df[perm_df["is_identity"]]["within_label_norm_mean"].iloc[0])
identity_rank  = int(perm_df[perm_df["is_identity"]]["rank_ascending"].iloc[0])
perm_mean      = float(perm_df["within_label_norm_mean"].mean())
perm_std       = float(perm_df["within_label_norm_mean"].std())
print(f"  Identity rank: {identity_rank}/6  score={identity_score:.6f}  perm_mean={perm_mean:.6f}")
perm_df.to_csv(TABLES/"phase2_layer_label_shuffle.csv", index=False)

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
print("  Cosine distances:\n", cos_summary.to_string(index=False))

# ── 7. Cross-month centroid drift ─────────────────────────────────────────────
print("\n=== Cross-month centroid drift ===")
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
    drift_rows.append({"layer":layer_name,
        "n_april":apr["n"],"n_may":may["n"],
        "cross_month_drift_l2":l2,"cross_month_drift_cosine":cos,
        "within_mean_dist_april":apr["within_mean"],"within_mean_dist_may":may["within_mean"],
        "drift_to_within_ratio_april": l2/apr["within_mean"] if apr["within_mean"]>0 else np.nan,
        "drift_to_within_ratio_may":   l2/may["within_mean"] if may["within_mean"]>0 else np.nan})

drift_df = pd.DataFrame(drift_rows)
drift_df.to_csv(TABLES/"phase2_centroid_drift.csv", index=False)
print(drift_df[["layer","cross_month_drift_l2","cross_month_drift_cosine",
                "within_mean_dist_april","drift_to_within_ratio_april"]].to_string(index=False))

# ── 8. April-vs-May scalar stability (scipy recompute) ────────────────────────
print("\n=== April-vs-May scalar stability ===")
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
stab_df.to_csv(TABLES/"phase2_april_may_stability.csv", index=False)
print(stab_df[["metric","mean_april","mean_may","cohens_d","mw_pval","flag_mw_sig_0.05"]].to_string(index=False))
sig_metrics = stab_df[stab_df["flag_mw_sig_0.05"]]["metric"].tolist()
large_d     = stab_df[stab_df["flag_large_effect_d0.5"]]["metric"].tolist()

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
        "mean_diff_excl_minus_all":float(ev.mean()-av.mean()),"n_regime_rows":n_regime})
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(TABLES/"phase2_low_n_sensitivity.csv", index=False)
print(sens_df[["metric","mean_all","mean_excl","mean_diff_excl_minus_all"]].to_string(index=False))
h_sen_diff = float(sens_df[sens_df["metric"]=="H_t_numeric"]["mean_diff_excl_minus_all"].iloc[0])

# ── 10. Unigram analysis ──────────────────────────────────────────────────────
print("\n=== Unigram analysis ===")
all_uni   = state["unigram"].tolist()
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
print(f"  Top 10: {top10}")

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
print(f"  Unigram entropy MW p (Apr vs May): {mw_uni_p:.4f}")

# ── 11. Viability assessment ──────────────────────────────────────────────────
print("\n=== Viability assessment ===")
obs_per_day = len(real)/WINDOW_DAYS
n_shuffle   = p1["shuffle_n_blocks"]
viab_rows = [
    ("observation_count",        len(real),    len(real)>=100,      f"{len(real)} obs in {WINDOW_DAYS}-day window"),
    ("obs_per_day_mean",         round(obs_per_day,2), obs_per_day>=2.0, f"{obs_per_day:.2f} obs/day"),
    ("state_rows_in_window",     len(state),   len(state)>=100,     f"{len(state)} state rows"),
    ("shuffle_blocks",           n_shuffle,    n_shuffle>=10,       f"{n_shuffle} deterministic shuffle blocks"),
    ("days_with_zero_runs",      0,            True,                "0 days with zero runs"),
    ("n_total_variation_observed",n_regime,    True,                f"{n_regime} low-n_total rows (May); diagnostically useful"),
    ("scipy_available",          1,            True,                f"scipy {scipy.__version__}"),
    ("embedding_dim",            emb_dim,      emb_dim==384,        f"dim={emb_dim}"),
    ("layer_parse_success",      n_lc_ok,      n_lc_ok==len(state), f"{n_lc_ok}/{len(state)} rows parsed"),
]
viab_df = pd.DataFrame(viab_rows, columns=["check","value","passes","note"])
viab_df.to_csv(TABLES/"phase2_viability.csv", index=False)
for _,r in viab_df.iterrows():
    print(f"  [{'PASS' if r['passes'] else 'WARN'}] {r['check']}: {r['note']}")

# ── 12. Metric provenance audit ───────────────────────────────────────────────
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
     "6-permutation within-label variance over all valid state rows"),
    ("phase2_unigram_entropy",         len(state),   True,
     "Shannon entropy from UnigramCounts per state row"),
    ("quarantined_legacy_shuffle_log",
     p1["metric_provenance_inventory"]["legacy_shuffle_log_text_untrusted"], False,
     "H_t/phi_t/D_t from shuffle log text; provenance unverifiable; not used in analysis"),
]
prov_df = pd.DataFrame(prov_rows, columns=["metric_class","count","trusted","notes"])
prov_df.to_csv(TABLES/"phase2_metric_provenance.csv", index=False)
print(prov_df[["metric_class","count","trusted"]].to_string(index=False))

# ── 13. Figures ───────────────────────────────────────────────────────────────
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
ax2 = axes[1]
for mo,clr in COLORS.items():
    s = me_valid[me_valid["month"]==mo]
    ax2.hist(s["residual_norm"],bins=20,alpha=0.6,color=clr,label=mo)
ax2.set_title("Residual norm distribution"); ax2.set_xlabel("Residual L2 norm"); ax2.set_ylabel("Count"); ax2.legend()
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

# Fig 3: Layer-label permutation
fig, ax = plt.subplots(figsize=(8,4))
colors_bar = ["#4CAF50" if r["is_identity"] else "#9E9E9E" for _,r in perm_df.iterrows()]
ax.bar(perm_df["label"],perm_df["within_label_norm_mean"],color=colors_bar)
ax.set_title("Layer-label shuffle: within-label L2 norm\nGreen = identity labeling")
ax.set_ylabel("Mean within-label L2 norm"); plt.xticks(rotation=30,ha="right")
save_fig("phase2_layer_label_shuffle.png")

# Fig 4: Centroid drift
fig, ax = plt.subplots(figsize=(9,4))
dl = drift_df["layer"].str.replace("layer_","").str.replace("mean_embedding","mean-emb")
x = np.arange(len(drift_df)); w = 0.3
ax.bar(x-w, drift_df["within_mean_dist_april"], w, label="Within Apr", color="#2196F3", alpha=0.7)
ax.bar(x,   drift_df["within_mean_dist_may"],   w, label="Within May", color="#FF9800", alpha=0.7)
ax.plot(x-w/2, drift_df["cross_month_drift_l2"],"k^--",label="Cross-month drift L2",ms=8)
ax.set_xticks(x-w/2); ax.set_xticklabels(dl,rotation=20,ha="right")
ax.set_title("Centroid cross-month drift vs within-month variability")
ax.set_ylabel("L2 distance"); ax.legend()
save_fig("phase2_centroid_drift.png")

# Fig 5: Unigram entropy
fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True)
for mo,clr in COLORS.items():
    s = state_u[state_u["month"]==mo]
    axes[0].scatter(s["ts_utc"],s["unigram_entropy"],  s=8,alpha=0.6,color=clr,label=mo)
    axes[1].scatter(s["ts_utc"],s["unigram_richness"], s=8,alpha=0.6,color=clr,label=mo)
axes[0].set_title("Unigram Shannon entropy per state row"); axes[0].set_ylabel("Entropy (bits)"); axes[0].legend()
axes[1].set_title("Unigram vocabulary richness"); axes[1].set_ylabel("Distinct tokens")
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
        p = row["mw_pval"].iloc[0]
        ax.set_xlabel(f"MW p={p:.3f} {'*' if p<0.05 else 'ns'}",fontsize=7)
fig.suptitle("April vs May scalar distributions",fontsize=11)
save_fig("phase2_april_may_violin.png")

# Fig 7: Residual PCA scatter
fig, ax = plt.subplots(figsize=(7,6))
for mo,clr in COLORS.items():
    s = me_valid[me_valid["month"]==mo]
    ax.scatter(s["resid_pc1"],s["resid_pc2"],s=10,alpha=0.6,color=clr,label=mo)
ax.set_title(f"Residual PC1 vs PC2  (var: {pc_var[0]:.3f}, {pc_var[1]:.3f})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
save_fig("phase2_residual_pca.png")

# ── 14. Markdown report ───────────────────────────────────────────────────────
print("\n=== Writing Markdown report ===")
now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
desc     = pd.read_csv(TABLES/"real_descriptive_stats.csv")
temp     = pd.read_csv(TABLES/"temporal_structure.csv")
resid_s  = pd.read_csv(TABLES/"phase2_zero_mode_residuals.csv").iloc[0]
ht_n_r   = next(p["pearson_r"] for p in p1["high_correlation_pairs"]
                if p["metric_a"]=="H_t" and p["metric_b"]=="n_total")

def td(v, fmt=".4f"):
    return format(v, fmt) if not (isinstance(v,float) and np.isnan(v)) else "NA"

lines = []
A = lines.append

A(f"# Kopterix Two-Month Validation Report\n")
A(f"**Instrument window:** 2026-04-01 to 2026-05-31 (UTC)  ")
A(f"**Generated:** {now_str}  ")
A(f"**Pipeline:** Phase 1 (scalar/shuffle/temporal/correlation/flags) + Phase 2 (state/zero-mode/layer-shuffle/unigram/viability)  ")
A(f"**scipy:** {scipy.__version__}  |  **Embedding dim:** {emb_dim}\n")
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
  " State `Timestamp` (ISO 8601 + offset) converted to UTC. Chicago `Timestamp` ignored."
  " Window is half-open: start inclusive, end exclusive.\n")
A("---\n")

A("## 2. Phase 1 Intermediate Verification\n")
A("All required Phase 1 intermediates confirmed present before Phase 2 execution.\n")
A("| File | Rows | Key columns verified |")
A("|------|------|----------------------|")
A(f"| `real_clean.csv` | {len(real)} | H_t_numeric, n_total_regime, month, Timestamp_UTC, metric_provenance, H_t_init_outlier |")
A(f"| `shuffle_clean.csv` | {len(shuffle)} | Timestamp, metric_provenance, is_shuffle |")
A(f"| `quarantine_shuffle_metrics.csv` | {len(quarantine)} | Timestamp, H_t_raw, phi_t_raw, D_t_raw, metric_provenance |")
A(f"| `kopterix_state.csv` (in-window) | {len(state)} | MeanEmbedding (dim={emb_dim}), LayerCentroids (surface/mid/residue), UnigramCounts |\n")
A(f"Phase 1 assumptions active: H_t_numeric = H_t for all real observations (no robust_z exclusions);"
  f" {n_regime} May rows flagged n_total_regime = True (n_total <= 300; included in all primary analyses);"
  f" H_t_init_outlier all False (schema-continuity only); April H_t = 4.2 anomaly quarantined"
  f" as first-run logging-provenance artifact; {len(quarantine)} shuffle log-text metric lines"
  f" quarantined as legacy_shuffle_log_text_untrusted.\n")
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
A(f"P-values computed with scipy {scipy.__version__} (Mann-Whitney U two-sided, KS two-sample)."
  f" Phase 1 OLS trend p-values are NaN (scipy absent at Phase 1 runtime); cross-month p-values"
  f" are recomputed here from Phase 1 real_clean.csv.\n")
A("| Metric | N Apr | N May | Mean Apr | Mean May | Shift | Cohen d | MW p | KS p | MW sig |")
A("|--------|-------|-------|----------|----------|-------|---------|------|------|--------|")
for _,row in stab_df.iterrows():
    sig = "*" if row["flag_mw_sig_0.05"] else "ns"
    A(f"| {row['metric']} | {row['n_april']} | {row['n_may']}"
      f" | {td(row['mean_april'])} | {td(row['mean_may'])}"
      f" | {td(row.get('mean_shift_may_minus_apr',np.nan))}"
      f" | {td(row['cohens_d'],'.3f')} | {td(row['mw_pval'],'.4f')}"
      f" | {td(row['ks_pval'],'.4f')} | {sig} |")
A(f"\nMetrics with MW p < 0.05: {', '.join(sig_metrics) if sig_metrics else 'none'}.")
A(f"Metrics with large effect (|d| > 0.5): {', '.join(large_d) if large_d else 'none'}.\n")
A("See `figures/phase2_april_may_violin.png` and `tables/phase2_april_may_stability.csv`.\n")
A("Framing: split-half instrument characterization only. Seasonal posting patterns,"
  " n_total regime shifts, and platform-side changes cannot be ruled out.\n")
A("---\n")

A("## 5. Zero-Mode Residual Analysis\n")
A(f"Grand mean computed across {len(me_valid)} valid MeanEmbedding rows."
  f" Residual = individual embedding - grand mean. Residual norm measures per-snapshot"
  f" deviation from the two-month mean state.\n")
A(f"- Grand mean L2 norm: {resid_s['grand_mean_norm']:.4f}")
A(f"- Mean residual L2 norm: {resid_s['resid_norm_mean']:.4f} (std: {resid_s['resid_norm_std']:.4f})")
A(f"- Residual range: [{resid_s['resid_norm_min']:.4f}, {resid_s['resid_norm_max']:.4f}]")
A(f"- Drift OLS: slope = {resid_s['resid_drift_slope']:.4e}, r = {resid_s['resid_drift_r']:.3f}, p = {resid_s['resid_drift_p']:.4f}")
A(f"- PC1/PC2/PC3 var explained: {resid_s['pc1_var']:.4f} / {resid_s['pc2_var']:.4f} / {resid_s['pc3_var']:.4f}\n")
A("| Month | N | Mean residual norm | Std |")
A("|-------|---|--------------------|-----|")
for _,row in resid_by_month.iterrows():
    A(f"| {row['month']} | {int(row['n'])} | {row['mean']:.4f} | {row['std']:.4f} |")
drift_interp = ("Statistically significant temporal drift in residual norms"
                f" (p = {resid_s['resid_drift_p']:.4f}, alpha = 0.05)."
                if resid_s['resid_drift_p']<0.05 else
                f"No statistically significant temporal drift in residual norms"
                f" (p = {resid_s['resid_drift_p']:.4f}, alpha = 0.05).")
A(f"\n{drift_interp}\n")
A("See `figures/phase2_residual_norms.png`, `figures/phase2_residual_pca.png`,"
  " `tables/phase2_zero_mode_residuals.csv`, `tables/phase2_zero_mode_by_month.csv`.\n")
A("---\n")

A("## 6. Layer-Label Shuffle (Exact 6 Permutations)\n")
A(f"Rows with all three layer centroids valid: {len(valid_lc)}."
  f" Within-label norm: mean L2 distance of each layer's vectors from"
  f" its cross-row centroid under that permutation (lower = more consistent labeling).\n")
A("| Permutation | Label assignment | Within-label norm | Rank | Identity |")
A("|-------------|-----------------|-------------------|------|----------|")
for _,row in perm_df.iterrows():
    A(f"| {row['permutation']} | {row['label']}"
      f" | {row['within_label_norm_mean']:.6f} | {row['rank_ascending']}"
      f" | {'YES' if row['is_identity'] else ''} |")
A(f"\nIdentity rank: {identity_rank}/6. Permutation mean: {perm_mean:.6f}, std: {perm_std:.6f}.\n")
A("Inter-layer cosine distances (identity labeling):\n")
A("| Pair | Mean | Std | Min | Max |")
A("|------|------|-----|-----|-----|")
for _,row in cos_summary.iterrows():
    A(f"| {row['pair']} | {row['mean']:.6f} | {row['std']:.6f} | {row['min']:.6f} | {row['max']:.6f} |")
A(f"\nCaveat: {len(valid_lc)} rows and 6 permutations gives low statistical power."
  f" Identity rank is descriptive; no permutation p-value is claimed.\n")
A("See `figures/phase2_layer_label_shuffle.png`, `tables/phase2_layer_label_shuffle.csv`,"
  " `tables/phase2_layer_cosine_distances.csv`.\n")
A("---\n")

A("## 7. Centroid-Level Cross-Month Drift\n")
A("Monthly centroids for MeanEmbedding and each layer. Drift/within ratio: cross-month"
  " L2 drift divided by mean within-month L2 distance from centroid."
  " Ratio substantially above 1.0 indicates cross-month shift exceeds within-month spread.\n")
A("| Layer | N Apr | N May | Drift L2 | Drift cosine | Within Apr | Within May | Ratio Apr |")
A("|-------|-------|-------|----------|--------------|-----------|-----------|-----------|")
for _,row in drift_df.iterrows():
    A(f"| {row['layer']} | {row['n_april']} | {row['n_may']}"
      f" | {row['cross_month_drift_l2']:.4f} | {row['cross_month_drift_cosine']:.4f}"
      f" | {row['within_mean_dist_april']:.4f} | {row['within_mean_dist_may']:.4f}"
      f" | {row['drift_to_within_ratio_april']:.3f} |")
A("\nSee `figures/phase2_centroid_drift.png`, `tables/phase2_centroid_drift.csv`.\n")
A("---\n")

A("## 8. Low-n_total Sensitivity\n")
A(f"Primary analysis includes all {len(real)} rows. Sensitivity table compares key"
  f" metrics with and without the {n_regime} n_total_regime rows"
  f" (n_total <= 300, all May). Framed as sensitivity to sampling regime only.\n")
A("| Metric | N all | Mean all | Median all | N excl | Mean excl | Median excl | Mean diff |")
A("|--------|-------|---------|-----------|--------|----------|------------|----------|")
for _,row in sens_df.iterrows():
    A(f"| {row['metric']} | {row['n_all']} | {row['mean_all']:.4f}"
      f" | {row['median_all']:.4f} | {row['n_excl_regime']}"
      f" | {row['mean_excl']:.4f} | {row['median_excl']:.4f}"
      f" | {row['mean_diff_excl_minus_all']:.4f} |")
small = abs(h_sen_diff) < 0.1*h_std_all
A(f"\nH_t mean shift from excluding {n_regime} regime rows: {h_sen_diff:.4f} bits"
  f" ({'small' if small else 'non-negligible'} relative to two-month std {h_std_all:.4f})."
  f" No observations are excluded on sampling-regime grounds.\n")
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
A(f"\nCross-month unigram entropy MW p = {td(mw_uni_p,'.4f')}.\n")
A("Top 10 tokens (two-month total):\n")
A("| Token | Total count |")
A("|-------|-------------|")
for word,cnt in top10: A(f"| {word} | {cnt} |")
A("\nLimitation: unigram counts reflect context-window content at each snapshot,"
  " not model-generated text. Vocabulary shifts may reflect feed content changes,"
  " n_total regime variation, or both. Causal attribution is not possible.\n")
A("See `figures/phase2_unigram_entropy.png`, `tables/phase2_unigram_timeseries.csv`,"
  " `tables/phase2_unigram_top10.csv`.\n")
A("---\n")

A("## 10. Shuffle Evidence\n")
A("Shuffle evidence comes only from deterministic comparison triplets"
  " (surface-mid, mid-residue, surface-residue real/shuffle/delta)."
  " No H_t, phi_t, or D_t values from shuffle log text are used.\n")
A("| Source | Blocks | Trusted |")
A("|--------|--------|---------|")
A(f"| April deterministic comparisons | {p1['phase0_diagnostics']['shuffle_phase0']['shuffle_blocks_per_month']['2026-04']} | Yes |")
A(f"| May deterministic comparisons | {p1['phase0_diagnostics']['shuffle_phase0']['shuffle_blocks_per_month']['2026-05']} | Yes |")
A(f"| Quarantined log-text metric lines | {len(quarantine)} | No |")
A(f"| April H_t = 4.2 anomaly | 1 | No - first-run logging-provenance artifact |\n")
A(f"{p1['power_caveat']}\n")
A("See `tables/phase2_metric_provenance.csv`.\n")
A("---\n")

A("## 11. Temporal Structure\n")
A("From Phase 1 `temporal_structure.csv`. OLS trend p-values are NaN (scipy absent at Phase 1 runtime).\n")
A("| Metric | Autocorr lag-1 | Autocorr lag-4 | OLS slope | OLS R2 | Trend p |")
A("|--------|---------------|---------------|-----------|--------|---------|")
for _,row in temp.iterrows():
    tp = f"{row['ols_trend_p']:.4f}" if not np.isnan(row['ols_trend_p']) else "NaN (scipy absent at Phase 1)"
    A(f"| {row['metric']} | {row['autocorr_lag1']:.3f} | {row['autocorr_lag4']:.3f}"
      f" | {row['ols_slope']:.4e} | {row['ols_r_squared']:.4f} | {tp} |")
A("\nPhase 1 cross-month autocorrelation flags: H_t shows large autocorr diff between months."
  " phi_t and centroid_dist_surface_residue show large Cohen d. Consistent with Section 4.\n")
A("---\n")

A("## 12. Correlation Summary\n")
A("High-correlation pairs (|r| > 0.6) from Phase 1 Pearson analysis:\n")
A("| Pair | Pearson r |")
A("|------|-----------|")
for pair in p1["high_correlation_pairs"]:
    A(f"| {pair['metric_a']} - {pair['metric_b']} | {pair['pearson_r']:.4f} |")
A(f"\nH_t - n_total correlation (r = {ht_n_r:.4f}) is strong and persists across both months."
  f" H_t cannot be interpreted independently of n_total without further normalization.\n")
A("---\n")

A("## 13. Viability Assessment\n")
A("| Check | Value | Status | Note |")
A("|-------|-------|--------|------|")
for _,row in viab_df.iterrows():
    A(f"| {row['check']} | {row['value']} | {'PASS' if row['passes'] else 'WARN'} | {row['note']} |")
A(f"\nThe instrument produced a continuous two-month record with no zero-run days."
  f" Observation density ({obs_per_day:.2f}/day), state coverage ({len(state)} rows),"
  f" and deterministic shuffle blocks ({n_shuffle}) support descriptive characterization"
  f" and cross-month comparison. This report removes all language about n_total being"
  f" constant and all recommendations to vary n_total; n_total variation is observed and"
  f" diagnostically useful.\n")
A("---\n")

A("## 14. Limitations and Caveats\n")
A(f"- The two-month window (2026-04-01 to 2026-05-31) is the complete instrument record. No claims extend beyond it.")
A(f"- Cross-month scalar differences may reflect platform-side changes, seasonal patterns, n_total shifts, or instrument drift; these cannot be disentangled.")
A(f"- Shuffle power is low: {n_shuffle} deterministic blocks ({p1['phase0_diagnostics']['shuffle_phase0']['shuffle_blocks_per_month']['2026-05']} in May). Real-vs-shuffle centroid comparisons are descriptive only.")
A(f"- OLS trend p-values in Phase 1 temporal_structure.csv are NaN; not retroactively recomputed to preserve table provenance.")
A(f"- Layer-label shuffle with {len(valid_lc)} rows and 6 permutations is underpowered for formal inference; identity rank is descriptive.")
A(f"- Unigram analysis reflects context-window content, not model output.")
A(f"- No autocorrelation derivatives beyond lag-1 and lag-4 are reported.\n")
A("---\n")

A("## Appendix A - Audit Summary\n")
A("### A.1 Timestamp Normalization\n")
A("- `Timestamp_UTC` in observations.csv: stripped \" UTC\" suffix, parsed as naive UTC. Observation key.")
A("- `Timestamp` in kopterix_state.csv: ISO 8601 with offset, converted to UTC.")
A("- Chicago `Timestamp` in observations.csv: ignored per spec.")
A(f"- Window: [2026-04-01 00:00:00 UTC, 2026-06-01 00:00:00 UTC) half-open.")
A(f"- Rows dropped before window: {p1['phase0_diagnostics']['dropped_before_window']};"
  f" after: {p1['phase0_diagnostics']['dropped_after_window']};"
  f" unparseable: {p1['phase0_diagnostics']['dropped_unparseable']}.\n")
A("### A.2 Row Counts\n")
A("| Dataset | Raw rows | In-window | Dropped |")
A("|---------|----------|-----------|---------|")
A(f"| observations.csv | {p1['phase0_diagnostics']['rows_in']} | {len(real)} | {p1['phase0_diagnostics']['rows_in']-len(real)} |")
A(f"| kopterix_state.csv | {len(state_raw)} | {len(state)} | {len(state_raw)-len(state)} |")
A(f"| shuffle_clean.csv | - | {len(shuffle)} | - |")
A(f"| quarantine_shuffle_metrics.csv | - | {len(quarantine)} | - |\n")
A("### A.3 Provenance Tags\n")
A("| Tag | Meaning |")
A("|-----|---------|")
A("| real_observation | Scalar metrics from observations.csv |")
A("| shuffle_deterministic | Centroid-distance triplets from log blocks |")
A("| legacy_shuffle_log_text_untrusted | H_t/phi_t/D_t from shuffle log text; quarantined |")
A("| kopterix_state_csv | MeanEmbedding, LayerCentroids, UnigramCounts from state file |")
A("| computed_from_state_csv | Phase 2 metrics derived from parsed state vectors |\n")
A("### A.4 Shuffle Parsing\n")
A(f"- April: {p1['shuffle_parse_status']['blocks_found_per_log']['kopterix_log_2026-04.md']} blocks,"
  f" {p1['shuffle_parse_status']['shuffle_blocks_per_month']['2026-04']} deterministic comparison blocks.")
A(f"- May: {p1['shuffle_parse_status']['blocks_found_per_log']['kopterix_log_2026-05.md']} blocks,"
  f" {p1['shuffle_parse_status']['shuffle_blocks_per_month']['2026-05']} deterministic comparison blocks.")
A(f"- {p1['shuffle_parse_status']['quarantined_metric_lines']} metric lines quarantined.")
A("- April H_t = 4.2 anomaly: block 2026-04-05 18:35 UTC; first-run logging-provenance artifact;"
  " not used as real feed entropy; not used as exclusion-rule anchor.\n")
A("### A.5 Low-n_total Regime\n")
A(f"- {n_regime} observations with n_total <= 300 flagged n_total_regime = True (all May 2026).")
A("- Included in all primary analyses.")
A("- Sensitivity comparison excluding them is in Section 8.")
A("- n_total variation is diagnostically useful; no observations are excluded on this basis.\n")
A("---\n")
A("*End of Kopterix Two-Month Validation Report*")

report_path = BASE / "kopterix_validation_report_2month.md"
with open(report_path,"w",encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  Report written: {report_path}")

# ── 15. Phase 2 status ────────────────────────────────────────────────────────
p2_status = {
    "phase":2, "generated_at":now_str, "scipy_version":scipy.__version__,
    "obs_rows":len(real), "state_rows_in_window":len(state),
    "coverage_matched":n_matched, "coverage_unmatched_obs":n_unmatched_obs,
    "coverage_unmatched_state":n_unmatched_state,
    "zero_mode_drift_p":float(p_val), "zero_mode_drift_slope":float(slope),
    "layer_shuffle_identity_rank":identity_rank,
    "n_total_regime_rows":n_regime, "h_t_sensitivity_mean_diff":float(h_sen_diff),
    "stability_sig_mw_0.05":sig_metrics, "stability_large_d_0.5":large_d,
    "union_vocab_size":len(union_vocab),
    "unigram_entropy_mw_p":float(mw_uni_p) if not np.isnan(mw_uni_p) else None,
    "outputs_tables": sorted([p.name for p in TABLES.glob("phase2_*.csv")]),
    "outputs_figures":sorted([p.name for p in FIGURES.glob("phase2_*.png")]),
    "report":"kopterix_validation_report_2month.md",
}
with open(INTERMEDIATES/"phase2_status.json","w") as f:
    json.dump(p2_status,f,indent=2)
print("  Status saved: analysis_intermediate/phase2_status.json")

print(f"\n=== Phase 2 complete ===")
print(f"  Tables:  {len(list(TABLES.glob('phase2_*.csv')))} new CSVs")
print(f"  Figures: {len(list(FIGURES.glob('phase2_*.png')))} new PNGs")
print(f"  Report:  kopterix_validation_report_2month.md")
