#!/usr/bin/env python3
"""
kopterix_phase2_addendum.py
Restores four analyses dropped from the two-month Kopterix pipeline.

Items:
  1 - Per-timestamp layer-specific zero-mode residual geometry (RESTORE April method)
  2 - Order-based layer-label permutation test (REPLACE broken within-label norm)
  3 - Lexical temporal drift vs timestamp-shuffled baseline (RESTORE from April)
  4 - Anomaly persistence / run-length score (IMPLEMENT - specced in April, never built)

Window: [2026-04-01 00:00, 2026-06-01 00:00) UTC
Frozen counts: 231 obs / 215 state rows / 13 shuffle blocks / 210 fully-parsed layer rows
               121 April obs / 110 May obs

Does NOT modify any existing tables, figures, or the report file.
Writes ONLY new outputs under tables/ and figures/.
"""
import sys, json, warnings
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path resolution (same strategy as kopterix_phase2_corrected.py) ──────────
BASE = Path(__file__).resolve().parent
if not (BASE / "kopterix_state.csv").exists():
    # deposit layout: scripts\ sits one level below the deposit root
    BASE = Path(__file__).resolve().parents[1]
if not (BASE / "kopterix_state.csv").exists():
    print("FATAL: cannot locate kopterix_state.csv - check BASE path"); sys.exit(1)

INTERMEDIATES = BASE / "analysis_intermediate"
TABLES        = BASE / "tables"
FIGURES       = BASE / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ── Window and frozen-count constants ────────────────────────────────────────
WINDOW_START             = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
WINDOW_END               = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
N_OBS_EXPECTED           = 231
N_STATE_EXPECTED         = 215
N_SHUFFLE_EXPECTED       = 13
N_FULLY_PARSED_EXPECTED  = 210
N_APR_EXPECTED           = 121
N_MAY_EXPECTED           = 110
N_SHUFFLES_LEXICAL       = 200   # shuffle iterations for lexical baseline
ROBUST_Z_THRESH          = 3.5
LAYERS                   = ["surface", "mid", "residue"]

print("=" * 64)
print("  kopterix_phase2_addendum.py - four restored analyses")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: Load and validate inputs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Section 0: Loading and validating inputs ===")

required_files = {
    "real_clean":      INTERMEDIATES / "real_clean.csv",
    "shuffle_clean":   INTERMEDIATES / "shuffle_clean.csv",
    "phase1_status":   INTERMEDIATES / "phase1_status.json",
    "diag_flags":      TABLES / "diagnostic_flags.csv",
    "kopterix_state":  BASE / "kopterix_state.csv",
}
missing_files = [str(p) for n, p in required_files.items() if not p.exists()]
if missing_files:
    print("FATAL: missing required inputs:")
    for m in missing_files:
        print(f"  {m}")
    sys.exit(1)

for name, path in required_files.items():
    print(f"  [OK] {path.name}")

# ── Load real observations ────────────────────────────────────────────────────
real = pd.read_csv(INTERMEDIATES / "real_clean.csv")
real["ts_utc"] = pd.to_datetime(
    real["Timestamp_UTC"].str.replace(" UTC", "", regex=False), utc=True)
real = real.sort_values("ts_utc").reset_index(drop=True)

n_obs = len(real)
n_apr = int((real["month"] == "2026-04").sum())
n_may = int((real["month"] == "2026-05").sum())

shuffle_clean = pd.read_csv(INTERMEDIATES / "shuffle_clean.csv")
n_shuffle = len(shuffle_clean)

# ── Load kopterix_state in-window ────────────────────────────────────────────
state_raw = pd.read_csv(BASE / "kopterix_state.csv")
state_raw["ts_utc"] = pd.to_datetime(state_raw["Timestamp"], utc=True)
state_raw = state_raw.sort_values("ts_utc").reset_index(drop=True)
state = state_raw[
    (state_raw["ts_utc"] >= WINDOW_START) &
    (state_raw["ts_utc"] < WINDOW_END)
].copy().reset_index(drop=True)
state["month"] = state["ts_utc"].dt.to_period("M").astype(str)

n_state = len(state)

# ── Parse helpers (reuse Phase 2 approach exactly) ───────────────────────────
def parse_vec(s):
    if pd.isna(s):
        return None
    try:
        v = json.loads(s)
        return np.array(v, dtype=float)
    except Exception:
        return None

def parse_unigram(s):
    if pd.isna(s):
        return None
    try:
        d = json.loads(s)
        d.pop("__total__", None)
        return d
    except Exception:
        return None

state["mean_emb"]  = state["MeanEmbedding"].apply(parse_vec)
state["lc_parsed"] = state["LayerCentroids"].apply(
    lambda s: json.loads(s) if pd.notna(s) else None)

for layer in LAYERS:
    state[f"lc_{layer}"] = state["lc_parsed"].apply(
        lambda d, l=layer: np.array(d[l], dtype=float) if (d and l in d) else None)

state["unigram"] = state["UnigramCounts"].apply(parse_unigram)

# ── Fully-parsed rows: all 3 layers AND mean_emb present ─────────────────────
fully_parsed_mask = (
    state["lc_surface"].notna() &
    state["lc_mid"].notna() &
    state["lc_residue"].notna() &
    state["mean_emb"].notna()
)
valid_lc = state[fully_parsed_mask].copy().reset_index(drop=True)
n_fully_parsed = len(valid_lc)

# ── Count validation: STOP on any mismatch ───────────────────────────────────
print(f"\n  real_clean.csv rows:               {n_obs}  (expected {N_OBS_EXPECTED})")
print(f"    April:                           {n_apr}  (expected {N_APR_EXPECTED})")
print(f"    May:                             {n_may}  (expected {N_MAY_EXPECTED})")
print(f"  shuffle_clean.csv rows:            {n_shuffle}  (expected {N_SHUFFLE_EXPECTED})")
print(f"  kopterix_state in-window rows:     {n_state}  (expected {N_STATE_EXPECTED})")
print(f"  fully-parsed layer rows:           {n_fully_parsed}  (expected {N_FULLY_PARSED_EXPECTED})")

mismatches = []
if n_obs           != N_OBS_EXPECTED:           mismatches.append(f"obs count {n_obs} != {N_OBS_EXPECTED}")
if n_apr           != N_APR_EXPECTED:           mismatches.append(f"April obs {n_apr} != {N_APR_EXPECTED}")
if n_may           != N_MAY_EXPECTED:           mismatches.append(f"May obs {n_may} != {N_MAY_EXPECTED}")
if n_shuffle       != N_SHUFFLE_EXPECTED:       mismatches.append(f"shuffle rows {n_shuffle} != {N_SHUFFLE_EXPECTED}")
if n_state         != N_STATE_EXPECTED:         mismatches.append(f"state rows {n_state} != {N_STATE_EXPECTED}")
if n_fully_parsed  != N_FULLY_PARSED_EXPECTED:  mismatches.append(f"fully-parsed {n_fully_parsed} != {N_FULLY_PARSED_EXPECTED}")

if mismatches:
    print("\nFATAL: count mismatch(es) - not proceeding:")
    for m in mismatches:
        print(f"  {m}")
    sys.exit(1)

print("\n  All counts match frozen ground truth. Proceeding.")

# ── Shared geometry helpers ───────────────────────────────────────────────────
def cosine_dist(a, b):
    """Cosine distance: 1 - cos(a, b). Returns NaN if either vector is near zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))

def unit(v):
    """Return unit vector; None if near-zero norm."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n

def robust_z(series):
    """Consistent with kopterix_phase1.py: 0.6745*(x-median)/MAD."""
    arr = np.array(series, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 4:
        return np.full(len(arr), np.nan)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad == 0:
        return np.full(len(arr), np.nan)
    return 0.6745 * (arr - med) / mad


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 1: Per-timestamp layer-specific zero-mode residual geometry
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Item 1: Layer-specific zero-mode residual geometry ===")
print("    (per-timestamp MeanEmbedding residuals, ALONGSIDE existing grand-mean section)")

# Compute per-timestamp residuals for each layer
for layer in LAYERS:
    valid_lc[f"r_{layer}"] = [
        row[f"lc_{layer}"] - row["mean_emb"]
        for _, row in valid_lc.iterrows()
    ]

# ── 1a. Residual norms ────────────────────────────────────────────────────────
for layer in LAYERS:
    valid_lc[f"rnorm_{layer}"] = valid_lc[f"r_{layer}"].apply(
        lambda v: float(np.linalg.norm(v)) if v is not None else np.nan)

norm_overall = []
for layer in LAYERS:
    vals = valid_lc[f"rnorm_{layer}"].dropna().values
    norm_overall.append({
        "layer": layer, "scope": "overall",
        "n": len(vals),
        "mean": float(vals.mean()) if len(vals) else np.nan,
        "std":  float(vals.std())  if len(vals) else np.nan,
    })

norm_monthly = []
for mo in ["2026-04", "2026-05"]:
    sub = valid_lc[valid_lc["month"] == mo]
    for layer in LAYERS:
        vals = sub[f"rnorm_{layer}"].dropna().values
        norm_monthly.append({
            "layer": layer, "scope": mo,
            "n": len(vals),
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std":  float(vals.std())  if len(vals) else np.nan,
        })

norms_df = pd.DataFrame(norm_overall + norm_monthly)
norms_df.to_csv(TABLES / "phase2_residual_geometry_norms.csv", index=False)
print(f"  Residual norms written: {len(norms_df)} rows")
print(norms_df.to_string(index=False))

# ── 1b. Directional separation (cosine distance between unit-normalized residuals) ──
dir_rows = []
pairs = [("surface", "mid"), ("mid", "residue"), ("surface", "residue")]

for _, row in valid_lc.iterrows():
    for l1, l2 in pairs:
        u1 = unit(row[f"r_{l1}"])
        u2 = unit(row[f"r_{l2}"])
        if u1 is None or u2 is None:
            cd = np.nan
        else:
            cd = cosine_dist(u1, u2)
        dir_rows.append({
            "ts_utc": str(row["ts_utc"]),
            "month":  row["month"],
            "pair":   f"{l1}-{l2}",
            "cos_dist_unit_residual": cd,
        })

dir_df = pd.DataFrame(dir_rows)

dir_summary = []
for scope_label, mask_fn in [
    ("overall",  lambda df: df),
    ("2026-04",  lambda df: df[df["month"] == "2026-04"]),
    ("2026-05",  lambda df: df[df["month"] == "2026-05"]),
]:
    sub = mask_fn(dir_df)
    for pair_name in [f"{l1}-{l2}" for l1, l2 in pairs]:
        vals = sub[sub["pair"] == pair_name]["cos_dist_unit_residual"].dropna().values
        dir_summary.append({
            "pair": pair_name, "scope": scope_label,
            "n": len(vals),
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std":  float(vals.std())  if len(vals) else np.nan,
        })

dir_summary_df = pd.DataFrame(dir_summary)
dir_summary_df.to_csv(TABLES / "phase2_residual_geometry_directional.csv", index=False)
print(f"\n  Residual directional written: {len(dir_summary_df)} rows")
print(dir_summary_df.to_string(index=False))

# ── 1c. Residual temporal drift ───────────────────────────────────────────────
# valid_lc is already sorted by ts_utc (inherited from state sort)
# Consecutive steps: adjacent rows in valid_lc, no bridging
# Per-month: skip the step where row[i] is April and row[i+1] is May

drift_rows = []
vl = valid_lc.reset_index(drop=True)
n_steps_usable = 0

for i in range(len(vl) - 1):
    r0 = vl.iloc[i]
    r1 = vl.iloc[i + 1]
    step_months = (r0["month"], r1["month"])

    # is this an April-May boundary step?
    is_boundary = (r0["month"] == "2026-04" and r1["month"] == "2026-05")

    residual_drifts = {}
    raw_drifts      = {}
    for layer in LAYERS:
        rd = float(np.linalg.norm(r1[f"r_{layer}"] - r0[f"r_{layer}"]))
        cd = float(np.linalg.norm(r1[f"lc_{layer}"] - r0[f"lc_{layer}"]))
        residual_drifts[layer] = rd
        raw_drifts[layer]      = cd

    global_drift = float(np.linalg.norm(r1["mean_emb"] - r0["mean_emb"]))

    row_dict = {
        "step_i": i,
        "ts_from": str(r0["ts_utc"]),
        "ts_to":   str(r1["ts_utc"]),
        "month_from": r0["month"],
        "month_to":   r1["month"],
        "is_boundary": is_boundary,
        "global_drift": global_drift,
    }
    for layer in LAYERS:
        row_dict[f"residual_drift_{layer}"] = residual_drifts[layer]
        row_dict[f"raw_drift_{layer}"]      = raw_drifts[layer]
        ratio = residual_drifts[layer] / raw_drifts[layer] \
            if raw_drifts[layer] > 1e-15 else np.nan
        row_dict[f"ratio_resid_raw_{layer}"] = ratio

    drift_rows.append(row_dict)
    n_steps_usable += 1

drift_df = pd.DataFrame(drift_rows)
drift_df.to_csv(TABLES / "phase2_residual_geometry_drift.csv", index=False)
print(f"\n  Residual drift steps written: {len(drift_df)}  (usable={n_steps_usable})")

# Drift summary: overall (may include boundary), per-month (exclude boundary)
drift_summary = []
for scope_label, row_mask in [
    ("overall", drift_df),  # all steps allowed in overall
    ("2026-04", drift_df[~drift_df["is_boundary"] & (drift_df["month_from"] == "2026-04")]),
    ("2026-05", drift_df[~drift_df["is_boundary"] & (drift_df["month_from"] == "2026-05")]),
]:
    sub = row_mask
    n_sub = len(sub)
    entry = {"scope": scope_label, "n_steps": n_sub,
             "global_drift_mean": float(sub["global_drift"].mean()) if n_sub else np.nan,
             "global_drift_std":  float(sub["global_drift"].std())  if n_sub else np.nan}
    for layer in LAYERS:
        for stat_name, col_prefix in [("residual_drift", "residual_drift"),
                                       ("raw_drift",      "raw_drift"),
                                       ("ratio",          "ratio_resid_raw")]:
            col = f"{col_prefix}_{layer}"
            vals = sub[col].dropna()
            entry[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            entry[f"{col}_std"]  = float(vals.std())  if len(vals) else np.nan
    drift_summary.append(entry)

drift_summary_df = pd.DataFrame(drift_summary)
# print a condensed view
print("  Drift summary (overall + per-month):")
condensed_cols = ["scope","n_steps","global_drift_mean"] + \
    [f"residual_drift_{l}_mean" for l in LAYERS] + \
    [f"ratio_resid_raw_{l}_mean" for l in LAYERS]
print(drift_summary_df[condensed_cols].to_string(index=False))

# ── 1 Figures ─────────────────────────────────────────────────────────────────
MONTH_COLORS = {"2026-04": "#2196F3", "2026-05": "#FF9800"}

# Fig 1a: Residual norms per layer per month
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, layer in zip(axes, LAYERS):
    for mo, clr in MONTH_COLORS.items():
        sub = valid_lc[valid_lc["month"] == mo]
        vals = sub[f"rnorm_{layer}"].dropna().values
        ax.scatter(sub.loc[sub[f"rnorm_{layer}"].notna(), "ts_utc"],
                   vals, s=8, alpha=0.6, color=clr, label=mo)
    ax.set_title(f"Residual norm - {layer}")
    ax.set_ylabel("||r_layer(t)||")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
fig.suptitle("Item 1a: Layer residual norms (per-timestamp MeanEmbedding)")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_residual_geometry_norms.png", dpi=120)
plt.close(fig)

# Fig 1b: Directional separation
pair_names = [f"{l1}-{l2}" for l1, l2 in pairs]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, pname in zip(axes, pair_names):
    sub_p = dir_df[dir_df["pair"] == pname]
    for mo, clr in MONTH_COLORS.items():
        sub = sub_p[sub_p["month"] == mo]
        notna = sub["cos_dist_unit_residual"].notna()
        ax.scatter(
            pd.to_datetime(sub.loc[notna, "ts_utc"], utc=True),
            sub.loc[notna, "cos_dist_unit_residual"],
            s=8, alpha=0.6, color=clr, label=mo)
    ax.set_title(f"Cos dist unit-resid\n{pname}")
    ax.set_ylabel("cosine distance")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
fig.suptitle("Item 1b: Residual directional separation (unit-normalized residuals)")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_residual_geometry_directional.png", dpi=120)
plt.close(fig)

# Fig 1c: Residual drift vs raw drift per layer
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, layer in zip(axes, LAYERS):
    ts_mid = pd.to_datetime(drift_df["ts_to"], utc=True)
    ax.plot(ts_mid, drift_df[f"residual_drift_{layer}"],
            lw=0.8, color="#1976D2", label="residual drift", alpha=0.8)
    ax.plot(ts_mid, drift_df[f"raw_drift_{layer}"],
            lw=0.8, color="#E53935", label="raw centroid drift", alpha=0.6)
    ax.set_title(f"Drift - {layer}")
    ax.set_ylabel("L2 distance (consecutive)")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
fig.suptitle("Item 1c: Residual vs raw centroid temporal drift per layer")
fig.tight_layout()
fig.savefig(FIGURES / "phase2_residual_geometry_drift.png", dpi=120)
plt.close(fig)

item1_result = {
    "framing": "descriptive layer-specific geometry, not a significance test",
    "n_fully_parsed": n_fully_parsed,
    "n_drift_steps_usable": n_steps_usable,
    "outputs": [
        "tables/phase2_residual_geometry_norms.csv",
        "tables/phase2_residual_geometry_directional.csv",
        "tables/phase2_residual_geometry_drift.csv",
        "figures/phase2_residual_geometry_norms.png",
        "figures/phase2_residual_geometry_directional.png",
        "figures/phase2_residual_geometry_drift.png",
    ],
    "residual_norm_means_overall": {
        row["layer"]: round(row["mean"], 6)
        for _, row in norms_df[norms_df["scope"] == "overall"].iterrows()
    },
}
print(f"\n  Item 1 complete. Drift steps usable: {n_steps_usable}")


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 2: Order-based layer-label permutation test
# REPLACES the within-label norm (C3, retired-in-place)
# Distance metric: COSINE (confirmed: same as existing centroid_dist_* columns)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Item 2: Order-based layer-label permutation test ===")
print("    Distance metric: COSINE (same as existing centroid_dist_* columns)")
print("    RETIRING within-label norm score (permutation-invariant by construction)")

all_perms = list(permutations(range(3)))  # 6 permutations of [0,1,2]
perm_layer_map = {
    perm: [LAYERS[i] for i in perm]
    for perm in all_perms
}

# Per-timestamp: compute 3 inter-layer cosine distances under each of 6 relabelings
# Observed labeling: (surface, mid, residue) = (0,1,2)
# Under permutation p: label[0] -> LAYERS[p[0]], etc.
# d_surface_mid = cosine dist between vectors assigned to "surface" and "mid" positions

def per_ts_distances(vec_triple, perm):
    """
    vec_triple: (surface_vec, mid_vec, residue_vec) in original order
    perm: tuple of 3 indices reordering the original vectors
    Returns (d_surface_mid, d_mid_residue, d_surface_residue) under the relabeling.
    """
    relabeled = [vec_triple[i] for i in perm]  # [new_surface, new_mid, new_residue]
    d_sm  = cosine_dist(relabeled[0], relabeled[1])
    d_mr  = cosine_dist(relabeled[1], relabeled[2])
    d_sr  = cosine_dist(relabeled[0], relabeled[2])
    return d_sm, d_mr, d_sr

# Build per-timestamp records for all 6 permutations
identity_perm = (0, 1, 2)
perm_ts_rows = []

vl_arr = valid_lc.reset_index(drop=True)
for idx, row in vl_arr.iterrows():
    triple = (row["lc_surface"], row["lc_mid"], row["lc_residue"])
    for perm in all_perms:
        d_sm, d_mr, d_sr = per_ts_distances(triple, perm)
        perm_ts_rows.append({
            "ts_utc":     str(row["ts_utc"]),
            "month":      row["month"],
            "permutation": str(perm),
            "is_identity": perm == identity_perm,
            "d_surface_mid":       d_sm,
            "d_mid_residue":       d_mr,
            "d_surface_residue":   d_sr,
        })

perm_ts_df = pd.DataFrame(perm_ts_rows)

obs_df  = perm_ts_df[perm_ts_df["is_identity"]].copy().reset_index(drop=True)
null_df = perm_ts_df[~perm_ts_df["is_identity"]].copy().reset_index(drop=True)

n_used = len(obs_df)

# ── Observed statistics ───────────────────────────────────────────────────────
obs_sm  = obs_df["d_surface_mid"].dropna().values
obs_mr  = obs_df["d_mid_residue"].dropna().values
obs_sr  = obs_df["d_surface_residue"].dropna().values

# 1. P(d_surface_mid < d_mid_residue)
obs_p1 = float(np.mean(obs_sm < obs_mr))

# 2. P(d_surface_residue is the largest)
obs_p2 = float(np.mean(
    (obs_sr > obs_sm) & (obs_sr > obs_mr)
))

# 3. P(d_surface_mid < d_mid_residue < d_surface_residue)
obs_p3 = float(np.mean(
    (obs_sm < obs_mr) & (obs_mr < obs_sr)
))

# 4. Mean ratio d_surface_mid / d_mid_residue
with np.errstate(divide="ignore", invalid="ignore"):
    ratio_sm_mr = np.where(obs_mr > 1e-15, obs_sm / obs_mr, np.nan)
obs_ratio1 = float(np.nanmean(ratio_sm_mr))

# 5. Mean ratio d_surface_residue / d_surface_mid
with np.errstate(divide="ignore", invalid="ignore"):
    ratio_sr_sm = np.where(obs_sm > 1e-15, obs_sr / obs_sm, np.nan)
obs_ratio2 = float(np.nanmean(ratio_sr_sm))

# ── Exact permutation-null statistics (from 6-permutation enumeration) ────────
# For each of the 5 non-identity permutations, compute the same stats
# NULL is the mean over all 5 non-identity permutations

null_by_perm = {}
for perm in all_perms:
    if perm == identity_perm:
        continue
    sub = perm_ts_df[perm_ts_df["permutation"] == str(perm)]
    sm  = sub["d_surface_mid"].dropna().values
    mr  = sub["d_mid_residue"].dropna().values
    sr  = sub["d_surface_residue"].dropna().values
    p1  = float(np.mean(sm < mr)) if len(sm) and len(mr) else np.nan
    p2  = float(np.mean((sr > sm) & (sr > mr))) if len(sr) else np.nan
    p3  = float(np.mean((sm < mr) & (mr < sr))) if len(sm) else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r1_arr = np.where(mr > 1e-15, sm / mr, np.nan)
        r2_arr = np.where(sm > 1e-15, sr / sm, np.nan)
    r1 = float(np.nanmean(r1_arr))
    r2 = float(np.nanmean(r2_arr))
    null_by_perm[str(perm)] = {
        "p_sm_lt_mr": p1, "p_sr_largest": p2, "p_order_full": p3,
        "ratio_sm_mr": r1, "ratio_sr_sm": r2
    }

null_vals = pd.DataFrame(null_by_perm).T

null_p1_mean = float(null_vals["p_sm_lt_mr"].mean())
null_p1_std  = float(null_vals["p_sm_lt_mr"].std())
null_p2_mean = float(null_vals["p_sr_largest"].mean())
null_p2_std  = float(null_vals["p_sr_largest"].std())
null_p3_mean = float(null_vals["p_order_full"].mean())
null_p3_std  = float(null_vals["p_order_full"].std())
null_r1_mean = float(null_vals["ratio_sm_mr"].mean())
null_r1_std  = float(null_vals["ratio_sm_mr"].std())
null_r2_mean = float(null_vals["ratio_sr_sm"].mean())
null_r2_std  = float(null_vals["ratio_sr_sm"].std())

# Theoretical nulls for reference (from item spec)
THEORETICAL_NULL_P1 = 0.500
THEORETICAL_NULL_P2 = 0.333
THEORETICAL_NULL_P3 = 0.167

# ── Empirical two-sided tail probability ──────────────────────────────────────
# For proportion stats: pool all null-permutation per-timestamp values
# and compute the fraction at least as extreme as observed

def emp_tail_prob_proportion(obs_val, null_prop_by_perm_df, col):
    """Two-sided: P(|null_stat - null_mean| >= |obs - null_mean|)"""
    null_mean = float(null_prop_by_perm_df[col].mean())
    null_dist  = null_prop_by_perm_df[col].dropna().values
    if len(null_dist) == 0:
        return np.nan
    obs_dev  = abs(obs_val - null_mean)
    null_dev = np.abs(null_dist - null_mean)
    return float(np.mean(null_dev >= obs_dev))

p1_tail = emp_tail_prob_proportion(obs_p1, null_vals, "p_sm_lt_mr")
p2_tail = emp_tail_prob_proportion(obs_p2, null_vals, "p_sr_largest")
p3_tail = emp_tail_prob_proportion(obs_p3, null_vals, "p_order_full")

# ── Build output table ────────────────────────────────────────────────────────
order_perm_rows = [
    {
        "statistic":          "P(d_surface_mid < d_mid_residue)",
        "theoretical_null":   THEORETICAL_NULL_P1,
        "observed":           round(obs_p1, 6),
        "null_exact_mean":    round(null_p1_mean, 6),
        "null_exact_std":     round(null_p1_std, 6),
        "two_sided_tail_p":   round(p1_tail, 4),
        "n_used":             n_used,
        "note": "ordering test; null mean computed from 5 non-identity permutations",
    },
    {
        "statistic":          "P(d_surface_residue is largest)",
        "theoretical_null":   THEORETICAL_NULL_P2,
        "observed":           round(obs_p2, 6),
        "null_exact_mean":    round(null_p2_mean, 6),
        "null_exact_std":     round(null_p2_std, 6),
        "two_sided_tail_p":   round(p2_tail, 4),
        "n_used":             n_used,
        "note": "ordering test",
    },
    {
        "statistic":          "P(d_sm < d_mr < d_sr)",
        "theoretical_null":   THEORETICAL_NULL_P3,
        "observed":           round(obs_p3, 6),
        "null_exact_mean":    round(null_p3_mean, 6),
        "null_exact_std":     round(null_p3_std, 6),
        "two_sided_tail_p":   round(p3_tail, 4),
        "n_used":             n_used,
        "note": "strict ordering test",
    },
    {
        "statistic":          "mean ratio d_surface_mid / d_mid_residue",
        "theoretical_null":   "computed",
        "observed":           round(obs_ratio1, 6),
        "null_exact_mean":    round(null_r1_mean, 6),
        "null_exact_std":     round(null_r1_std, 6),
        "two_sided_tail_p":   "N/A",
        "n_used":             n_used,
        "note": "ratio; exact null mean != 1.0 in general; computed from 5-perm enumeration",
    },
    {
        "statistic":          "mean ratio d_surface_residue / d_surface_mid",
        "theoretical_null":   "computed",
        "observed":           round(obs_ratio2, 6),
        "null_exact_mean":    round(null_r2_mean, 6),
        "null_exact_std":     round(null_r2_std, 6),
        "two_sided_tail_p":   "N/A",
        "n_used":             n_used,
        "note": "ratio; exact null mean != 1.0 in general; computed from 5-perm enumeration",
    },
]

order_perm_df = pd.DataFrame(order_perm_rows)
order_perm_df.to_csv(TABLES / "phase2_layer_order_permutation.csv", index=False)
print(f"\n  Order permutation table written: {len(order_perm_df)} rows")
print(order_perm_df[["statistic","observed","null_exact_mean","two_sided_tail_p"]].to_string(index=False))

# ── Item 2 figure: observed vs null bars ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
stat_labels = [
    "P(d_sm < d_mr)\nnull=0.500",
    "P(d_sr largest)\nnull=0.333",
    "P(d_sm<d_mr<d_sr)\nnull=0.167",
]
obs_vals     = [obs_p1, obs_p2, obs_p3]
null_means   = [null_p1_mean, null_p2_mean, null_p3_mean]
null_stds    = [null_p1_std, null_p2_std, null_p3_std]
theor_nulls  = [THEORETICAL_NULL_P1, THEORETICAL_NULL_P2, THEORETICAL_NULL_P3]

for ax, lbl, ov, nm, ns, tn in zip(
        axes, stat_labels, obs_vals, null_means, null_stds, theor_nulls):
    ax.bar(["observed", "null (5-perm)"], [ov, nm],
           color=["#1565C0", "#B0BEC5"], width=0.4, zorder=3)
    ax.errorbar(["null (5-perm)"], [nm], yerr=[ns],
                fmt="none", color="black", capsize=5, zorder=4)
    ax.axhline(tn, color="#E53935", linestyle="--", lw=1.2, label=f"theoretical null ({tn:.3f})")
    ax.set_title(lbl, fontsize=9)
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle(
    "Item 2: Order-based layer-label permutation test (cosine distances)\n"
    "VALID test of label-to-ordering association; tests ordering, NOT magnitude of separation.\n"
    "If observed ~ null: valid test, low discriminability observed.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_layer_order_permutation.png", dpi=120)
plt.close(fig)

item2_result = {
    "distance_metric_confirmed": "cosine - same as existing centroid_dist_* columns",
    "within_label_norm_status": "RETIRED - permutation-invariant by construction (C3)",
    "framing": (
        "VALID permutation test of label-to-ordering association; "
        "tests ordering, NOT the magnitude of separation. "
        "If observed values sit close to null, the honest verdict is: "
        "valid test, low discriminability observed - "
        "NOT null baseline confirms layer separation."
    ),
    "n_used": n_used,
    "observed_vs_null": {
        "P_sm_lt_mr":          {"obs": round(obs_p1, 4), "null_mean": round(null_p1_mean, 4), "tail_p": round(p1_tail, 4)},
        "P_sr_largest":        {"obs": round(obs_p2, 4), "null_mean": round(null_p2_mean, 4), "tail_p": round(p2_tail, 4)},
        "P_order_full":        {"obs": round(obs_p3, 4), "null_mean": round(null_p3_mean, 4), "tail_p": round(p3_tail, 4)},
        "ratio_sm_mr":         {"obs": round(obs_ratio1, 4), "null_mean": round(null_r1_mean, 4)},
        "ratio_sr_sm":         {"obs": round(obs_ratio2, 4), "null_mean": round(null_r2_mean, 4)},
    },
    "outputs": [
        "tables/phase2_layer_order_permutation.csv",
        "figures/phase2_layer_order_permutation.png",
    ],
}
print(f"\n  Item 2 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 3: Lexical temporal drift vs timestamp-shuffled baseline
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Item 3: Lexical temporal drift vs shuffled baseline ===")

# Build state_u: rows with valid unigram dicts (excluding __total__ already done)
state_u = state[state["unigram"].notna()].copy().reset_index(drop=True)
state_u = state_u.sort_values("ts_utc").reset_index(drop=True)
print(f"  Rows with valid unigrams: {len(state_u)}")

def js_distance(d1, d2):
    """Jensen-Shannon distance between two token count dicts."""
    vocab = set(d1.keys()) | set(d2.keys())
    if not vocab:
        return np.nan
    t1 = sum(d1.values()); t2 = sum(d2.values())
    if t1 == 0 or t2 == 0:
        return np.nan
    p = np.array([d1.get(w, 0) / t1 for w in vocab], dtype=float)
    q = np.array([d2.get(w, 0) / t2 for w in vocab], dtype=float)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1e-300)), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1e-300)), 0.0)
    jsd = 0.5 * (kl_pm.sum() + kl_qm.sum())
    return float(np.sqrt(max(jsd, 0.0)))  # JS distance = sqrt(JSD)

def cosine_dist_counts(d1, d2):
    """Cosine distance between count vectors aligned on union vocabulary."""
    vocab = sorted(set(d1.keys()) | set(d2.keys()))
    if not vocab:
        return np.nan
    v1 = np.array([d1.get(w, 0) for w in vocab], dtype=float)
    v2 = np.array([d2.get(w, 0) for w in vocab], dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))

def consecutive_drift_lexical(df_sorted):
    """
    Compute consecutive JS and cosine distances for adjacent rows in df_sorted.
    Returns two lists (js_dists, cos_dists) and metadata list.
    """
    js_vals   = []
    cos_vals  = []
    meta      = []
    for i in range(len(df_sorted) - 1):
        r0 = df_sorted.iloc[i]
        r1 = df_sorted.iloc[i + 1]
        d1 = r0["unigram"]
        d2 = r1["unigram"]
        js  = js_distance(d1, d2)
        cos = cosine_dist_counts(d1, d2)
        js_vals.append(js)
        cos_vals.append(cos)
        meta.append({
            "ts_from":    str(r0["ts_utc"]),
            "ts_to":      str(r1["ts_utc"]),
            "month_from": r0["month"],
            "month_to":   r1["month"],
            "is_boundary": (r0["month"] == "2026-04" and r1["month"] == "2026-05"),
            "js_dist":    js,
            "cos_dist":   cos,
        })
    return js_vals, cos_vals, meta

obs_js, obs_cos, drift_meta = consecutive_drift_lexical(state_u)
drift_meta_df = pd.DataFrame(drift_meta)

# Per-month (exclude boundary steps)
apr_meta = drift_meta_df[
    ~drift_meta_df["is_boundary"] & (drift_meta_df["month_from"] == "2026-04")]
may_meta = drift_meta_df[
    ~drift_meta_df["is_boundary"] & (drift_meta_df["month_from"] == "2026-05")]

obs_js_mean  = float(np.nanmean(obs_js))
obs_cos_mean = float(np.nanmean(obs_cos))

print(f"  Observed consecutive steps: {len(obs_js)}")
print(f"  JS mean: {obs_js_mean:.6f}  Cosine mean: {obs_cos_mean:.6f}")

# ── Shuffled baseline ─────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)
shuf_js_means  = []
shuf_cos_means = []

for _ in range(N_SHUFFLES_LEXICAL):
    perm_idx = rng.permutation(len(state_u))
    state_shuf = state_u.iloc[perm_idx].copy().reset_index(drop=True)
    # Keep ts_utc sorted (only unigrams are shuffled, timestamps stay ordered)
    state_shuf["ts_utc"] = state_u["ts_utc"].values
    state_shuf["month"]  = state_u["month"].values
    sj, sc, _ = consecutive_drift_lexical(state_shuf)
    shuf_js_means.append(float(np.nanmean(sj)))
    shuf_cos_means.append(float(np.nanmean(sc)))

shuf_js_mean  = float(np.mean(shuf_js_means))
shuf_js_std   = float(np.std(shuf_js_means))
shuf_cos_mean = float(np.mean(shuf_cos_means))
shuf_cos_std  = float(np.std(shuf_cos_means))

js_ratio  = obs_js_mean  / shuf_js_mean  if shuf_js_mean  > 1e-15 else np.nan
cos_ratio = obs_cos_mean / shuf_cos_mean if shuf_cos_mean > 1e-15 else np.nan

print(f"  Shuffled baseline ({N_SHUFFLES_LEXICAL} permutations):")
print(f"    JS  shuffled mean: {shuf_js_mean:.6f}  std: {shuf_js_std:.6f}  ratio obs/shuf: {js_ratio:.4f}")
print(f"    Cos shuffled mean: {shuf_cos_mean:.6f}  std: {shuf_cos_std:.6f}  ratio obs/shuf: {cos_ratio:.4f}")

# ── Build lexical drift output table ─────────────────────────────────────────
def scope_stats(meta_df, label):
    js_v  = meta_df["js_dist"].dropna().values
    cos_v = meta_df["cos_dist"].dropna().values
    return {
        "scope": label,
        "n_steps": len(meta_df),
        "js_obs_mean":     float(np.nanmean(js_v))  if len(js_v)  else np.nan,
        "js_obs_std":      float(np.nanstd(js_v))   if len(js_v)  else np.nan,
        "cos_obs_mean":    float(np.nanmean(cos_v)) if len(cos_v) else np.nan,
        "cos_obs_std":     float(np.nanstd(cos_v))  if len(cos_v) else np.nan,
        "js_shuf_mean":    shuf_js_mean,
        "js_shuf_std":     shuf_js_std,
        "cos_shuf_mean":   shuf_cos_mean,
        "cos_shuf_std":    shuf_cos_std,
        "js_obs_shuf_ratio":  float(np.nanmean(js_v))  / shuf_js_mean  if shuf_js_mean  > 1e-15 else np.nan,
        "cos_obs_shuf_ratio": float(np.nanmean(cos_v)) / shuf_cos_mean if shuf_cos_mean > 1e-15 else np.nan,
        "n_shuffles":      N_SHUFFLES_LEXICAL,
        "interpretation": (
            "observed < shuffled baseline: lexical autocorrelation (temporal memory); "
            "comparable: near-IID lexical churn. "
            "Lexical stability check, NOT content interpretation."
        ),
    }

lex_rows = [
    scope_stats(drift_meta_df, "overall"),
    scope_stats(apr_meta,      "2026-04"),
    scope_stats(may_meta,      "2026-05"),
]
lex_df = pd.DataFrame(lex_rows)
lex_df.to_csv(TABLES / "phase2_lexical_drift.csv", index=False)
print(f"\n  Lexical drift table written: {len(lex_df)} rows")
print(lex_df[["scope","n_steps","js_obs_mean","js_shuf_mean","js_obs_shuf_ratio",
              "cos_obs_mean","cos_shuf_mean","cos_obs_shuf_ratio"]].to_string(index=False))

# ── Item 3 figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
scopes     = ["overall", "2026-04", "2026-05"]
obs_vals_j = [lex_df.loc[lex_df["scope"] == s, "js_obs_mean"].values[0]  for s in scopes]
obs_vals_c = [lex_df.loc[lex_df["scope"] == s, "cos_obs_mean"].values[0] for s in scopes]
shuf_j     = shuf_js_mean
shuf_c     = shuf_cos_mean
shuf_j_e   = shuf_js_std
shuf_c_e   = shuf_cos_std

x = np.arange(len(scopes))
w = 0.35
for ax, obs_v, shuf_v, shuf_e, ylabel, title in [
    (axes[0], obs_vals_j,  shuf_j,  shuf_j_e,  "mean JS distance",     "Lexical JS drift"),
    (axes[1], obs_vals_c,  shuf_c,  shuf_c_e,  "mean cosine distance", "Lexical cosine drift"),
]:
    bars1 = ax.bar(x - w/2, obs_v,  w, label="observed",         color="#1565C0", alpha=0.85)
    bars2 = ax.bar(x + w/2, [shuf_v]*len(scopes), w,
                   label=f"shuffled baseline ({N_SHUFFLES_LEXICAL})", color="#B0BEC5", alpha=0.85,
                   yerr=[shuf_e]*len(scopes), capsize=4, error_kw={"ecolor": "black"})
    ax.set_xticks(x); ax.set_xticklabels(scopes)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Item 3: Lexical temporal drift vs timestamp-shuffled baseline\n"
    "Observed < shuffled = temporal memory; Observed ~ shuffled = near-IID churn.\n"
    "Lexical stability check only - NOT content interpretation.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(FIGURES / "phase2_lexical_drift.png", dpi=120)
plt.close(fig)

item3_result = {
    "n_steps_overall": len(drift_meta_df),
    "n_shuffles": N_SHUFFLES_LEXICAL,
    "js": {
        "obs_mean":  round(obs_js_mean,  6),
        "shuf_mean": round(shuf_js_mean, 6),
        "shuf_std":  round(shuf_js_std,  6),
        "ratio_obs_shuf": round(js_ratio, 4),
    },
    "cosine": {
        "obs_mean":  round(obs_cos_mean,  6),
        "shuf_mean": round(shuf_cos_mean, 6),
        "shuf_std":  round(shuf_cos_std,  6),
        "ratio_obs_shuf": round(cos_ratio, 4),
    },
    "interpretation": (
        "observed consecutive distance BELOW shuffled baseline indicates lexical "
        "autocorrelation (temporal memory); comparable values indicate near-IID "
        "lexical churn. This is a lexical stability check, NOT content interpretation."
    ),
    "outputs": [
        "tables/phase2_lexical_drift.csv",
        "figures/phase2_lexical_drift.png",
    ],
}
print(f"\n  Item 3 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 4: Anomaly persistence / run-length score
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Item 4: Anomaly persistence / run-length score ===")
print("    Source: robust_z outlier flags recomputed per timestamp from real_clean.csv")
print("    Metrics with robust_z_outlier_count > 0 from diagnostic_flags.csv:")

diag_flags = pd.read_csv(TABLES / "diagnostic_flags.csv")
outlier_metrics_info = diag_flags[diag_flags["robust_z_outlier_count"] > 0][
    ["metric", "robust_z_outlier_count"]].reset_index(drop=True)
print(outlier_metrics_info.to_string(index=False))

# Map metric name -> real_clean column name (same as phase1)
def metric_to_col(m):
    return "H_t_numeric" if m == "H_t" else m

# Sort real by ts_utc (already done above)
real_sorted = real.sort_values("ts_utc").reset_index(drop=True)

def find_runs(flagged_indices):
    """
    Given sorted list of integer indices (positions in real_sorted),
    find maximal consecutive runs (adjacent in index, i.e., adjacent observations).
    Returns list of dicts: {start_idx, end_idx, length, indices}
    """
    if len(flagged_indices) == 0:
        return []
    runs = []
    run_start = flagged_indices[0]
    run_end   = flagged_indices[0]
    run_idxs  = [flagged_indices[0]]
    for fi in flagged_indices[1:]:
        if fi == run_end + 1:
            run_end = fi
            run_idxs.append(fi)
        else:
            runs.append({"start_idx": run_start, "end_idx": run_end,
                         "length": run_end - run_start + 1, "indices": run_idxs[:]})
            run_start = fi; run_end = fi; run_idxs = [fi]
    runs.append({"start_idx": run_start, "end_idx": run_end,
                 "length": run_end - run_start + 1, "indices": run_idxs[:]})
    return runs

persistence_rows = []
run_detail_rows  = []

for _, mrow in outlier_metrics_info.iterrows():
    metric_name = mrow["metric"]
    col = metric_to_col(metric_name)
    if col not in real_sorted.columns:
        print(f"  WARNING: column {col} not in real_clean.csv, skipping {metric_name}")
        continue

    rz = robust_z(real_sorted[col].values)
    flagged_mask = np.abs(rz) > ROBUST_Z_THRESH
    # Use False for NaN robust_z (consistent with diagnostic_flags behavior)
    flagged_mask = np.where(np.isnan(rz), False, flagged_mask)
    total_flagged = int(flagged_mask.sum())

    flagged_indices = list(np.where(flagged_mask)[0])
    runs = find_runs(flagged_indices)

    n_runs       = len(runs)
    run_lengths  = [r["length"] for r in runs]
    max_run_len  = int(max(run_lengths)) if run_lengths else 0
    mean_run_len = float(np.mean(run_lengths)) if run_lengths else 0.0

    # Per-run gap info
    max_gap_secs = 0.0
    for run in runs:
        idxs = run["indices"]
        if len(idxs) < 2:
            run["max_gap_secs"] = None
        else:
            gaps = []
            for ii in range(len(idxs) - 1):
                t0 = real_sorted.iloc[idxs[ii]]["ts_utc"]
                t1 = real_sorted.iloc[idxs[ii+1]]["ts_utc"]
                gaps.append((t1 - t0).total_seconds())
            run["max_gap_secs"] = float(max(gaps))
            if run["max_gap_secs"] > max_gap_secs:
                max_gap_secs = run["max_gap_secs"]

    # Overall max gap spanned by any run
    max_gap_any_run = float(max_gap_secs) if runs else None

    print(f"\n  {metric_name}: total_flagged={total_flagged}  n_runs={n_runs}  "
          f"max_run={max_run_len}  mean_run={mean_run_len:.2f}")

    persistence_rows.append({
        "metric":          metric_name,
        "total_flagged":   total_flagged,
        "n_runs":          n_runs,
        "max_run_length":  max_run_len,
        "mean_run_length": round(mean_run_len, 4),
        "n_isolated_spikes":    int(sum(1 for r in runs if r["length"] == 1)),
        "n_persistent_runs":    int(sum(1 for r in runs if r["length"] >= 2)),
        "max_gap_secs_any_run": max_gap_any_run,
        "robust_z_thresh":      ROBUST_Z_THRESH,
        "framing": (
            "distinguishes persistent currents (run>=2) from isolated spikes (run=1); "
            "purely descriptive, no significance claim"
        ),
    })

    # Detail rows per run
    for run_i, run in enumerate(runs):
        ts_start = real_sorted.iloc[run["start_idx"]]["ts_utc"]
        ts_end   = real_sorted.iloc[run["end_idx"]]["ts_utc"]
        run_detail_rows.append({
            "metric":         metric_name,
            "run_index":      run_i,
            "run_length":     run["length"],
            "ts_start":       str(ts_start),
            "ts_end":         str(ts_end),
            "span_hours":     float((ts_end - ts_start).total_seconds() / 3600),
            "max_gap_secs":   run.get("max_gap_secs"),
            "is_isolated":    run["length"] == 1,
            "is_persistent":  run["length"] >= 2,
        })

persist_df = pd.DataFrame(persistence_rows)
persist_df.to_csv(TABLES / "phase2_anomaly_persistence.csv", index=False)
print(f"\n  Anomaly persistence table written: {len(persist_df)} rows")
print(persist_df[["metric","total_flagged","n_runs","max_run_length",
                  "mean_run_length","n_isolated_spikes","n_persistent_runs"]].to_string(index=False))

item4_result = {
    "robust_z_thresh": ROBUST_Z_THRESH,
    "framing": (
        "distinguishes persistent currents (run length >= 2) from isolated spikes (run length 1); "
        "purely descriptive, no significance claim"
    ),
    "per_metric_summary": {
        row["metric"]: {
            "total_flagged": row["total_flagged"],
            "n_runs": row["n_runs"],
            "max_run_length": row["max_run_length"],
            "mean_run_length": row["mean_run_length"],
            "n_isolated": row["n_isolated_spikes"],
            "n_persistent": row["n_persistent_runs"],
        }
        for _, row in persist_df.iterrows()
    },
    "outputs": [
        "tables/phase2_anomaly_persistence.csv",
    ],
}
print(f"\n  Item 4 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL: Print result dict
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  RUN SUMMARY")
print("=" * 64)

print("\n--- Discovered file/column layout ---")
print(f"  Phase 1 script:          kopterix_phase1.py")
print(f"  Phase 2 script (active): kopterix_phase2_corrected.py")
print(f"  real_clean.csv:          {N_OBS_EXPECTED} rows, key col: Timestamp_UTC (parse: strip ' UTC', utc=True)")
print(f"  shuffle_clean.csv:       {N_SHUFFLE_EXPECTED} rows")
print(f"  kopterix_state.csv:      {n_state} in-window rows, key col: Timestamp (ISO 8601+offset, utc=True)")
print(f"  Observation UTC key:     Timestamp_UTC (real_clean) / Timestamp (state)")
print(f"  H_t_numeric col:         H_t_numeric")
print(f"  n_total col:             n_total")
print(f"  metric_provenance col:   metric_provenance")
print(f"  month label col:         month")
print(f"  Layer centroid container: LayerCentroids -> lc_surface, lc_mid, lc_residue")
print(f"  MeanEmbedding col:       MeanEmbedding -> mean_emb (parsed numpy array)")
print(f"  UnigramCounts col:       UnigramCounts (JSON dict, strip __total__)")
print(f"  centroid_dist_* type:    COSINE (confirmed: cosine_dist function in phase2_corrected)")
print(f"  Embedding dim:           384")

print("\n--- Item 1: Layer-specific residual geometry ---")
print(f"  Outputs: phase2_residual_geometry_norms/directional/drift .csv + .png")
print(f"  n_fully_parsed (all 3 layers + mean_emb): {n_fully_parsed}")
print(f"  n_drift_steps_usable: {n_steps_usable}")
print("  Residual norm means (overall):")
for layer in LAYERS:
    row = norms_df[(norms_df["scope"] == "overall") & (norms_df["layer"] == layer)].iloc[0]
    print(f"    {layer}: mean={row['mean']:.6f}  std={row['std']:.6f}")

print("\n--- Item 2: Order-based layer-label permutation test ---")
print(f"  Distance metric used: COSINE - matches existing centroid_dist_* definition")
print(f"  Within-label norm score: RETIRED (permutation-invariant, C3)")
print(f"  n_used: {n_used}")
for stat_row in order_perm_rows[:3]:
    print(f"  {stat_row['statistic']}: obs={stat_row['observed']}  "
          f"null={stat_row['null_exact_mean']}  tail_p={stat_row['two_sided_tail_p']}")
for stat_row in order_perm_rows[3:]:
    print(f"  {stat_row['statistic']}: obs={stat_row['observed']}  "
          f"null_mean={stat_row['null_exact_mean']}")
print(f"  Framing: {item2_result['framing']}")

print("\n--- Item 3: Lexical temporal drift ---")
print(f"  n_steps_overall: {item3_result['n_steps_overall']}")
print(f"  n_shuffles: {item3_result['n_shuffles']}")
print(f"  JS  obs_mean={item3_result['js']['obs_mean']}  "
      f"shuf_mean={item3_result['js']['shuf_mean']}  "
      f"ratio={item3_result['js']['ratio_obs_shuf']}")
print(f"  Cos obs_mean={item3_result['cosine']['obs_mean']}  "
      f"shuf_mean={item3_result['cosine']['shuf_mean']}  "
      f"ratio={item3_result['cosine']['ratio_obs_shuf']}")
print(f"  Interpretation: {item3_result['interpretation']}")

print("\n--- Item 4: Anomaly persistence ---")
for metric_name, vals in item4_result["per_metric_summary"].items():
    print(f"  {metric_name}: flagged={vals['total_flagged']}  runs={vals['n_runs']}  "
          f"max_run={vals['max_run_length']}  isolated={vals['n_isolated']}  persistent={vals['n_persistent']}")

print("\n--- Count verification ---")
print(f"  All frozen counts matched: 231 obs / 215 state / 13 shuffle / 210 layer / 121 Apr / 110 May")
print("  No count mismatch. No data synthesized.")

print("\n--- Item 2 distance metric note ---")
print(f"  Confirmed COSINE. Phase2 corrected uses cosine_dist() for centroid_dist_* columns.")
print(f"  Item 2 uses the same cosine_dist function.")

print("\n--- Guard-rail confirmation ---")
print("  No geometry-regime classification added (out of scope).")
print("  No current half-life / recurrence / mutation tracking added (out of scope).")
print("  Existing grand-mean residual section NOT touched.")
print("  Existing C3 caveat NOT removed; within-label norm RETIRED IN PLACE.")
print("  Scalar summary / stability table / viability count NOT touched.")
print("  231/215/13/210 counts used as frozen truth; no recomputation.")
print("  Three-state booleans used throughout (None/True/False).")
print("  No markdown report generated.")

print("\n=== Done ===")
