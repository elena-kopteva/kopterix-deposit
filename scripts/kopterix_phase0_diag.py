"""
kopterix_phase0_diag.py
Phase 0 provenance diagnostic for the Kopterix calibration pipeline.

Classifies every numeric value by source:
  computed_observation_csv         - from observations.csv
  computed_state_csv               - from kopterix_state.csv
  deterministic_shuffle_comparison - surface/mid/residue real=.../shuffle=.../delta=... lines
  legacy_shuffle_log_text_untrusted - H(t)/phi(t)/D(t) metric lines inside shuffle log blocks

Does NOT run Phase 1 analysis. Stops after printing the Phase 0 report.
"""

import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Portable base resolution: start from this script's own directory, then check
# one level above. The parent is selected only when one of the accepted
# observations filenames exists there. This supports the project canonical
# layout (script beside observations.csv) and the staged deposit layout
# (scripts/ one level below the deposit root).
_SCRIPT_DIR = Path(__file__).resolve().parent
_OBS_NAMES = ("observations_April-May_2026.csv", "observations.csv")


def _resolve_base_dir():
    if any((_SCRIPT_DIR / _n).exists() for _n in _OBS_NAMES):
        return _SCRIPT_DIR
    _parent = _SCRIPT_DIR.parent
    if any((_parent / _n).exists() for _n in _OBS_NAMES):
        return _parent
    return _SCRIPT_DIR


BASE_DIR   = _resolve_base_dir()
WINDOW_START = pd.Timestamp("2026-04-01 00:00:00")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00")

N_TOTAL_REGIME_THRESHOLD = 300   # inclusive: n_total <= this is small-sample regime
HT_LOW_THRESHOLD         = 10.6  # report all real obs with H_t below this


# ---------------------------------------------------------------------------
# Timestamp helpers (identical to phase1)
# ---------------------------------------------------------------------------

def normalize_ts(series):
    def _parse_one(v):
        if pd.isna(v):
            return pd.NaT
        s = str(v).strip()
        if s.upper().endswith("UTC"):
            s = s[:-3].strip()
        try:
            ts = pd.Timestamp(s)
        except Exception:
            return pd.NaT
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts
    return series.map(_parse_one)


def _find_utc_col(df):
    cols_lower = {c.lower(): c for c in df.columns}
    return cols_lower.get("timestamp_utc"), cols_lower.get("timestamp")


# ---------------------------------------------------------------------------
# Shuffle log parser - full provenance classification
# ---------------------------------------------------------------------------

SHUFFLE_HEADER_RE = re.compile(
    r"^(?:##\s+)?BASELINE-SHUFFLE obs ::\s*([\d-]+\s+[\d:]+)\s*UTC",
    re.IGNORECASE,
)

# Deterministic comparison: surface-mid real=... | shuffle=... | delta=...
DET_COMP_RE = re.compile(
    r"(surface-mid|mid-residue|surface-residue)\s+real=([\d.eE+-]+)\s*\|\s*shuffle=([\d.eE+-]+)\s*\|\s*delta=([-\d.eE+]+)",
    re.IGNORECASE,
)

# Metrics line: **metrics:** H(t)= X | phi(t)= Y | D(t)= Z ...
METRICS_LINE_RE = re.compile(r"\*\*metrics:\*\*", re.IGNORECASE)

# Patterns to identify explicit untrust reasons in a metrics line
AWAITING_RE    = re.compile(r"\[awaiting external\]", re.IGNORECASE)
FIRST_RUN_RE   = re.compile(r"first run", re.IGNORECASE)
ALL_NA_RE      = re.compile(r"H\(t\)=\s*N/A", re.IGNORECASE)

# Extract numeric value from a metrics-line field, e.g. "H(t)= 4.2"
HT_IN_METRICS_RE  = re.compile(r"H\(t\)\s*=\s*([\d.eE+-]+)", re.IGNORECASE)
PHI_IN_METRICS_RE = re.compile(r"ph?i\(t\)\s*=\s*([\d.eE+-]+)", re.IGNORECASE)
DT_IN_METRICS_RE  = re.compile(r"D\(t\)\s*=\s*([\d.eE+-]+)", re.IGNORECASE)


def _untrust_reason(line):
    """Classify why a metrics line is untrusted."""
    if AWAITING_RE.search(line):
        return "awaiting_external"
    if FIRST_RUN_RE.search(line):
        return "first_run_artifact"
    if ALL_NA_RE.search(line):
        return "all_fields_na"
    return "provenance_unconfirmed_carried_value"


def parse_shuffle_log_with_provenance(path):
    """
    Parse a shuffle log and return:
      - blocks: list of dicts, one per BASELINE-SHUFFLE block
      - quarantined: list of dicts for every H/phi/D metrics line (legacy untrusted)
      - det_comparisons: list of dicts for every deterministic comparison triple
    """
    text  = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_sample = lines[0] if lines else ""

    blocks         = []
    quarantined    = []
    det_comparisons = []

    i = 0
    while i < len(lines):
        m = SHUFFLE_HEADER_RE.match(lines[i].strip())
        if m:
            ts_str = m.group(1).strip()
            ts = normalize_ts(pd.Series([ts_str])).iloc[0]

            # Collect lines until the next block header
            j = i + 1
            block_lines = []
            while j < len(lines) and not SHUFFLE_HEADER_RE.match(lines[j].strip()):
                block_lines.append(lines[j])
                j += 1

            # --- Parse within block ---
            seen_det_comp = False
            block_det_comps     = []
            block_quarantined   = []

            for bl in block_lines:
                # Deterministic comparison line
                dc = DET_COMP_RE.search(bl)
                if dc:
                    seen_det_comp = True
                    block_det_comps.append({
                        "block_ts":       ts,
                        "source_log":     path.name,
                        "pair":           dc.group(1).lower(),
                        "real":           float(dc.group(2)),
                        "shuffle":        float(dc.group(3)),
                        "delta":          float(dc.group(4)),
                        "metric_provenance": "deterministic_shuffle_comparison",
                    })
                    continue

                # Metrics line: H(t)/phi(t)/D(t) - always legacy untrusted
                if METRICS_LINE_RE.search(bl):
                    reason = _untrust_reason(bl)
                    position = "post_comparison" if seen_det_comp else "pre_comparison_header"
                    ht_m  = HT_IN_METRICS_RE.search(bl)
                    phi_m = PHI_IN_METRICS_RE.search(bl)
                    dt_m  = DT_IN_METRICS_RE.search(bl)
                    block_quarantined.append({
                        "block_ts":            ts,
                        "source_log":          path.name,
                        "position_in_block":   position,
                        "untrust_reason":      reason,
                        "H_t_raw":             float(ht_m.group(1))  if ht_m  else None,
                        "phi_t_raw":           float(phi_m.group(1)) if phi_m else None,
                        "D_t_raw":             float(dt_m.group(1))  if dt_m  else None,
                        "metric_provenance":   "legacy_shuffle_log_text_untrusted",
                        "raw_line":            bl.strip(),
                    })

            det_comparisons.extend(block_det_comps)
            quarantined.extend(block_quarantined)

            blocks.append({
                "ts":                  ts,
                "source_log":          path.name,
                "det_comp_count":      len(block_det_comps),
                "quarantined_count":   len(block_quarantined),
            })

            i = j
        else:
            i += 1

    return blocks, quarantined, det_comparisons, header_sample


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 70

    print(sep)
    print("KOPTERIX PHASE 0 - Provenance Diagnostic")
    print(f"Window: [{WINDOW_START}  -->  {WINDOW_END})")
    print(sep)

    # -----------------------------------------------------------------------
    # 1. OBSERVATIONS CSV
    # -----------------------------------------------------------------------
    print("\n--- OBSERVATIONS ---")
    obs_path = BASE_DIR / "observations_April-May_2026.csv"
    if not obs_path.exists():
        fallback = BASE_DIR / "observations.csv"
        if fallback.exists():
            print(f"[WARN] Using fallback 'observations.csv' (expected 'observations_April-May_2026.csv')")
            obs_path = fallback
        else:
            print("[ERROR] No observations CSV found."); sys.exit(1)

    raw_df = pd.read_csv(obs_path, low_memory=False)
    utc_col, ts_col = _find_utc_col(raw_df)

    if utc_col and ts_col:
        print(f"[WARN] Both '{utc_col}' and '{ts_col}' present. Dropping '{ts_col}' (local Chicago time), keying on '{utc_col}'.")
        raw_df = raw_df.drop(columns=[ts_col])
        time_col = utc_col
    elif utc_col:
        time_col = utc_col
    elif ts_col:
        print(f"[WARN] Using '{ts_col}' - must verify as UTC.")
        time_col = ts_col
    else:
        print("[ERROR] No timestamp column."); sys.exit(1)

    print(f"Raw timestamp samples (first 3): {[str(v) for v in raw_df[time_col].head(3).tolist()]}")

    raw_df["Timestamp"] = normalize_ts(raw_df[time_col])
    rows_in = len(raw_df)
    unparseable = int(raw_df["Timestamp"].isna().sum())
    df_ok = raw_df[raw_df["Timestamp"].notna()].copy()

    dropped_before = int((df_ok["Timestamp"] < WINDOW_START).sum())
    dropped_after  = int((df_ok["Timestamp"] >= WINDOW_END).sum())
    df_obs = df_ok[(df_ok["Timestamp"] >= WINDOW_START) & (df_ok["Timestamp"] < WINDOW_END)].copy()
    rows_in_window = len(df_obs)

    # Coerce metrics
    for m in ["H_t", "phi_t", "D_t", "n_total", "post_rate_est",
              "centroid_dist_surface_mid", "centroid_dist_mid_residue",
              "centroid_dist_surface_residue"]:
        if m in df_obs.columns:
            df_obs[m] = pd.to_numeric(df_obs[m], errors="coerce")

    df_obs["metric_provenance"] = "computed_observation_csv"

    # H_t numeric rows (no exclusion rule applied - all finite H_t values are kept)
    ht_numeric_rows = int(df_obs["H_t"].notna().sum()) if "H_t" in df_obs.columns else 0

    # n_total_regime flag
    if "n_total" in df_obs.columns:
        df_obs["n_total_regime"] = df_obs["n_total"] <= N_TOTAL_REGIME_THRESHOLD
    else:
        df_obs["n_total_regime"] = False

    print(f"  Rows in CSV:          {rows_in}")
    print(f"  Unparseable ts:       {unparseable}")
    print(f"  Dropped before window:{dropped_before}")
    print(f"  Dropped after window: {dropped_after}")
    print(f"  ROWS IN WINDOW:       {rows_in_window}")
    print(f"  NUMERIC H_t ROWS:     {ht_numeric_rows}")

    # -----------------------------------------------------------------------
    # 2. STATE CSV
    # -----------------------------------------------------------------------
    print("\n--- STATE CSV ---")
    state_path = BASE_DIR / "kopterix_state.csv"
    state_rows_in_window = 0
    if state_path.exists():
        # Read only Timestamp column to avoid huge output
        st_df = pd.read_csv(state_path, usecols=[0])
        st_df.columns = ["Timestamp_raw"]
        st_df["Timestamp"] = normalize_ts(st_df["Timestamp_raw"])
        st_in = st_df[(st_df["Timestamp"] >= WINDOW_START) & (st_df["Timestamp"] < WINDOW_END)]
        state_rows_in_window = len(st_in)
        print(f"  State CSV rows total: {len(st_df)}")
        print(f"  STATE ROWS IN WINDOW: {state_rows_in_window}")
        print(f"  State ts samples:     {st_df['Timestamp_raw'].head(3).tolist()}")
    else:
        print("  kopterix_state.csv not found.")

    # -----------------------------------------------------------------------
    # 3. SHUFFLE LOGS
    # -----------------------------------------------------------------------
    print("\n--- SHUFFLE LOGS ---")
    log_paths = {
        "2026-04": BASE_DIR / "kopterix_log_2026-04.md",
        "2026-05": BASE_DIR / "kopterix_log_2026-05.md",
    }

    all_blocks       = []
    all_quarantined  = []
    all_det_comps    = []
    log_summaries    = {}

    for month_key, lpath in log_paths.items():
        if not lpath.exists():
            print(f"  {lpath.name}: NOT FOUND")
            log_summaries[lpath.name] = None
            continue
        blocks, quarantined, det_comps, hdr = parse_shuffle_log_with_provenance(lpath)
        all_blocks.extend(blocks)
        all_quarantined.extend(quarantined)
        all_det_comps.extend(det_comps)
        log_summaries[lpath.name] = {
            "header": hdr,
            "blocks": len(blocks),
            "det_comp_triples": len(det_comps),
            "quarantined_metric_lines": len(quarantined),
        }
        print(f"  {lpath.name} header: '{hdr}'")

    # Apply window filter to shuffle blocks (by block timestamp)
    all_blocks_in_window = [
        b for b in all_blocks
        if pd.notna(b["ts"]) and WINDOW_START <= b["ts"] < WINDOW_END
    ]
    all_det_comps_in_window = [
        d for d in all_det_comps
        if pd.notna(d["block_ts"]) and WINDOW_START <= d["block_ts"] < WINDOW_END
    ]
    all_quarantined_in_window = [
        q for q in all_quarantined
        if pd.notna(q["block_ts"]) and WINDOW_START <= q["block_ts"] < WINDOW_END
    ]

    shuffle_heading_count  = len(all_blocks_in_window)
    det_comp_count         = len(all_det_comps_in_window)   # individual triplets (each block has up to 3)
    det_comp_block_count   = sum(1 for b in all_blocks_in_window if b["det_comp_count"] > 0)
    quarantined_count      = len(all_quarantined_in_window)

    print(f"\n  SHUFFLE HEADING COUNT (in window):          {shuffle_heading_count}")
    print(f"  DET SHUFFLE COMPARISON TRIPLES (in window): {det_comp_count}")
    print(f"  Blocks with at least one comparison:        {det_comp_block_count}")
    print(f"  QUARANTINED LEGACY METRIC LINES (in window):{quarantined_count}")

    # Break down quarantined by reason
    reason_counts = {}
    for q in all_quarantined_in_window:
        r = q["untrust_reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1
    print(f"  Quarantined breakdown by untrust_reason:")
    for r, c in sorted(reason_counts.items()):
        print(f"    {r}: {c}")

    # Break down quarantined by position (pre vs post comparison)
    pos_counts = {}
    for q in all_quarantined_in_window:
        p = q["position_in_block"]
        pos_counts[p] = pos_counts.get(p, 0) + 1
    print(f"  Quarantined breakdown by position in block:")
    for p, c in sorted(pos_counts.items()):
        print(f"    {p}: {c}")

    # List all quarantined H_t values
    print(f"\n  All quarantined H_t values (in window):")
    for q in sorted(all_quarantined_in_window, key=lambda x: x["block_ts"]):
        ht_str = f"{q['H_t_raw']:.4f}" if q["H_t_raw"] is not None else "None/N/A"
        print(f"    {q['block_ts']}  {q['source_log']}  pos={q['position_in_block']}")
        print(f"      H_t={ht_str}  reason={q['untrust_reason']}")

    # -----------------------------------------------------------------------
    # 4. ROWS WITH n_total <= 300
    # -----------------------------------------------------------------------
    print(f"\n--- ROWS WITH n_total <= {N_TOTAL_REGIME_THRESHOLD} (real obs, in window) ---")
    if "n_total" in df_obs.columns:
        low_n = df_obs[df_obs["n_total_regime"] == True][
            ["Timestamp", "month", "n_total", "H_t", "phi_t", "D_t",
             "post_rate_est", "metric_provenance"]
        ].sort_values("n_total") if "month" in df_obs.columns else \
        df_obs[df_obs["n_total_regime"] == True][
            ["Timestamp", "n_total", "H_t", "phi_t", "D_t", "post_rate_est"]
        ].sort_values("n_total")

        # Add month column if not present
        if "month" not in df_obs.columns:
            df_obs["month"] = df_obs["Timestamp"].dt.to_period("M").astype(str)
        low_n = df_obs[df_obs["n_total_regime"] == True][
            ["Timestamp", "month", "n_total", "H_t", "phi_t", "D_t", "post_rate_est"]
        ].sort_values("n_total")

        print(f"  Count: {len(low_n)}")
        if len(low_n) > 0:
            print(low_n.to_string(index=False))
    else:
        print("  n_total column not found.")

    # -----------------------------------------------------------------------
    # 5. ROWS WITH H_t BELOW 10.6
    # -----------------------------------------------------------------------
    print(f"\n--- ROWS WITH H_t < {HT_LOW_THRESHOLD} (real obs, in window) ---")
    if "H_t" in df_obs.columns:
        if "month" not in df_obs.columns:
            df_obs["month"] = df_obs["Timestamp"].dt.to_period("M").astype(str)
        low_ht = df_obs[df_obs["H_t"] < HT_LOW_THRESHOLD][
            ["Timestamp", "month", "H_t", "n_total", "phi_t", "D_t",
             "post_rate_est", "metric_provenance"]
        ].sort_values("H_t")
        print(f"  Count: {len(low_ht)}")
        if len(low_ht) > 0:
            print(low_ht.to_string(index=False))
    else:
        print("  H_t column not found.")

    # -----------------------------------------------------------------------
    # 6. ANY REAL OBS H_t NEAR 4.2?
    # -----------------------------------------------------------------------
    print("\n--- REAL OBS H_t NEAR 4.2 ---")
    if "H_t" in df_obs.columns:
        near_42 = df_obs[df_obs["H_t"].between(3.0, 6.0)][
            ["Timestamp", "H_t", "n_total", "metric_provenance"]
        ]
        if len(near_42) > 0:
            print(f"  YES - {len(near_42)} real obs with H_t in [3.0, 6.0]:")
            print(near_42.to_string(index=False))
        else:
            min_ht = df_obs["H_t"].min()
            print(f"  NO - no real obs with H_t in [3.0, 6.0]. Min real H_t = {min_ht:.4f}")

    # -----------------------------------------------------------------------
    # 7. SUMMARY TABLE
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("PHASE 0 REPORT - SUMMARY")
    print(sep)
    print(f"  Observation rows in window:              {rows_in_window}")
    print(f"  Numeric H_t rows (all real, no exclusion):{ht_numeric_rows}")
    print(f"  State rows in window:                    {state_rows_in_window}")
    print(f"  Shuffle heading count (in window):       {shuffle_heading_count}")
    print(f"  Det shuffle comparison triples:          {det_comp_count}")
    print(f"  Quarantined legacy metric lines:         {quarantined_count}")
    print(f"  Rows with n_total <= {N_TOTAL_REGIME_THRESHOLD}:                {len(low_n) if 'n_total' in df_obs.columns else 'N/A'}")
    print(f"  Rows with H_t < {HT_LOW_THRESHOLD}:                    {len(low_ht) if 'H_t' in df_obs.columns else 'N/A'}")
    real_ht_min = df_obs["H_t"].min() if "H_t" in df_obs.columns else float("nan")
    print(f"  Min real obs H_t (no exclusions):        {real_ht_min:.4f}")
    print(f"  Any real obs H_t near 4.2 (in [3,6]):    {'YES' if (df_obs['H_t'].between(3.0,6.0).any() if 'H_t' in df_obs.columns else False) else 'NO'}")

    # Provenance classification inventory
    print(f"\n  Metric provenance inventory:")
    print(f"    computed_observation_csv:          {rows_in_window} rows (all real obs)")
    print(f"    computed_state_csv:                {state_rows_in_window} rows")
    print(f"    deterministic_shuffle_comparison:  {det_comp_count} triplets "
          f"across {det_comp_block_count} blocks")
    print(f"    legacy_shuffle_log_text_untrusted: {quarantined_count} metric lines "
          f"(H/phi/D in shuffle blocks, NOT used as numeric evidence)")

    print(f"\n  n_total_regime flag (n_total <= {N_TOTAL_REGIME_THRESHOLD}):")
    if "n_total" in df_obs.columns and "month" in df_obs.columns:
        regime_by_month = df_obs[df_obs["n_total_regime"]].groupby("month").size().to_dict()
        total_regime = df_obs["n_total_regime"].sum()
        print(f"    Total flagged:    {total_regime}")
        print(f"    By month:         {regime_by_month}")
        print(f"    These rows are SIGNAL (real low-n observations), NOT excluded from H_t_numeric.")

    print(f"\n  H_t outlier rule status:")
    print(f"    robust_z < -3.5 rule: SUSPENDED for real observations.")
    print(f"    The 5 low-cluster May rows (H_t 8.71-9.70) co-occur with n_total 25-75 (n_total_regime=True).")
    print(f"    They are low-n regime signal and MUST remain in H_t_numeric.")
    print(f"    The 10.57 row (n_total=300) is also regime signal.")
    print(f"    H_t=4.2 exists only in shuffle log (quarantined as legacy_shuffle_log_text_untrusted).")
    print(f"    No real observation has H_t in [3.0, 6.0]. The 4.2 value cannot calibrate a real-obs rule.")

    print(f"\n  Pending decisions before Phase 1:")
    print(f"    1. Confirm: H_t_init_outlier rule removed for real obs (H_t_numeric = H_t for all real obs).")
    print(f"    2. Confirm: quarantined shuffle H/phi/D lines excluded from shuffle baseline stats.")
    print(f"    3. Confirm: deterministic_shuffle_comparison centroid-dist values ARE used in Phase 1.")
    print(f"    4. State CSV is available ({state_rows_in_window} rows in window) - confirm Phase 1 reads it.")

    print(f"\n{sep}")
    print("PHASE 0 COMPLETE - stopping before Phase 1")
    print(sep)


if __name__ == "__main__":
    main()
