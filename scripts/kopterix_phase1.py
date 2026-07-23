"""
kopterix_phase1.py  --  Kopterix instrument-calibration pipeline, Phase 1.

Settled Phase 0 decisions applied here:
  - H_t_init_outlier exclusion rule SUSPENDED for real observations.
    H_t_numeric = H_t for all real obs (no rows excluded).
    H_t_init_outlier column kept (all False) for schema continuity.
  - n_total_regime flag added: n_total <= 300.
  - Shuffle frame built from deterministic comparison lines ONLY.
    Log-text H(t)/phi(t)/D(t) lines are quarantined as
    legacy_shuffle_log_text_untrusted and excluded from baseline stats.
  - H_t = 4.2 (April first-run shuffle block) is retained as an
    instrument/logging-provenance anomaly for transparency and
    instrument characterization. It is NOT evidence that the feed had
    real entropy H_t = 4.2. It is excluded from shuffle baseline
    statistics and is NOT used as an exclusion-rule anchor.
  - metric_provenance column added to all frames.
  - Phase 1 does NOT read the state table and does NOT write a report.

Conventions: hyphens only; three-state booleans preserved;
outputs as CSV and PNG only.
"""

import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not available - Mann-Whitney U, KS, linregress skipped")

try:
    import seaborn as sns
    SEABORN_OK = True
except ImportError:
    SEABORN_OK = False
    print("[WARN] seaborn not available - heatmap uses matplotlib fallback")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and constants
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
INTER_DIR  = BASE_DIR / "analysis_intermediate"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"

for _d in [INTER_DIR, TABLES_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2026-04-01 00:00:00")
WINDOW_END   = pd.Timestamp("2026-06-01 00:00:00")

N_TOTAL_REGIME_THRESHOLD = 300   # n_total <= this -> n_total_regime = True

ALL_METRICS = [
    "post_rate_est",
    "H_t",
    "phi_t",
    "D_t",
    "n_total",
    "centroid_dist_surface_mid",
    "centroid_dist_mid_residue",
    "centroid_dist_surface_residue",
]

# Metrics for which trusted shuffle baseline values exist
# (deterministic comparison lines). H_t/phi_t/D_t are quarantined.
SHUFFLE_COMPARABLE = [
    "centroid_dist_surface_mid",
    "centroid_dist_mid_residue",
    "centroid_dist_surface_residue",
]

# Full comparable set for real-vs-shuffle table; H_t/phi_t/D_t will
# show n_shuffle=0 with a provenance note.
COMPARABLE_METRICS = [
    "H_t", "phi_t", "D_t",
    "centroid_dist_surface_mid",
    "centroid_dist_mid_residue",
    "centroid_dist_surface_residue",
]

WARNINGS_LIST = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def warn(msg):
    WARNINGS_LIST.append(msg)
    print(f"[WARN] {msg}")


def normalize_ts(series):
    """Parse to tz-naive UTC. Strips trailing ' UTC' suffix."""
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


def robust_z(series):
    arr = np.array(series, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 4:
        return np.full(len(arr), np.nan)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad == 0:
        return np.full(len(arr), np.nan)
    return 0.6745 * (arr - med) / mad


def safe_autocorr(series, lag):
    s = pd.Series(series, dtype=float).dropna()
    if len(s) <= lag + 1 or s.std() == 0:
        return np.nan
    return float(s.autocorr(lag=lag))


def safe_linregress(y_vals):
    y = np.array(y_vals, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return dict(slope=np.nan, intercept=np.nan, r_squared=np.nan, trend_p=np.nan)
    x = np.arange(len(y), dtype=float)
    if SCIPY_OK:
        res = sp_stats.linregress(x, y)
        return dict(slope=res.slope, intercept=res.intercept,
                    r_squared=res.rvalue ** 2, trend_p=res.pvalue)
    coeffs = np.polyfit(x, y, 1)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(slope=coeffs[0], intercept=coeffs[1], r_squared=r2, trend_p=np.nan)


def cohen_d(a, b):
    a = np.array(a, dtype=float); a = a[np.isfinite(a)]
    b = np.array(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pv = ((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)
    return (a.mean() - b.mean()) / np.sqrt(pv) if pv > 0 else np.nan


def mw_pval(a, b):
    if not SCIPY_OK:
        return np.nan
    a = np.array(a, dtype=float); a = a[np.isfinite(a)]
    b = np.array(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        _, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def ks_pval(a, b):
    if not SCIPY_OK:
        return np.nan
    a = np.array(a, dtype=float); a = a[np.isfinite(a)]
    b = np.array(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        _, p = sp_stats.ks_2samp(a, b)
        return float(p)
    except Exception:
        return np.nan


def empirical_pct(val, dist):
    d = np.array(dist, dtype=float); d = d[np.isfinite(d)]
    if len(d) == 0 or not np.isfinite(val):
        return np.nan
    return float(np.mean(d <= val))


def pearson_r(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) != len(b):
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman_r(a, b):
    if not SCIPY_OK:
        return np.nan
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) != len(b):
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return np.nan
    try:
        r, _ = sp_stats.spearmanr(a, b)
        return float(r)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Provenance-aware shuffle log parser
# ---------------------------------------------------------------------------

SHUFFLE_HEADER_RE = re.compile(
    r"^(?:##\s+)?BASELINE-SHUFFLE obs ::\s*([\d-]+\s+[\d:]+)\s*UTC",
    re.IGNORECASE,
)

# Deterministic comparison: surface-mid real=... | shuffle=... | delta=...
DET_COMP_RE = re.compile(
    r"(surface-mid|mid-residue|surface-residue)\s+"
    r"real=([\d.eE+-]+)\s*\|\s*shuffle=([\d.eE+-]+)\s*\|\s*delta=([-\d.eE+]+)",
    re.IGNORECASE,
)

METRICS_LINE_RE  = re.compile(r"\*\*metrics:\*\*", re.IGNORECASE)
AWAITING_RE      = re.compile(r"\[awaiting external\]", re.IGNORECASE)
FIRST_RUN_RE     = re.compile(r"first run", re.IGNORECASE)
ALL_NA_RE        = re.compile(r"H\(t\)\s*=\s*N/A", re.IGNORECASE)
HT_METRICS_RE    = re.compile(r"H\(t\)\s*=\s*([\d.eE+-]+)", re.IGNORECASE)
PHI_METRICS_RE   = re.compile(r"ph?i\(t\)\s*=\s*([\d.eE+-]+)", re.IGNORECASE)
DT_METRICS_RE    = re.compile(r"D\(t\)\s*=\s*([\d.eE+-]+)", re.IGNORECASE)

PAIR_TO_COL = {
    "surface-mid":     "centroid_dist_surface_mid",
    "mid-residue":     "centroid_dist_mid_residue",
    "surface-residue": "centroid_dist_surface_residue",
}


def _untrust_reason(line):
    if AWAITING_RE.search(line):    return "awaiting_external"
    if FIRST_RUN_RE.search(line):   return "first_run_artifact"
    if ALL_NA_RE.search(line):      return "all_fields_na"
    return "provenance_unconfirmed_carried_value"


def parse_shuffle_log(path):
    """
    Returns:
      det_rows        - list of dicts, one per deterministic comparison line
                        (metric_provenance = deterministic_shuffle_comparison)
      quarantine_rows - list of dicts, one per H/phi/D metrics line
                        (metric_provenance = legacy_shuffle_log_text_untrusted)
      instrument_anomalies - list of dicts for explicitly flagged anomalies
      blocks_found    - int
      header_sample   - str
    """
    text  = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_sample = lines[0] if lines else ""

    det_rows, quarantine_rows, instrument_anomalies = [], [], []
    blocks_found = 0
    i = 0

    while i < len(lines):
        m = SHUFFLE_HEADER_RE.match(lines[i].strip())
        if m:
            blocks_found += 1
            ts_str = m.group(1).strip()
            ts = normalize_ts(pd.Series([ts_str])).iloc[0]

            j = i + 1
            block_lines = []
            while j < len(lines) and not SHUFFLE_HEADER_RE.match(lines[j].strip()):
                block_lines.append(lines[j])
                j += 1

            seen_det = False
            for bl in block_lines:
                dc = DET_COMP_RE.search(bl)
                if dc:
                    seen_det = True
                    det_rows.append({
                        "Timestamp":           ts,
                        "source_log":          path.name,
                        "pair":                dc.group(1).lower(),
                        "col":                 PAIR_TO_COL[dc.group(1).lower()],
                        "real_value":          float(dc.group(2)),
                        "shuffle_value":       float(dc.group(3)),
                        "delta":               float(dc.group(4)),
                        "metric_provenance":   "deterministic_shuffle_comparison",
                    })
                    continue

                if METRICS_LINE_RE.search(bl):
                    reason   = _untrust_reason(bl)
                    position = "post_comparison" if seen_det else "pre_comparison_header"
                    ht_m  = HT_METRICS_RE.search(bl)
                    phi_m = PHI_METRICS_RE.search(bl)
                    dt_m  = DT_METRICS_RE.search(bl)
                    ht_val = float(ht_m.group(1)) if ht_m else None

                    q_row = {
                        "Timestamp":          ts,
                        "source_log":         path.name,
                        "position_in_block":  position,
                        "untrust_reason":     reason,
                        "H_t_raw":            ht_val,
                        "phi_t_raw":          float(phi_m.group(1)) if phi_m else None,
                        "D_t_raw":            float(dt_m.group(1))  if dt_m  else None,
                        "metric_provenance":  "legacy_shuffle_log_text_untrusted",
                    }
                    quarantine_rows.append(q_row)

                    if reason == "first_run_artifact" and ht_val is not None:
                        instrument_anomalies.append({
                            "metric":          "H_t",
                            "value":           ht_val,
                            "block_ts":        str(ts),
                            "source_log":      path.name,
                            "position":        position,
                            "untrust_reason":  reason,
                            "metric_provenance": "legacy_shuffle_log_text_untrusted",
                            "description": (
                                f"H_t = {ht_val} appears in the April first-run shuffle record "
                                f"({path.name}, block {ts_str} UTC) as a pre-comparison header "
                                f"value. It is an instrument/logging-provenance anomaly from "
                                f"the April first-run shuffle record, not evidence that the "
                                f"feed itself had real entropy H_t = {ht_val}. Retained for "
                                f"transparency and instrument characterization. Excluded from "
                                f"shuffle baseline statistics and not used as an "
                                f"exclusion-rule anchor."
                            ),
                        })

            i = j
        else:
            i += 1

    return det_rows, quarantine_rows, instrument_anomalies, blocks_found, header_sample


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("KOPTERIX PHASE 1 - Instrument Calibration Pipeline")
    print(f"Window: [{WINDOW_START}  -->  {WINDOW_END})")
    print(f"scipy available  : {SCIPY_OK}")
    print(f"seaborn available: {SEABORN_OK}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # STEP 1: LOAD OBSERVATIONS
    # -----------------------------------------------------------------------
    print("\n[STEP 1] Loading observations CSV ...")
    obs_path = BASE_DIR / "observations_April-May_2026.csv"
    if not obs_path.exists():
        fallback = BASE_DIR / "observations.csv"
        if fallback.exists():
            warn("Expected 'observations_April-May_2026.csv' not found; "
                 "falling back to 'observations.csv'. Verify this is the correct file.")
            obs_path = fallback
        else:
            print("[ERROR] No observations CSV found."); sys.exit(1)

    raw_df = pd.read_csv(obs_path, low_memory=False)
    utc_col, ts_col = _find_utc_col(raw_df)

    if utc_col and ts_col:
        warn(f"Both '{utc_col}' and '{ts_col}' found. Dropping '{ts_col}' "
             f"(assumed local Chicago time); keying on '{utc_col}'.")
        raw_df = raw_df.drop(columns=[ts_col])
        time_col = utc_col
    elif utc_col:
        time_col = utc_col
    elif ts_col:
        warn(f"No Timestamp_UTC column found; using '{ts_col}'. "
             "Must be verified as UTC.")
        time_col = ts_col
    else:
        print("[ERROR] No timestamp column found."); sys.exit(1)

    phase0_obs_raw_samples = [str(v) for v in raw_df[time_col].head(3).tolist()]

    raw_df["Timestamp"] = normalize_ts(raw_df[time_col])
    rows_in            = len(raw_df)
    dropped_unparseable = int(raw_df["Timestamp"].isna().sum())
    df_ok = raw_df[raw_df["Timestamp"].notna()].copy()

    before_mask    = df_ok["Timestamp"] < WINDOW_START
    after_mask     = df_ok["Timestamp"] >= WINDOW_END
    dropped_before = int(before_mask.sum())
    dropped_after  = int(after_mask.sum())
    df_obs = df_ok[~before_mask & ~after_mask].copy()
    rows_after_filter = len(df_obs)

    for m in ALL_METRICS:
        if m in df_obs.columns:
            df_obs[m] = pd.to_numeric(df_obs[m], errors="coerce")

    df_obs = df_obs.sort_values("Timestamp").reset_index(drop=True)
    df_obs["is_shuffle"] = False
    df_obs["month"]      = df_obs["Timestamp"].dt.to_period("M").astype(str)
    df_obs["metric_provenance"] = "computed_observation_csv"

    # H_t_init_outlier rule SUSPENDED for real observations.
    # Column kept (all False) for schema continuity.
    df_obs["H_t_init_outlier"] = False
    df_obs["H_t_numeric"]      = df_obs["H_t"].copy() if "H_t" in df_obs.columns else np.nan

    # n_total_regime flag: real low-n observations, NOT excluded.
    if "n_total" in df_obs.columns:
        df_obs["n_total_regime"] = df_obs["n_total"].le(N_TOTAL_REGIME_THRESHOLD) & df_obs["n_total"].notna()
    else:
        df_obs["n_total_regime"] = False

    regime_rows = df_obs[df_obs["n_total_regime"]]
    if len(regime_rows) > 0:
        print(f"  n_total_regime rows (n_total <= {N_TOTAL_REGIME_THRESHOLD}): "
              f"{len(regime_rows)}, by month: "
              f"{regime_rows.groupby('month').size().to_dict()}")

    rows_per_month = df_obs.groupby("month").size().to_dict()
    ts_min = df_obs["Timestamp"].min()
    ts_max = df_obs["Timestamp"].max()

    print(f"  rows_in={rows_in}, rows_after_filter={rows_after_filter}")
    print(f"  dropped: before={dropped_before}, after={dropped_after}, "
          f"unparseable={dropped_unparseable}")
    print(f"  Surviving: {ts_min} --> {ts_max}")
    print(f"  Rows per month: {rows_per_month}")

    df_obs.to_csv(INTER_DIR / "real_clean.csv", index=False)

    # -----------------------------------------------------------------------
    # DATA COVERAGE
    # -----------------------------------------------------------------------
    print("\n[COV] Computing data_coverage.csv ...")
    days_in_range = len(pd.date_range(WINDOW_START, WINDOW_END - pd.Timedelta(days=1), freq="D"))
    ts_sorted  = df_obs["Timestamp"].sort_values()
    dup_ts     = int(ts_sorted.duplicated().sum())
    gaps_h     = ts_sorted.diff().dt.total_seconds().dropna() / 3600.0
    median_gap = float(gaps_h.median()) if len(gaps_h) > 0 else np.nan
    max_gap    = float(gaps_h.max())    if len(gaps_h) > 0 else np.nan
    large_gap_count = int((gaps_h > 3 * median_gap).sum()) if np.isfinite(median_gap) else 0
    day_counts = df_obs.groupby(df_obs["Timestamp"].dt.date).size()
    days_with_zero = days_in_range - int((day_counts > 0).sum())
    rpd_mean = float(day_counts.mean()) if len(day_counts) > 0 else np.nan
    rpd_min  = float(day_counts.min())  if len(day_counts) > 0 else np.nan
    rpd_max  = float(day_counts.max())  if len(day_counts) > 0 else np.nan

    cov = {
        "row_count": rows_after_filter,
        "date_range_start": str(ts_min), "date_range_end": str(ts_max),
        "duplicate_timestamps": dup_ts,
        "median_gap_hours": median_gap, "max_gap_hours": max_gap,
        "large_gap_count_3x_median": large_gap_count,
        "days_with_zero_runs": days_with_zero, "days_in_range": days_in_range,
        "runs_per_day_mean": rpd_mean, "runs_per_day_min": rpd_min,
        "runs_per_day_max": rpd_max,
    }
    for m in ALL_METRICS:
        cov[f"nan_count_{m}"]     = int(df_obs[m].isna().sum()) if m in df_obs.columns else -1
        cov[f"malformed_neg_{m}"] = int((df_obs[m] < 0).sum()) if m in df_obs.columns else -1
    pd.DataFrame([cov]).to_csv(TABLES_DIR / "data_coverage.csv", index=False)
    print("  Saved tables/data_coverage.csv")

    # -----------------------------------------------------------------------
    # STEP 2: PARSE SHUFFLE LOGS
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Parsing shuffle logs ...")
    log_paths = {
        "2026-04": BASE_DIR / "kopterix_log_2026-04.md",
        "2026-05": BASE_DIR / "kopterix_log_2026-05.md",
    }
    present_logs = {k: v for k, v in log_paths.items() if v.exists()}
    absent_logs  = [k for k in log_paths if k not in present_logs]
    if absent_logs:
        warn(f"Shuffle log(s) absent and skipped: {absent_logs}")

    phase0_log_headers  = {}
    blocks_found_per_log = {}
    all_det_rows        = []
    all_quarantine_rows = []
    all_instrument_anomalies = []

    if not present_logs:
        parse_failure = None
        print("  No log files present.")
    else:
        parse_failure = False
        for month_key, lpath in present_logs.items():
            det_rows, q_rows, anomalies, n_blocks, hdr = parse_shuffle_log(lpath)
            phase0_log_headers[lpath.name]   = hdr
            blocks_found_per_log[lpath.name] = n_blocks
            print(f"  {lpath.name}: {n_blocks} blocks, "
                  f"{len(det_rows)} det-comp lines, "
                  f"{len(q_rows)} quarantined metric lines")
            for r in det_rows:
                r["month"] = month_key
            all_det_rows.extend(det_rows)
            all_quarantine_rows.extend(q_rows)
            all_instrument_anomalies.extend(anomalies)
        if not all_det_rows:
            parse_failure = True

    # Print instrument anomalies
    if all_instrument_anomalies:
        print(f"  Instrument anomalies flagged: {len(all_instrument_anomalies)}")
        for a in all_instrument_anomalies:
            print(f"    [{a['untrust_reason']}] {a['metric']}={a['value']} "
                  f"@ {a['block_ts']}  ({a['source_log']})")

    # Save quarantine CSV
    if all_quarantine_rows:
        pd.DataFrame(all_quarantine_rows).to_csv(
            INTER_DIR / "quarantine_shuffle_metrics.csv", index=False)
        print(f"  Saved analysis_intermediate/quarantine_shuffle_metrics.csv "
              f"({len(all_quarantine_rows)} rows)")

    # Build df_shuf: one row per (block_ts, source_log) with centroid shuffle values
    # from deterministic comparison lines. H_t/phi_t/D_t = NaN (no trusted values).
    if all_det_rows:
        det_df = pd.DataFrame(all_det_rows)

        # Apply window filter
        det_in = det_df[
            det_df["Timestamp"].notna() &
            (det_df["Timestamp"] >= WINDOW_START) &
            (det_df["Timestamp"] < WINDOW_END)
        ].copy()

        if len(det_in) > 0:
            # Pivot to wide: one row per block, columns = centroid distances
            pivot = (
                det_in
                .pivot_table(index="Timestamp", columns="col",
                             values="shuffle_value", aggfunc="first")
                .reset_index()
            )
            pivot.columns.name = None

            # Merge in source_log and month
            block_meta = (
                det_in[["Timestamp", "source_log", "month"]]
                .drop_duplicates("Timestamp")
            )
            df_shuf = pivot.merge(block_meta, on="Timestamp", how="left")

            # Add quarantined metric columns as NaN - no trusted shuffle H/phi/D
            for m in ["H_t", "phi_t", "D_t", "post_rate_est", "n_total"]:
                df_shuf[m] = np.nan

            df_shuf["is_shuffle"]          = True
            df_shuf["metric_provenance"]   = "deterministic_shuffle_comparison"
            df_shuf["H_t_init_outlier"]    = False
            df_shuf["H_t_numeric"]         = np.nan   # no trusted shuffle H_t
            df_shuf["n_total_regime"]      = False

            # Sort, dedup on Timestamp
            df_shuf = (df_shuf
                       .sort_values("Timestamp")
                       .drop_duplicates(subset=["Timestamp"], keep="first")
                       .reset_index(drop=True))

            shuffle_count_per_month = df_shuf.groupby("month").size().to_dict()
            rows_kept_shuf = len(df_shuf)

            print(f"  Shuffle frame: {rows_kept_shuf} rows "
                  f"(one per block with det-comp data)")
            print(f"  Shuffle blocks per month: {shuffle_count_per_month}")

            # Save shuffle_clean.csv
            save_cols = (["Timestamp"] + ALL_METRICS +
                         ["is_shuffle", "source_log", "month",
                          "metric_provenance", "H_t_init_outlier",
                          "H_t_numeric", "n_total_regime"])
            for c in save_cols:
                if c not in df_shuf.columns:
                    df_shuf[c] = np.nan
            df_shuf[save_cols].to_csv(INTER_DIR / "shuffle_clean.csv", index=False)
            print("  Saved analysis_intermediate/shuffle_clean.csv")

            total_shuffle_n = rows_kept_shuf
            phase0_shuf_stats = {
                "blocks_found_per_log":     blocks_found_per_log,
                "det_comp_rows_in_window":  len(det_in),
                "shuffle_blocks_per_month": shuffle_count_per_month,
                "quarantined_metric_lines": len(all_quarantine_rows),
                "instrument_anomalies":     len(all_instrument_anomalies),
            }
        else:
            df_shuf = pd.DataFrame()
            total_shuffle_n = 0
            rows_kept_shuf  = 0
            shuffle_count_per_month = {}
            parse_failure   = True
            phase0_shuf_stats = {
                "blocks_found_per_log": blocks_found_per_log,
                "det_comp_rows_in_window": 0,
                "shuffle_blocks_per_month": {},
                "quarantined_metric_lines": len(all_quarantine_rows),
                "instrument_anomalies": len(all_instrument_anomalies),
            }
    else:
        df_shuf = pd.DataFrame()
        total_shuffle_n = 0
        rows_kept_shuf  = 0
        shuffle_count_per_month = {}
        phase0_shuf_stats = {
            "blocks_found_per_log": blocks_found_per_log,
            "det_comp_rows_in_window": 0,
            "shuffle_blocks_per_month": {},
            "quarantined_metric_lines": len(all_quarantine_rows),
            "instrument_anomalies": len(all_instrument_anomalies),
        }

    # Three-state parse_failure for status JSON
    if parse_failure is None:
        shuffle_parsing_success = None
    elif parse_failure:
        shuffle_parsing_success = False
    else:
        shuffle_parsing_success = total_shuffle_n > 0

    # -----------------------------------------------------------------------
    # PHASE 0 SUMMARY
    # -----------------------------------------------------------------------
    print("\n[PHASE 0] Summary")
    print(f"  Obs raw ts samples (first 3):  {phase0_obs_raw_samples}")
    print(f"  Log header samples:            {phase0_log_headers}")
    print(f"  Obs rows_in={rows_in}, rows_after_filter={rows_after_filter}")
    print(f"  dropped_before={dropped_before}, dropped_after={dropped_after}, "
          f"dropped_unparseable={dropped_unparseable}")
    print(f"  surviving_min={ts_min}, surviving_max={ts_max}")
    print(f"  rows_per_month={rows_per_month}")
    print(f"  Shuffle: {phase0_shuf_stats}")

    # -----------------------------------------------------------------------
    # STEP 3: DESCRIPTIVE STATS (real, H_t via H_t_numeric)
    # -----------------------------------------------------------------------
    print("\n[STEP 3] Descriptive stats ...")
    desc_rows = []
    for m in ALL_METRICS:
        col = "H_t_numeric" if m == "H_t" else m
        if col not in df_obs.columns:
            continue
        s  = df_obs[col].dropna()
        rz = robust_z(df_obs[col])
        rz_f = rz[np.isfinite(rz)]
        desc_rows.append({
            "metric":       m,
            "count":        len(s),
            "nan_count":    int(df_obs[col].isna().sum()),
            "mean":         s.mean()           if len(s) > 0 else np.nan,
            "median":       s.median()         if len(s) > 0 else np.nan,
            "std":          s.std()            if len(s) > 0 else np.nan,
            "min":          s.min()            if len(s) > 0 else np.nan,
            "max":          s.max()            if len(s) > 0 else np.nan,
            "q05":          s.quantile(0.05)   if len(s) > 0 else np.nan,
            "q25":          s.quantile(0.25)   if len(s) > 0 else np.nan,
            "q75":          s.quantile(0.75)   if len(s) > 0 else np.nan,
            "q95":          s.quantile(0.95)   if len(s) > 0 else np.nan,
            "cv":           s.std() / s.mean() if (len(s) > 0 and s.mean() != 0) else np.nan,
            "mad":          float(np.median(np.abs(s - s.median()))) if len(s) > 0 else np.nan,
            "robust_z_min": float(rz_f.min()) if len(rz_f) > 0 else np.nan,
            "robust_z_max": float(rz_f.max()) if len(rz_f) > 0 else np.nan,
            "n_total_regime_count": int(df_obs["n_total_regime"].sum()) if m == "H_t" else np.nan,
        })
    pd.DataFrame(desc_rows).to_csv(TABLES_DIR / "real_descriptive_stats.csv", index=False)
    print("  Saved tables/real_descriptive_stats.csv")

    # -----------------------------------------------------------------------
    # STEP 4: SHUFFLE BASELINE STATS
    # (centroid distances only - H_t/phi_t/D_t have no trusted shuffle values)
    # -----------------------------------------------------------------------
    print("\n[STEP 4] Shuffle baseline stats ...")
    shuf_stat_rows = []
    for m in COMPARABLE_METRICS:
        col = "H_t_numeric" if m == "H_t" else m
        if not df_shuf.empty and col in df_shuf.columns:
            s = df_shuf[col].dropna()
        else:
            s = pd.Series([], dtype=float)
        note = ""
        if m in ["H_t", "phi_t", "D_t"]:
            note = ("no trusted shuffle values - log-text H/phi/D lines are "
                    "legacy_shuffle_log_text_untrusted")
        shuf_stat_rows.append({
            "metric":    m,
            "n_shuffle": len(s),
            "mean":      s.mean() if len(s) > 0 else np.nan,
            "std":       s.std()  if len(s) > 0 else np.nan,
            "min":       s.min()  if len(s) > 0 else np.nan,
            "max":       s.max()  if len(s) > 0 else np.nan,
            "provenance_note": note,
        })
    pd.DataFrame(shuf_stat_rows).to_csv(TABLES_DIR / "shuffle_baseline_stats.csv", index=False)
    print("  Saved tables/shuffle_baseline_stats.csv")

    # -----------------------------------------------------------------------
    # STEP 5: REAL VS SHUFFLE
    # -----------------------------------------------------------------------
    print("\n[STEP 5] Real vs shuffle ...")
    rvs_rows = []
    for m in COMPARABLE_METRICS:
        col = "H_t_numeric" if m == "H_t" else m
        rv = df_obs[col].dropna().values if col in df_obs.columns else np.array([])
        sv = (df_shuf[col].dropna().values
              if (not df_shuf.empty and col in df_shuf.columns)
              else np.array([]))
        note = ""
        if m in ["H_t", "phi_t", "D_t"] and len(sv) == 0:
            note = "no trusted shuffle values (legacy_shuffle_log_text_untrusted)"
        rvs_rows.append({
            "metric":       m,
            "n_real":       len(rv),
            "n_shuffle":    len(sv),
            "mean_real":    rv.mean() if len(rv) > 0 else np.nan,
            "mean_shuffle": sv.mean() if len(sv) > 0 else np.nan,
            "std_real":     rv.std()  if len(rv) > 0 else np.nan,
            "std_shuffle":  sv.std()  if len(sv) > 0 else np.nan,
            "mean_diff":    (rv.mean() - sv.mean())
                            if (len(rv) > 0 and len(sv) > 0) else np.nan,
            "cohens_d":     cohen_d(rv, sv),
            "mw_pval":      mw_pval(rv, sv),
            "empirical_pct_real_in_shuffle": empirical_pct(
                rv.mean() if len(rv) > 0 else np.nan, sv),
            "provenance_note": note,
        })
    pd.DataFrame(rvs_rows).to_csv(TABLES_DIR / "real_vs_shuffle.csv", index=False)
    print("  Saved tables/real_vs_shuffle.csv")

    # -----------------------------------------------------------------------
    # STEP 6: TEMPORAL STRUCTURE
    # -----------------------------------------------------------------------
    print("\n[STEP 6] Temporal structure ...")
    temp_rows = []
    for m in ALL_METRICS:
        col = "H_t_numeric" if m == "H_t" else m
        if col not in df_obs.columns:
            continue
        s   = df_obs[col]
        ac1 = safe_autocorr(s, 1)
        ac4 = safe_autocorr(s, 4)
        rm  = s.rolling(8, min_periods=4).mean()
        rs  = s.rolling(8, min_periods=4).std()
        d1  = s.diff()
        ols = safe_linregress(s.values)
        temp_rows.append({
            "metric":             m,
            "autocorr_lag1":      ac1,
            "autocorr_lag4":      ac4,
            "roll_mean_last":     float(rm.dropna().iloc[-1]) if rm.dropna().size > 0 else np.nan,
            "roll_std_last":      float(rs.dropna().iloc[-1]) if rs.dropna().size > 0 else np.nan,
            "diff1_mean":         float(d1.mean()) if d1.notna().any() else np.nan,
            "diff1_std":          float(d1.std())  if d1.notna().any() else np.nan,
            "ols_slope":          ols["slope"],
            "ols_intercept":      ols["intercept"],
            "ols_r_squared":      ols["r_squared"],
            "ols_trend_p":        ols["trend_p"],
            "flag_autocorr_lag1": bool(abs(ac1) > 0.3) if np.isfinite(ac1) else False,
            "flag_autocorr_lag4": bool(abs(ac4) > 0.3) if np.isfinite(ac4) else False,
            "flag_trend_p":       bool(ols["trend_p"] < 0.05)
                                  if np.isfinite(ols.get("trend_p", np.nan)) else False,
        })
    pd.DataFrame(temp_rows).to_csv(TABLES_DIR / "temporal_structure.csv", index=False)
    print("  Saved tables/temporal_structure.csv")

    # -----------------------------------------------------------------------
    # STEP 7: CORRELATIONS
    # -----------------------------------------------------------------------
    print("\n[STEP 7] Correlations ...")
    corr_df = df_obs.copy()
    if "H_t" in corr_df.columns:
        corr_df["H_t"] = corr_df["H_t_numeric"]

    metric_cols = [m for m in ALL_METRICS if m in corr_df.columns]
    n_m = len(metric_cols)
    pearson_mat  = np.full((n_m, n_m), np.nan)
    spearman_mat = np.full((n_m, n_m), np.nan)

    for i, mi in enumerate(metric_cols):
        for j, mj in enumerate(metric_cols):
            pearson_mat[i, j]  = pearson_r(corr_df[mi].values, corr_df[mj].values)
            spearman_mat[i, j] = spearman_r(corr_df[mi].values, corr_df[mj].values)

    pd.DataFrame(pearson_mat, index=metric_cols, columns=metric_cols
                 ).to_csv(TABLES_DIR / "pearson_correlation.csv")
    pd.DataFrame(spearman_mat, index=metric_cols, columns=metric_cols
                 ).to_csv(TABLES_DIR / "spearman_correlation.csv")
    print("  Saved tables/pearson_correlation.csv, spearman_correlation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in [
        (axes[0], pearson_mat,  "Pearson Correlation - April-May 2026"),
        (axes[1], spearman_mat, "Spearman Correlation - April-May 2026"),
    ]:
        if SEABORN_OK:
            sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=metric_cols,
                        yticklabels=metric_cols, ax=ax, vmin=-1, vmax=1,
                        cmap="coolwarm")
        else:
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
            ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(metric_cols, fontsize=7)
            for ii in range(n_m):
                for jj in range(n_m):
                    v = mat[ii, jj]
                    ax.text(jj, ii, f"{v:.2f}" if np.isfinite(v) else "nan",
                            ha="center", va="center", fontsize=6)
            plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=100)
    plt.close()
    print("  Saved figures/correlation_heatmap.png")

    high_corr_pairs = []
    for i in range(n_m):
        for j in range(i + 1, n_m):
            r = pearson_mat[i, j]
            if np.isfinite(r) and abs(r) > 0.5:
                high_corr_pairs.append({
                    "metric_a":  metric_cols[i],
                    "metric_b":  metric_cols[j],
                    "pearson_r": round(float(r), 4),
                })

    # -----------------------------------------------------------------------
    # STEP 8: DIAGNOSTIC FLAGS
    # -----------------------------------------------------------------------
    print("\n[STEP 8] Diagnostic flags ...")
    n_total_vals   = corr_df["n_total"].values    if "n_total"       in corr_df.columns else np.array([])
    post_rate_vals = corr_df["post_rate_est"].values if "post_rate_est" in corr_df.columns else np.array([])

    diag_rows = []
    for m in ALL_METRICS:
        col = "H_t_numeric" if m == "H_t" else m
        if col not in corr_df.columns:
            continue
        s  = corr_df[col]
        fn = s.dropna().values
        n  = len(fn)
        cv_val = s.std() / s.mean() if (n > 0 and s.mean() != 0) else np.nan
        rz_arr = robust_z(s)
        outlier_count = int(np.sum(np.abs(rz_arr) > 3.5)) if np.isfinite(rz_arr).any() else 0
        r_n    = pearson_r(s.values, n_total_vals)   if len(n_total_vals)   == len(s) else np.nan
        r_post = pearson_r(s.values, post_rate_vals) if len(post_rate_vals) == len(s) else np.nan
        diag_rows.append({
            "metric":                       m,
            "low_variance_flag":            bool(abs(cv_val) < 0.01)  if np.isfinite(cv_val) else False,
            "high_variance_flag":           bool(abs(cv_val) > 2.0)   if np.isfinite(cv_val) else False,
            "ceiling_floor_concentration":  float((s == s.max()).sum() / max(n, 1)),
            "exact_zero_count":             int((s == 0).sum()),
            "impossible_neg_count":         int((s < 0).sum()) if m != "phi_t" else 0,
            "phi_out_of_01":                int(((s < 0) | (s > 1)).sum()) if m == "phi_t" else 0,
            "robust_z_outlier_count":       outlier_count,
            "n_total_lt50_count":           int((s < 50).sum()) if m == "n_total" else 0,
            "pearson_r_with_n_total":       r_n,
            "depends_on_n_total":           bool(abs(r_n)    > 0.5) if np.isfinite(r_n)    else False,
            "pearson_r_with_post_rate_est": r_post,
            "depends_on_post_rate_est":     bool(abs(r_post) > 0.5) if np.isfinite(r_post) else False,
        })
    pd.DataFrame(diag_rows).to_csv(TABLES_DIR / "diagnostic_flags.csv", index=False)
    print("  Saved tables/diagnostic_flags.csv")

    # -----------------------------------------------------------------------
    # STEP 8b: APRIL-VS-MAY STABILITY
    # -----------------------------------------------------------------------
    print("\n[STEP 8b] April-vs-May stability ...")
    months_present = sorted(df_obs["month"].unique())
    cm_dist_rows, cm_autocorr_rows = [], []
    cross_month_skip_reason = None

    if len(months_present) < 2:
        cross_month_skip_reason = f"Fewer than two months present: {months_present}"
        warn(cross_month_skip_reason)
    else:
        month_pops = df_obs.groupby("month").size().sort_values(ascending=False)
        month_a, month_b = sorted(list(month_pops.index[:2]))
        print(f"  Comparing month_a={month_a} vs month_b={month_b}")
        df_a = df_obs[df_obs["month"] == month_a]
        df_b = df_obs[df_obs["month"] == month_b]

        for m in ALL_METRICS:
            col = "H_t_numeric" if m == "H_t" else m
            if col not in df_obs.columns:
                continue
            av = df_a[col].dropna().values
            bv = df_b[col].dropna().values
            d  = cohen_d(bv, av)
            cm_dist_rows.append({
                "metric":               m,
                "month_a":              month_a,
                "month_b":              month_b,
                "n_a":                  len(av),
                "n_b":                  len(bv),
                "mean_a":               av.mean() if len(av) > 0 else np.nan,
                "mean_b":               bv.mean() if len(bv) > 0 else np.nan,
                "mean_shift_b_minus_a": (bv.mean()-av.mean()) if (len(av)>0 and len(bv)>0) else np.nan,
                "cohens_d_b_vs_a":      d,
                "mw_pval":              mw_pval(bv, av),
                "ks_pval":              ks_pval(bv, av),
            })
            ac_a    = safe_autocorr(df_a[col], 1)
            ac_b    = safe_autocorr(df_b[col], 1)
            ac_diff = (ac_b - ac_a) if (np.isfinite(ac_a) and np.isfinite(ac_b)) else np.nan
            cm_autocorr_rows.append({
                "metric":                  m,
                "month_a":                 month_a,
                "month_b":                 month_b,
                "autocorr_lag1_a":         ac_a,
                "autocorr_lag1_b":         ac_b,
                "autocorr_diff_b_minus_a": ac_diff,
                "flag_large_autocorr_diff": bool(abs(ac_diff) > 0.3) if np.isfinite(ac_diff) else False,
                "cohens_d_b_vs_a":         d,
                "flag_large_d":            bool(abs(d) >= 0.5) if np.isfinite(d) else False,
            })

    pd.DataFrame(cm_dist_rows).to_csv(
        TABLES_DIR / "cross_month_distribution.csv", index=False)
    pd.DataFrame(cm_autocorr_rows).to_csv(
        TABLES_DIR / "cross_month_autocorr_persistence.csv", index=False)
    print("  Saved tables/cross_month_distribution.csv, "
          "cross_month_autocorr_persistence.csv")

    # -----------------------------------------------------------------------
    # STEP 9: FIGURES
    # -----------------------------------------------------------------------
    print("\n[STEP 9] Generating figures ...")
    saved_figures = []

    for m in ALL_METRICS:
        raw_col = m
        num_col = "H_t_numeric" if m == "H_t" else m
        if raw_col not in df_obs.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_obs["Timestamp"], df_obs[raw_col], "b.", markersize=3,
                alpha=0.6, label="real")

        # Highlight n_total_regime rows
        regime_mask = df_obs["n_total_regime"]
        if regime_mask.any():
            ax.scatter(df_obs.loc[regime_mask, "Timestamp"],
                       df_obs.loc[regime_mask, raw_col],
                       color="green", s=40, zorder=6, marker="^",
                       label=f"n_total_regime (n<={N_TOTAL_REGIME_THRESHOLD})")

        # Shuffle centroid-distance points where available
        if not df_shuf.empty and num_col in df_shuf.columns:
            sv = df_shuf[num_col]
            valid = sv.dropna()
            if len(valid) > 0:
                ax.scatter(df_shuf.loc[valid.index, "Timestamp"], valid,
                           color="orange", s=25, zorder=5, label="shuffle (det-comp)")
                if len(valid) >= 2:
                    ax.axhline(valid.mean(), color="darkorange", linestyle="--",
                               linewidth=1.2, label=f"shuffle mean={valid.mean():.4f}")
                    ax.axhline(valid.mean() + valid.std(), color="darkorange",
                               linestyle=":", linewidth=0.8)
                    ax.axhline(valid.mean() - valid.std(), color="darkorange",
                               linestyle=":", linewidth=0.8,
                               label="shuffle mean +/- std")

        ax.set_title(f"{m} - April-May 2026")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel(m)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        fname = f"timeseries_{m}.png"
        plt.savefig(FIGURES_DIR / fname, dpi=100)
        plt.close()
        saved_figures.append(fname)

    # Rolling summary
    roll_specs = [("H_t", "H_t_numeric"), ("phi_t", "phi_t"), ("D_t", "D_t")]
    fig, axes = plt.subplots(len(roll_specs), 1, figsize=(12, 9), sharex=True)
    for ax, (m, col) in zip(axes, roll_specs):
        if col not in df_obs.columns:
            ax.set_title(f"{m} - not available"); continue
        s  = df_obs[col]
        rm = s.rolling(8, min_periods=4).mean()
        rs = s.rolling(8, min_periods=4).std()
        ax.plot(df_obs["Timestamp"], s, "b.", markersize=2, alpha=0.4, label="raw")
        regime_mask = df_obs["n_total_regime"]
        if regime_mask.any():
            ax.scatter(df_obs.loc[regime_mask, "Timestamp"],
                       s[regime_mask], color="green", s=30, zorder=6,
                       marker="^", label=f"n_total_regime")
        ax.plot(df_obs["Timestamp"], rm, "r-", linewidth=1.5, label="roll mean (w=8)")
        ax.fill_between(df_obs["Timestamp"], rm - rs, rm + rs,
                        alpha=0.2, color="red", label="roll mean +/- std")
        ax.set_ylabel(m)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{m} rolling metrics - April-May 2026")
    axes[-1].set_xlabel("Timestamp (UTC)")
    plt.tight_layout()
    fname = "rolling_metrics_summary.png"
    plt.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close()
    saved_figures.append(fname)
    print(f"  Saved {len(saved_figures)} figures to figures/")

    # -----------------------------------------------------------------------
    # STEP 10: STATUS JSON
    # -----------------------------------------------------------------------
    print("\n[STEP 10] Writing phase1_status.json ...")

    if total_shuffle_n == 0:
        power_caveat = (
            "No deterministic shuffle comparison blocks available; "
            "no shuffle baseline for centroid distances."
        )
    elif total_shuffle_n < 20:
        pm_str = ", ".join(f"{k}: {v}" for k, v in shuffle_count_per_month.items())
        power_caveat = (
            f"Low shuffle power: only {total_shuffle_n} shuffle blocks with "
            f"deterministic comparisons (per month - {pm_str}). Centroid-distance "
            f"real-vs-shuffle comparisons are descriptive only. H_t/phi_t/D_t "
            f"have no trusted shuffle baseline (log-text values quarantined)."
        )
    else:
        pm_str = ", ".join(f"{k}: {v}" for k, v in shuffle_count_per_month.items())
        power_caveat = (
            f"Shuffle baseline has {total_shuffle_n} blocks with deterministic "
            f"comparisons (per month - {pm_str}). Centroid-distance comparisons "
            f"are more stable but remain descriptive. H_t/phi_t/D_t have no "
            f"trusted shuffle baseline; this is instrument characterization only."
        )

    cm_summary = {
        "months_compared": months_present if len(months_present) >= 2 else [],
        "skip_reason":     cross_month_skip_reason,
        "distribution_rows": len(cm_dist_rows),
        "autocorr_rows":     len(cm_autocorr_rows),
        "large_autocorr_diff_flags": [
            r["metric"] for r in cm_autocorr_rows
            if r.get("flag_large_autocorr_diff")
        ],
        "large_d_flags": [
            r["metric"] for r in cm_autocorr_rows
            if r.get("flag_large_d")
        ],
    }

    status = {
        "phase": 1,
        "window_start": str(WINDOW_START),
        "window_end":   str(WINDOW_END),
        "row_count_real": rows_after_filter,
        "real_date_range_min": str(ts_min),
        "real_date_range_max": str(ts_max),
        "gap_stats": {
            "median_gap_hours":          median_gap,
            "max_gap_hours":             max_gap,
            "large_gap_count_3x_median": large_gap_count,
            "days_with_zero_runs":       days_with_zero,
        },
        "shuffle_parsing_success":  shuffle_parsing_success,
        "shuffle_parse_status": {
            "logs_present":            list(present_logs.keys()),
            "logs_absent":             absent_logs,
            "parse_failure":           parse_failure,
            "blocks_found_per_log":    blocks_found_per_log,
            "shuffle_blocks_per_month": shuffle_count_per_month,
            "quarantined_metric_lines": len(all_quarantine_rows),
        },
        "shuffle_n_blocks":   total_shuffle_n,
        "scipy_available":    SCIPY_OK,
        "seaborn_available":  SEABORN_OK,
        "metric_provenance_inventory": {
            "computed_observation_csv":         rows_after_filter,
            "deterministic_shuffle_comparison": total_shuffle_n,
            "legacy_shuffle_log_text_untrusted": len(all_quarantine_rows),
        },
        "n_total_regime": {
            "threshold":   N_TOTAL_REGIME_THRESHOLD,
            "total_flagged": int(df_obs["n_total_regime"].sum()),
            "by_month":    df_obs[df_obs["n_total_regime"]].groupby("month").size().to_dict()
                           if df_obs["n_total_regime"].any() else {},
            "note": (
                "n_total_regime rows are real low-n observations (n_total <= "
                f"{N_TOTAL_REGIME_THRESHOLD}). They are signal, not excluded "
                "from H_t_numeric or any analysis."
            ),
        },
        "H_t_outlier_rule": {
            "status":    "suspended_for_real_observations",
            "H_t_numeric": "equals H_t for all real observations (no exclusions)",
            "H_t_init_outlier_col": "all False in real_clean.csv (schema continuity only)",
        },
        "instrument_anomalies": all_instrument_anomalies,
        "saved_tables": [
            "tables/data_coverage.csv",
            "tables/real_descriptive_stats.csv",
            "tables/shuffle_baseline_stats.csv",
            "tables/real_vs_shuffle.csv",
            "tables/temporal_structure.csv",
            "tables/pearson_correlation.csv",
            "tables/spearman_correlation.csv",
            "tables/diagnostic_flags.csv",
            "tables/cross_month_distribution.csv",
            "tables/cross_month_autocorr_persistence.csv",
            "analysis_intermediate/quarantine_shuffle_metrics.csv",
        ],
        "saved_figures": [f"figures/{f}" for f in saved_figures],
        "high_correlation_pairs": high_corr_pairs,
        "power_caveat": power_caveat,
        "phase0_diagnostics": {
            "obs_raw_timestamp_samples": phase0_obs_raw_samples,
            "log_header_samples":        phase0_log_headers,
            "rows_in":                   rows_in,
            "rows_after_filter":         rows_after_filter,
            "dropped_before_window":     dropped_before,
            "dropped_after_window":      dropped_after,
            "dropped_unparseable":       dropped_unparseable,
            "surviving_min":             str(ts_min),
            "surviving_max":             str(ts_max),
            "rows_per_month":            rows_per_month,
            "shuffle_phase0":            phase0_shuf_stats,
        },
        "cross_month_stability": cm_summary,
        "warnings": WARNINGS_LIST,
    }

    with open(INTER_DIR / "phase1_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    print("  Saved analysis_intermediate/phase1_status.json")

    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print(f"  Real obs: {rows_after_filter} rows, {ts_min} --> {ts_max}")
    print(f"  n_total_regime rows (signal, not excluded): "
          f"{int(df_obs['n_total_regime'].sum())}")
    print(f"  Shuffle blocks (det-comp): {total_shuffle_n}")
    print(f"  Quarantined log-text metric lines: {len(all_quarantine_rows)}")
    print(f"  shuffle_parsing_success (three-state): {shuffle_parsing_success}")
    n_hcp = len(high_corr_pairs)
    print(f"  High-correlation pairs (Pearson |r|>0.5): {n_hcp}")
    for p in high_corr_pairs:
        print(f"    {p['metric_a']} <-> {p['metric_b']}  r={p['pearson_r']}")
    if all_instrument_anomalies:
        print(f"  Instrument anomalies retained:")
        for a in all_instrument_anomalies:
            print(f"    {a['metric']}={a['value']} @ {a['block_ts']}: "
                  f"{a['untrust_reason']} - instrument/logging-provenance anomaly "
                  f"from the April first-run shuffle record, retained for "
                  f"transparency, excluded from baseline stats.")
    print(f"  Warnings ({len(WARNINGS_LIST)}):")
    for w in WARNINGS_LIST:
        print(f"    - {w}")
    print(f"  Power caveat: {power_caveat}")
    print("=" * 70)


if __name__ == "__main__":
    main()
