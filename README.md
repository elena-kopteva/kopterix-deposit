# Kopterix two-month instrument validation — deposit

## 1. Scope

This deposit holds the supporting data, analysis code, and provenance record for the
Kopterix two-month instrument validation. Kopterix is a passive instrument that fetched
and measured a sampled Moltbook public feed over the half-open UTC window
[2026-04-01, 2026-06-01). The accompanying report characterizes the instrument and the
statistical structure of what it recorded during that window. It does not present the
observed Moltbook content as general findings about the platform. All quantitative claims
are scoped to this instrument, this sampling design, and this validation window.

## 2. Directory layout

- deposit root: the run-level source records `observations.csv` and `kopterix_state.csv`,
  the `phase2_status.json` run-state summary, `requirements.txt`, this `README.md`, and
  the `RECOVERY_PROVENANCE.md` provenance record.
- `analysis_intermediate\`: cleaned and intermediate inputs consumed by the Phase 2 and
  Phase 3 scripts — `real_clean.csv`, `shuffle_clean.csv`, `quarantine_shuffle_metrics.csv`,
  and `phase1_status.json`.
- `tables\`: Phase 1 and Phase 2 output tables, including the five Phase 1 presence-gate
  tables (`data_coverage.csv`, `temporal_structure.csv`, `cross_month_distribution.csv`,
  `cross_month_autocorr_persistence.csv`, `diagnostic_flags.csv`) and the derived
  descriptive and comparison tables.
- `figures\`: the fifteen figures cited by the report.
- `scripts\`: the fourteen archived analysis scripts.
- `phase3\`: the Phase 3 subtree.
- `phase3\tables\`: Phase 3 output tables, including `phase3_memory_timescales.csv`,
  `phase3_changepoint_crossref.csv`, and the rarefied-entropy, floor, and layer-ordering
  tables.

## 3. Working-copy warning

Reproduction must be performed on a copy of the archive, never on the distributed files
themselves, because several scripts modify tables in place:

- `kopterix_phase3_task2_D.py` updates staged tables in place.
- `kopterix_phase3_task2_agg_ci.py` updates the aggregate confidence-interval fields in
  `phase3\tables\phase3_memory_timescales.csv`.
- A run of `kopterix_phase3_task2_D.py` without the excluded monthly logs regenerates
  `phase3\tables\phase3_changepoint_crossref.csv` with reduced qualitative annotation.
- The archived `phase3_changepoint_crossref.csv` is the authoritative publication record
  for the distributed annotations.
- The canonical scripts remain the computational authority for the analyses they implement.

Three reproduction details that a copy-based rerun must respect:

- Exact reproduction of the aggregate confidence-interval endpoints depends on the
  numerical environment. Two runs with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` set to `1` were bitwise identical to
  each other. Their confidence-interval endpoints differed from the archived publication
  table by at most 2.260e-07 absolute and 8.763e-09 relative. Every other numerical field
  matched exactly.
- A direct rerun of `kopterix_phase3_task2_agg_ci.py` rewrites the memory-table header with
  an outdated block=16 per-layer note. The archived header is authoritative. The published
  per-layer confidence intervals use block length 24.
- When `kopterix_phase3_task2_agg_ci.py` is run in isolation, `phase3\logs\` must already
  exist. In the documented full reproduction order it is created by an earlier Phase 3 task.

## 4. Reproduction order

Run the scripts on a working copy of the deposit root in this order:

```
kopterix_phase0_diag.py
kopterix_phase1.py
kopterix_phase2.py
kopterix_phase2_corrected.py
kopterix_phase2_addendum.py
kopterix_phase3.py
kopterix_phase3_task2.py
kopterix_phase3_task2_D.py
kopterix_phase3_task2b.py
kopterix_phase3_task2c.py
kopterix_phase3_task2d.py
kopterix_phase3_task2_agg_ci.py
kopterix_phase3_figures.py
kopterix_phase3_task2_figures.py
```

Limitations supported by the dependency audit:

- Phase 1 remains partial without the two monthly logs: the shuffle extraction and the
  deterministic-shuffle comparison cannot be regenerated, so `shuffle_clean.csv` and
  `quarantine_shuffle_metrics.csv` are supplied as archived derived files rather than
  regenerated during a log-free rerun.
- `kopterix_phase3_task2.py` and `kopterix_phase3_task2_D.py` run without the logs, but the
  changepoint cross-reference then loses the qualitative `surface_mood` annotation.
- NPZ series files (`C_series.npz`, `B_angle_arrays.npz`, `D_series.npz`) are generated
  during the Phase 3 sequence and are not independently staged.
- `phase3\figures\` and `phase3\logs\` may be created at runtime by the Phase 3 scripts.
  They do not exist in the distributed deposit.

## 5. Standing exclusions and consequences

- The two monthly logs (`kopterix_log_2026-04.md`, `kopterix_log_2026-05.md`) are excluded
  by author decision.
- Their absence prevents complete regeneration of the Phase 1 shuffle extraction.
- Their absence prevents regeneration of the qualitative `surface_mood` annotations in
  `phase3_changepoint_crossref.csv`.
- The staged derived files preserve the comparison values and annotations used in the
  report. They preserve the documented numerical outputs; the excluded logs additionally
  contain source material that the tables do not carry, so the logs are not replaceable by
  the staged tables.
- Repair and validation helper scripts are excluded.
- The canonical scripts are the authority for regeneration.
- `reconstruction_gate_report.md` is excluded; its outcome is summarized in
  `RECOVERY_PROVENANCE.md`.

## 6. Bootstrap repair verification

A block-length defect was found in the original confidence-interval calculations and
repaired. The corrected intervals in the deposit are produced by the canonical analysis
scripts, and every setting below is read from those scripts.

The moving-block rule, common to the corrected computations, is adjacency-preserving and
non-circular: for a series of length n and block length L_b, draw ceil(n / L_b) contiguous
original blocks x[s:s+L_b] with replacement, each start s uniform on 0..n-L_b inclusive with
no wrap-around; lagged pairs are formed only between members of the same sampled original
block, never across block boundaries and never wrapping the series end to its start; each
lag yields one pairwise-complete Pearson correlation. A block of length L_b supplies
within-block pairs only through lag L_b-1.

Settings by computation:

- Aggregate scalar exponential-fit CIs (`phi_t`, `D_t`, `lexical_cos_step`; raw and
  detrended), from `kopterix_phase3_task2_agg_ci.py`: block length 24 grid points (six days
  on the 6 h grid, 144 h), 5000 resamples, exponential fit over lags 1-16, percentile 95%
  interval from the 2.5th and 97.5th percentiles; seed 42 for the raw series and seed 43 for
  the detrended series.
- Per-layer exponential-fit CIs (`layer_surface`, `layer_mid`, `layer_residue`), from
  `kopterix_phase3_task2_D.py` and `kopterix_phase3_task2b.py`: block length 24 rows, 5000
  resamples, lags 1-16; raw seeds 42, 43, 44 for surface, mid, residue, and centered seeds
  52, 53, 54.
- Per-layer AR(1) CIs (lag 1 only), unchanged by the repair: block length 8, 1000 resamples.
- Aggregate AR(1) CIs (lag 1 only): computed by `kopterix_phase3_task2_agg_ci.py` from the same
  non-circular moving-block draw as the aggregate exponential fit, block length 24, 5000
  resamples, seeds 42 raw and 43 detrended.
- Floor CI over lags 12-16, from `kopterix_phase3_task2c.py`: block length 24, 1000
  resamples (a block of 8 cannot contain a lag 12-16 pair).
- Long-lag CIs to lag 56, from `kopterix_phase3_task2d.py`: block length 64, 1000 resamples.

Verified agreement. The 2026-07-16 canonical CHANGELOG record states that
`kopterix_phase3_task2_D.py` and `kopterix_phase3_task2b.py` compute the per-layer
exponential-fit 95% CIs with the exact within-block pooling of the authoritative repair
script `kopterix_phase3_task2_perlayer_ci.py`, using block 24 (start uniform on 0..n-24
inclusive, size = ceil(n/24)), 5000 resamples, seeds raw 42/43/44 and centered 52/53/54, the
AexpC model with fixed bounds, p0, lags 1-16 and dt, and that this yields exact
floating-point agreement with the repair authority for all twelve per-layer bounds (every
absolute difference 0.0e0), with the AR(1) family, aggregate rows, changepoints, and
residual-angle data unchanged. That record documents the per-layer repair.

For the aggregate confidence intervals, a controlled rerun under the pinned single-thread
environment reproduces the archived point estimates, R^2, and every non-CI field exactly,
and the aggregate CI endpoints agree with the archived publication table within the bounds
quoted in Section 3 (at most 2.260e-07 absolute and 8.763e-09 relative). The embedded
per-layer bootstrap inside `kopterix_phase3_task2.py` uses block length 8 and 500 resamples;
its per-layer interval results are superseded by the later `kopterix_phase3_task2_D.py` and
`kopterix_phase3_task2b.py` computations used for the publication record.

## 7. Permanent provenance gaps

The following gaps predate this deposit and were not repaired here:

- No archived script emits the exact schema of the staged layer-order permutation table.
- The code that added the qualitative annotations to `phase3_changepoint_crossref.csv` was
  not recovered.
- The historical `ruptures` version used for the original changepoint computation was not
  recorded.

These are described in the corresponding sections of `RECOVERY_PROVENANCE.md`.

## 8. Figure presentation

Some distributed figures include manual presentation edits, consistent with the disclosure
in the report: reduced image size, adjusted colors, and removal of headers duplicated by the
figure captions. These edits do not change the plotted data or axes. The numerical content
of every figure comes from the canonical analysis.

## 9. Regenerable report output

`kopterix_validation_report_2month.md` is an output of `kopterix_phase2.py` and
`kopterix_phase2_corrected.py`. It is read by no archived script and can be regenerated by
either script. It is a written report artifact, not a computational input.
