# Recovery provenance for three missing Phase 2 CSV tables

Recovered on 2026-07-15 from the surviving Kopterix raw files, logs, and the
complete inline tables preserved in `kopterix_unified_main.tex`.

## phase2_temporal_structure.csv

Fully regenerated from `observations.csv` over the half-open UTC window
[2026-04-01, 2026-06-01).

For each metric independently:

1. sort rows by `Timestamp_UTC`;
2. drop missing values for that metric;
3. compute Pearson autocorrelation at lags 1 and 4;
4. fit `scipy.stats.linregress` against the usable-row index 0,1,...;
5. report slope, R^2, and slope p value.

This reproduces every rendered table value numerically.

## phase2_viability_assessment.csv

All data-derived rows were regenerated from:

- `observations.csv`;
- `kopterix_state.csv`;
- April and May Markdown logs.

The SciPy version `1.15.3` is historical provenance from the Phase 2 report.
It must not be replaced by the currently installed version when reconstructing
the archived table.

## phase2_shuffle_evidence.csv

Directly recoverable from the Markdown logs:

- 8 usable April deterministic shuffle entries;
- 5 usable May deterministic shuffle entries;
- one April H_t=4.2 first-run logging artifact.

The `218` quarantined-row count is preserved in the report and the project
structure audit, but the row-level source file
`quarantine_shuffle_metrics.csv` is absent. Therefore this summary table is
recoverable exactly, while the missing 218-row intermediate cannot be
reconstructed row-for-row without the original parser output, repository
history, backup, or an older project snapshot.

### Addendum 2026-07-15: quarantine_shuffle_metrics.csv located (supersedes the "absent" claim above)

The statement immediately above that `quarantine_shuffle_metrics.csv` is
absent is superseded. The row-level file was subsequently located at the
project root (`quarantine_shuffle_metrics.csv`), with byte-identical copies in
`analysis_intermediate\` and `phase3\analysis_intermediate\` (all three share
MD5 `c5b1b0cdfec05fbc049287b3d3268e22`). A reconciliation against the summary
confirmed exactly 218 quarantined rows and exact derivation of the summary
values, with a maximum absolute discrepancy of 0. The project-root copy has
been staged into the deposit tables tree. The 218-row intermediate is
therefore recoverable row-for-row and no longer treated as missing.

### Note 2026-07-15: staging gate for the three reconstructed CSVs

The three reconstructed CSVs (`phase2_shuffle_evidence.csv`,
`phase2_temporal_structure.csv`, `phase2_viability_assessment.csv`) are
subject to a staging gate: each is compared against the active (uncommented)
report table in `kopterix_unified_main.tex` and enters the deposit only after
passing that comparison. The per-table verdicts and staging outcomes are
recorded in `reconstruction_gate_report.md`.

## Important

These files recover the table data and a sensible CSV schema. They are not
claimed to be byte-for-byte copies of the original missing CSVs because the
original delimiter, column spelling, and formatting metadata are unavailable.

## Staging omission corrected 2026-07-18 : five deposit-required tables

These five files were absent from the deposit staging tree, and the pass opened on the working suspicion that they had never existed in the project tree. A sweep of the project tree, excluding .git, contradicted that suspicion. Every one of the five was located, and for each file all located copies are byte-identical. The suspicion is recorded here because it was the premise of the pass, and it was wrong.

All five carry filesystem timestamps from the 2026-06-04 analysis runs: real_clean.csv and real_descriptive_stats.csv at 10:56:36, and the three phase2_* tables at 11:03:04. Each filename is written by a project script whose execution on that date is recorded in analysis_intermediate\. That establishes the files as long-predating this pass. It does not by itself prove which specific execution produced the bytes now on disk, and no log ties a run to a checksum, so their exact origin is documented as consistent-with rather than proven.

Because all five were found intact, nothing was regenerated. _zenodo_prep\regen_scratch\ was created and left empty so that the absence of regenerated artifacts is visible rather than implied. Each file went to the staging gate as found and was compared against the active uncommented table in kopterix_unified_main.tex. The gate compared 125 values in total: 77 cells displayed in the report, all of which agreed at the precision printed there, plus 48 descriptive statistics recomputed from real_clean.csv, which has no rendered table of its own and was checked instead by row count and by reproducing real_descriptive_stats.csv to within 1e-9 relative. No disagreement was found in any of the 125. The full cell-level correspondence tables are in _zenodo_prep\recovery5_gate_report.md.

Each file was then copied unmodified from its canonical location into _zenodo_prep\zenodo_deposit\tables\, with the destination checksum verified against the source. No file content was altered, reformatted, or recomputed at any point in this pass.

All five staged rows carry Status FOUND_VERIFIED in staging_log.csv. That string records that the file was located intact in the project tree and verified against the report before staging; it does not record a regeneration event. No file was regenerated. Note also that these five rows were appended by this pass: before 2026-07-18 none of the five appeared in staging_log.csv in any form, neither as staged nor as missing.

Status label correction: the 2026-07-18 run wrote these five rows with the string REGEN_VERIFIED, which had been fixed in advance as the pass label and was emitted unconditionally by the staging script regardless of what the pass found. Since nothing was regenerated, that label misdescribed the outcome. The five rows in staging_log.csv, and the hardcoded string in task4_stage_five.ps1 and task5_provenance_five.ps1, were subsequently relabelled to FOUND_VERIFIED. Only the status string changed: no path, checksum, or file content was altered. The generated reports _zenodo_prep\recovery5_staging_report.md and _zenodo_prep\recovery5_provenance_correction.md still quote the original REGEN_VERIFIED string and remain accurate as records of what the original run emitted.

Why the original staging pass skipped five files that were present on disk is not established. No log records the decision, and until the mechanism is identified it is not known whether any other present-on-disk file was omitted the same way.

### real_clean.csv

Prior status: not listed in staging_log.csv and not recorded in any discrepancy log, since discrepancy_log.csv is not present. Absent from the deposit tables tree. Diagnosis verdict FOUND, with 4 byte-identical copies in the project tree (MD5 B4AEC34F5BD2F57309C54B3B09D65F7B).

Generating script: kopterix_phase1.py, which writes the Phase 1 cleaned observation table into analysis_intermediate\.

Inputs: observations.csv, over the half-open UTC window [2026-04-01, 2026-06-01).

Regeneration: not required. The file exists and was staged as found, unmodified.

Gate verdict: MATCH. This is a data file with no rendered table, so the gate was the row-count and derived-statistics check rather than a cell comparison. The file carries 231 rows, 121 in April and 110 in May, matching the report. All 48 descriptive statistics recomputed from it, being count, mean, standard deviation, median, minimum, and maximum for each of the eight scalar metrics, reproduce real_descriptive_stats.csv to within 1e-9 relative.

Staging outcome: staged from analysis_intermediate\real_clean.csv to _zenodo_prep\zenodo_deposit\tables\real_clean.csv. SHA256 6B4D0B1B15ECE4C5F5E13A90BCCEDE4343067F44D28806A4E9A07AB8CE993564. Logged in staging_log.csv with Status FOUND_VERIFIED.

### real_descriptive_stats.csv

Prior status: not listed in staging_log.csv and not recorded in any discrepancy log, since discrepancy_log.csv is not present. Absent from the deposit tables tree. Diagnosis verdict FOUND, with 2 byte-identical copies in the project tree (MD5 BCA1680F3181158404403403A16F5F47).

Generating script: kopterix_phase1.py, which writes the two-month scalar summary into tables\.

Inputs: real_clean.csv, itself derived from observations.csv.

Regeneration: not required. The file exists and was staged as found, unmodified.

Gate verdict: DISPLAY_ONLY. Compared cell by cell against the active table labelled tab:real-descriptive-stats. All 48 compared cells agree at the precision printed in the report. The differences are presentational: the report renders a six-column subset of the seventeen CSV columns, orders the metric rows differently, substitutes mathematical symbols for the CSV metric names, and prints values to four decimal places.

Staging outcome: staged from tables\real_descriptive_stats.csv to _zenodo_prep\zenodo_deposit\tables\real_descriptive_stats.csv. SHA256 830B08BAF7F79C5E0C921E44DD5CFD0809456D60BCA93A288E6C74BC235CE5A3. Logged in staging_log.csv with Status FOUND_VERIFIED.

### phase2_coverage_audit.csv

Prior status: not listed in staging_log.csv and not recorded in any discrepancy log, since discrepancy_log.csv is not present. Absent from the deposit tables tree. Diagnosis verdict FOUND, with 2 byte-identical copies in the project tree (MD5 0AFE1B9E7DE877CE338BA3F63013DD24).

Generating script: kopterix_phase2_corrected.py, which writes the coverage audit into tables\.

Inputs: observations.csv and kopterix_state.csv, matched on Timestamp_UTC within a five-minute tolerance.

Regeneration: not required. The file exists and was staged as found, unmodified.

Gate verdict: DISPLAY_ONLY. Compared cell by cell against the active table labelled tab:phase2-coverage-audit. All six cells agree exactly, with no rounding involved: 231, 215, 214, 17, 1, 5. The only difference is that the report spells out the column headings in prose.

Staging outcome: staged from tables\phase2_coverage_audit.csv to _zenodo_prep\zenodo_deposit\tables\phase2_coverage_audit.csv. SHA256 D94B09EE17A77482CF7A4CBA18886C3C248327B05B7A770A3FD18103CC9885FF. Logged in staging_log.csv with Status FOUND_VERIFIED.

### phase2_zero_mode_residuals.csv

Prior status: not listed in staging_log.csv and not recorded in any discrepancy log, since discrepancy_log.csv is not present. Absent from the deposit tables tree. Diagnosis verdict FOUND, with 2 byte-identical copies in the project tree (MD5 528A0BD707B9AA2231C3BDFB3E767A1F).

Generating script: kopterix_phase2_corrected.py, which writes the zero mode residual summary into tables\.

Inputs: kopterix_state.csv mean embeddings, after subtracting the two-month grand mean embedding.

Regeneration: not required. The file exists and was staged as found, unmodified.

Gate verdict: DISPLAY_ONLY. Compared cell by cell against the active table labelled tab:phase2-zero-mode-residuals. All eleven cells agree at the precision printed in the report. The differences are presentational: reworded headings and printed rounding, including the drift slope rendered as -1.688e-09 against the stored -1.6878583619394293e-09.

Staging outcome: staged from tables\phase2_zero_mode_residuals.csv to _zenodo_prep\zenodo_deposit\tables\phase2_zero_mode_residuals.csv. SHA256 B142ADEB5434D4C54E9E59B1CC695F5556AFB7CB2FFD7A0A4C987657C7C1EC44. Logged in staging_log.csv with Status FOUND_VERIFIED.

### phase2_layer_cosine_distances.csv

Prior status: not listed in staging_log.csv and not recorded in any discrepancy log, since discrepancy_log.csv is not present. Absent from the deposit tables tree. Diagnosis verdict FOUND, with 2 byte-identical copies in the project tree (MD5 D035DBCACCEF4FA351B33E0A01765CA1).

Generating script: kopterix_phase2_corrected.py, which writes the inter-layer cosine distance summary into tables\.

Inputs: kopterix_state.csv layer centroids under identity labeling.

Regeneration: not required. The file exists and was staged as found, unmodified.

Gate verdict: DISPLAY_ONLY. Compared cell by cell against the active table labelled tab:phase2-layer-cosine-distances. All twelve cells agree at the precision printed in the report. The differences are presentational: the report labels the three pairs d_SM, d_MR, and d_SR rather than cos_dist_surface_mid, cos_dist_mid_residue, and cos_dist_surface_residue, and prints values to six decimal places.

Staging outcome: staged from tables\phase2_layer_cosine_distances.csv to _zenodo_prep\zenodo_deposit\tables\phase2_layer_cosine_distances.csv. SHA256 9CE7D434E3E767DED0A850D0A92C1D0123D8130B5B95FCA2A9D6C67071197374. Logged in staging_log.csv with Status FOUND_VERIFIED.

## Consolidated provenance record, R4 (2026-07-20)

This section consolidates the outcomes of review rounds R3 and R3.1 into the
deposit itself, so that the deposit carries its own provenance record and does
not depend on working files that are not distributed with it.

### R3 and R3.1 disposition

R3 is closed. Of the eighteen items opened, sixteen were confirmed. Items 17
and 18 were defects in the report text rather than in any data table; both have
been corrected in the current kopterix_unified_main.tex. No numerical item
remains unresolved.

R3.1 was a narrow, read-only closure check on the two items that turned on
seeded random number generation. Both are CONFIRMED. R3.1 ran entirely inside
_zenodo_prep\rng_verification_scratch\. A SHA256, length, and last-write-time
snapshot of every file in the project, excluding that scratch directory, was
taken before recomputation (559 files) and again afterwards (559 files). Zero
files outside the scratch directory were added, removed, or changed.

Recomputation environment at R3.1: Python 3.12.4 (Anaconda, MSC v.1929, 64 bit),
numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, ruptures 1.1.10. The historical Phase 2
run used scipy 1.15.3, recorded in phase2_status.json; the historical changepoint
runs used an unrecorded ruptures version.

### Floating point agreement standard

Differences of one unit in the last place (1 ULP) of an IEEE 754 double, found
during R3.1, are accepted as numerical agreement and are not treated as
mismatches. They arise from summation and accumulation order in numpy
reductions, not from any divergence of the random number stream. The exact
difference details are preserved below rather than summarized away.

Comparisons were performed with pandas.read_csv(..., float_precision="round_trip").
This matters and is recorded deliberately: the default pandas C float parser is
lossy at the seventeenth significant digit and, on a first pass, silently
reported all columns as identical. Every difference stated below is visible only
under the round-trip parser. Any future re-verification of these tables must use
float_precision="round_trip" or it will not reproduce these findings.

### phase2_lexical_drift.csv

Authoritative producer: scripts\kopterix_phase2_addendum.py, Item 3, "Lexical
temporal drift vs timestamp-shuffled baseline". This is the only script in the
tree that writes tables/phase2_lexical_drift.csv. A byte-identical copy of the
script exists at the project root.

Seed: 42, read from the script source as rng = np.random.default_rng(seed=42),
declared immediately above the shuffle loop in the lexical drift section.

Repetition count: 200, from the module-level constant N_SHUFFLES_LEXICAL, which
is consumed by for _ in range(N_SHUFFLES_LEXICAL) and is also written out as the
n_shuffles column of the table.

Row filters: the shuffle is a timestamp-shuffled baseline. The unigram rows are
permuted while ts_utc and month are held in their original sorted order, so only
the pairing of lexical content to time position is randomized. The table carries
three scope rows: 2026-04, 2026-05, and overall.

Numerical agreement: CONFIRMED. 30 of 33 numeric cells are bit-identical. Every
column that depends on the seeded RNG stream reproduces exactly, including
js_shuf_mean 0.3184482419036411, cos_shuf_mean 0.01611163193967038,
js_shuf_std 0.0017340942779102417, and cos_shuf_std 0.00030652603246667545, all
at absolute difference 0.0 across all three rows. The three cells that are not
bit-identical are, at full precision:

| Row | Column | Staged | Recomputed | Absolute difference |
|---|---|---|---|---|
| 2026-04 | js_obs_mean | 0.24363435664165192 | 0.2436343566416519 | 2.7755575615628914e-17 |
| overall | js_obs_std | 0.06067560845497103 | 0.060675608454971025 | 6.938893903907228e-18 |
| 2026-04 | js_obs_std | 0.03719221593710706 | 0.03719221593710705 | 6.938893903907228e-18 |

All three are 1 ULP. They occur in the deterministic observed columns, not in
the RNG-dependent columns. Maximum absolute difference 2.78e-17, maximum
relative difference approximately 1.1e-16.

### phase3_residual_angle_null.csv

Authoritative producer: scripts\kopterix_phase3_task2.py, ANALYSIS B,
"orthogonality null for layer residuals". This script computes and writes every
data row of the file.

A second script, scripts\kopterix_phase3_task2_D.py, opens the same path but
touches no data row. It inserts one additional comment line, the "# MECHANISM:"
line about the residuals summing to zero, after the "# CONCLUSION" line. This
fully accounts for the staged file carrying three comment lines where a fresh
Analysis B output carries two. Comment lines 1 and 2 are byte-identical between
staged and recomputed.

Seed: SEED = 42, module-level, used as rng = np.random.default_rng(SEED).
Constants N_NULL = 200 and DIM = 384.

Shared RNG order: one shared random number stream is drawn in a fixed order.
Null1, which draws Gaussian 384-dimensional vectors rescaled to each timestamp's
observed residual norms, is drawn first. Null2, which permutes layer labels
across timestamps, is drawn second. Reproducing this file therefore requires
running both nulls in that order. Running Null2 alone will not reproduce the
staged values.

Repetition count: 200 per null.

Row filters: kopterix_state.csv rows in the half-open UTC window
[2026-04-01, 2026-06-01) number 215. The valid-row filter is
lc.notna() & mean_emb.notna(), where parse_layers rejects a timestamp unless all
three layers parse to finite 384-dimensional vectors that are not all-zero, and
parse_vec requires a finite 384-dimensional MeanEmbedding. 210 rows survive, 104
in 2026-04 and 106 in 2026-05. This matches the "n=210 timestamps" stated in the
staged file header and the n_obs column. real_clean.csv contributes 231 rows in
window; it is loaded by the script but is not used by Analysis B. Derived counts
are n_obs = 210 per pair and 630 for ALL, and n_null = 42000 per pair, being
210 x 200, and 126000 for ALL.

Numerical agreement: CONFIRMED. 78 of 80 numeric cells are bit-identical. Every
deterministic observed column, the Gaussian null summaries, the 200-repetition
label permutation null under seed 42, all counts, both KS statistics, and both
KS p-value columns reproduce. ks_p reproduces exactly, including the denormal
3.952525166729972e-323 in the Null1 SM row and the exact 0.0 entries. The two
cells that are not bit-identical are both in the Null2_layer_label_perm / ALL
row:

| Column | Staged | Recomputed | Absolute difference |
|---|---|---|---|
| null_std | 0.12470072314201165 | 0.12470072314201164 | 1.3877787807814457e-17 |
| std_ratio_obs_over_null | 1.399157278496815 | 1.3991572784968151 | 2.220446049250313e-16 |

These are the same cell twice. std_ratio_obs_over_null is obs_std / null_std, so
the 1 ULP difference in null_std propagates directly into it. null_std for the
ALL row is a standard deviation over 126000 values, where summation order across
the three concatenated pair arrays is the natural source of a 1 ULP difference.
Every per-pair null_std is bit-identical, which is what would be expected if the
RNG stream itself is identical and only the final aggregation rounds
differently. Maximum absolute difference 2.22e-16, maximum relative difference
approximately 1.6e-16.

### Shuffled values versus observed values

tables\shuffle_clean.csv and tables\real_clean.csv are distinct kinds of record
and must not be pooled or compared row for row as if they were the same
quantity.

tables\real_clean.csv holds observed data. Its rows are the Phase 1 cleaned
observation records derived from observations.csv over the half-open UTC window
[2026-04-01, 2026-06-01). Every row carries is_shuffle = False and
metric_provenance = computed_observation_csv. It carries 231 rows, 121 in April
and 110 in May.

tables\shuffle_clean.csv holds shuffled baseline values. Its rows come from the
deterministic shuffle comparison blocks recorded in the monthly Markdown logs,
not from observation records. Every row carries is_shuffle = True and
metric_provenance = deterministic_shuffle_comparison. Only the three centroid
distance columns are populated; post_rate_est, H_t, phi_t, D_t, and n_total are
empty by construction, because the shuffle comparison blocks record centroid
distances only.

The two files therefore answer different questions. real_clean.csv is the
observed series. shuffle_clean.csv is the baseline against which the observed
centroid distances are read, and its values are not observations of the system.

### Provenance tag deterministic_shuffle_comparison

deterministic_shuffle_comparison is the staged provenance tag carried in the
metric_provenance column of tables\shuffle_clean.csv. It marks a value as having
been read from a deterministic shuffle comparison block in a monthly Markdown
log rather than computed from an observation record. It is the tag that
separates baseline rows from observed rows, and it is the reason the two files
above can be told apart without reference to any external document. The
corresponding tag on observed rows is computed_observation_csv.

### Quarantine timestamp caveat

tables\quarantine_shuffle_metrics.csv carries 218 rows and a Timestamp field.
That field stores the timestamp of the enclosing comparison block in the source
Markdown log. It is not an individual observation timestamp. Several quarantined
rows can therefore share one Timestamp value because they were parsed out of the
same block, and a Timestamp in this file must not be joined against
Timestamp_UTC in real_clean.csv or treated as identifying a distinct
observation. The position_in_block and untrust_reason columns, not the
Timestamp, are what distinguish rows within a block. All rows carry
metric_provenance = legacy_shuffle_log_text_untrusted, and none of them is used
in any reported analysis.

### Reconstruction gate outcomes

Three Phase 2 CSV tables were reconstructed and were admitted to the deposit
only after passing a staging gate. The gate compared each reconstructed CSV
against the active, uncommented table in kopterix_unified_main.tex. LaTeX
comments were stripped before parsing. For these three tables the \inputtable
lines are commented out and the three tables are rendered inline, so the inline
tabular bodies are the publication record.

Note on location: the detailed gate audit lives in the project working history
at _zenodo_prep\reconstruction_gate_report.md, which is not part of the deposit.
Earlier text in this document referred to that file by bare filename, which
would not resolve from inside the deposit. Its full content for all three gated
tables is transferred below so that this document is self-contained.

Verdict summary:

| Table | Gate verdict | Staging outcome |
| --- | --- | --- |
| phase2_viability_assessment.csv | MATCH | STAGED (REPAIR_VERIFIED) |
| phase2_shuffle_evidence.csv | DISPLAY_ONLY | STAGED (REPAIR_VERIFIED) |
| phase2_temporal_structure.csv | DISPLAY_ONLY | STAGED (REPAIR_VERIFIED) |

#### phase2_viability_assessment.csv

Input: Task 2 repair CSV _zenodo_prep\repair\phase2_viability_assessment.csv.

Verdict: MATCH.

All 9 active rows agree cell for cell across Check, Value, Status, and Note. The
Value column is string-typed (231, 3.79, 1.15.3, 210/215). Status wording is
preserved (Pass / Limit). The active table renders the Status word via \textsc{}
small caps only; the word itself is identical.

Mismatch recorded: none.

Staged: SHA256 2df2246c9d4e45cb8495bf0ce0dad75f16941b50eb176df62fb0e3a14c025c04.

#### phase2_shuffle_evidence.csv

Input: project-root phase2_shuffle_evidence.csv.

Verdict: DISPLAY_ONLY.

Mismatch recorded, exactly as written in the gate report: "Row 4 label: .tex uses
the LaTeX macro \Ht (renders as H_t); CSV uses literal H_t. Values 8/5/218/1 and
Used-in-analysis cells are identical."

Correspondence, CSV column to rendered heading:

| CSV column | Rendered heading | Agreement |
| --- | --- | --- |
| Source | Source | identical |
| Count | Count | identical |
| Used in analysis | Used in analysis | identical |

Row-label and value correspondence, all values agree:

| CSV label | Rendered label | Label note | CSV Count | Rendered Count | CSV Used | Rendered Used |
| --- | --- | --- | --- | --- | --- | --- |
| April baseline shuffle entries with centroid distance comparisons | April baseline shuffle entries with centroid distance comparisons | identical | 8 | 8 | Yes | Yes |
| May baseline shuffle entries with centroid distance comparisons | May baseline shuffle entries with centroid distance comparisons | identical | 5 | 5 | Yes | Yes |
| Quarantined log metric rows | Quarantined log metric rows | identical | 218 | 218 | No | No |
| April H_t=4.2 anomaly | April H_t=4.2 anomaly | H_t (\Ht macro) | 1 | 1 | No, first run logging provenance artifact | No, first run logging provenance artifact |

Staged: SHA256 ead1cf753f53a3eacacb1ab2f86aacdf56663fc3367432069edc20ab9207988d.

#### phase2_temporal_structure.csv

Input: project-root phase2_temporal_structure.csv.

Verdict: DISPLAY_ONLY.

Mismatch recorded, exactly as written in the gate report: "All 40 numeric cells
equal the active table at its printed precision (CSV carries full precision;
.tex prints rounded). Metric labels d_SM/d_MR/d_SR match the active table."

Correspondence, CSV column to rendered heading:

| CSV column | Rendered heading | Agreement |
| --- | --- | --- |
| Metric | Metric | labels identical incl d_SM/d_MR/d_SR |
| Autocorr lag 1 | Autocorr lag 1 | identical header |
| Autocorr lag 4 | Autocorr lag 4 | identical header |
| OLS slope | OLS slope | identical header |
| OLS R^2 | OLS R^2 | header .tex $R^2$ renders as R^2 |
| Trend p | Trend p | identical header |

Per-cell rounding confirmation, CSV full precision reformatted to the active
table's printed format equals the active cell:

| Metric | CSV reformatted (lag1/lag4/slope/R^2/p) | Active printed | Agree |
| --- | --- | --- | --- |
| post_rate_est | 0.150 / 0.158 / -1.1514e+00 / 0.0639 / 0.0001 | 0.150 / 0.158 / -1.1514e+00 / 0.0639 / 0.0001 | yes |
| H_t | -0.036 / -0.107 / -3.8901e-04 / 0.0072 / 0.2168 | -0.036 / -0.107 / -3.8901e-04 / 0.0072 / 0.2168 | yes |
| phi_t | 0.756 / 0.544 / -9.5529e-05 / 0.4232 / 3.88e-27 | 0.756 / 0.544 / -9.5529e-05 / 0.4232 / 3.88e-27 | yes |
| D_t | 0.436 / 0.156 / 5.9048e-05 / 0.0306 / 0.0108 | 0.436 / 0.156 / 5.9048e-05 / 0.0306 / 0.0108 | yes |
| n_total | -0.016 / -0.023 / -4.0744e-01 / 0.0503 / 0.0010 | -0.016 / -0.023 / -4.0744e-01 / 0.0503 / 0.0010 | yes |
| d_SM | 0.047 / -0.003 / 4.3473e-05 / 0.0099 / 0.1513 | 0.047 / -0.003 / 4.3473e-05 / 0.0099 / 0.1513 | yes |
| d_MR | 0.078 / 0.034 / 3.6348e-05 / 0.0061 / 0.2599 | 0.078 / 0.034 / 3.6348e-05 / 0.0061 / 0.2599 | yes |
| d_SR | 0.138 / 0.083 / 1.0612e-04 / 0.0774 / 4.34e-05 | 0.138 / 0.083 / 1.0612e-04 / 0.0774 / 4.34e-05 | yes |

Staged: SHA256 4c0f5095f3d761229539fc41568e33a1a7596be17058bae5db1cbeac8379779c.

The detailed gate audit is retained in the project working history and is not
intended to remain in the final deposit.

### Known provenance limitation: phase2_layer_order_permutation.csv

The staged tables\phase2_layer_order_permutation.csv is numerically valid and
was independently reproduced from kopterix_state.csv during R3, but the exact
implementation that produced it is not present in the deposit or in the project
tree. The staged file carries nine columns, one of which, n_perms_null, is
emitted by no script in the tree, and its note strings describe an exact
six-permutation null including the identity, whereas the related script
scripts\kopterix_phase2_addendum.py builds an eight-column table and computes
its null mean from the five non-identity permutations. kopterix_phase2_addendum.py
is therefore related but non-authoritative for this file. The staged table
matches the report, which describes a six-permutation exact null including the
identity, so the report and the table agree with each other; it is the script
that does not. The table was not modified and was not replaced with output from
the five-permutation script. This is recorded as a provenance limitation rather
than repaired.

### Open item found in R4: superseded per-layer bootstrap in two Phase 3 tables

This item is recorded, not repaired. No table and no script was altered.

phase3\tables\phase3_centered_layer_tau.csv and
phase3\tables\phase3_memory_timescales.csv as currently staged carry per-layer
exponential-fit confidence intervals from the earlier moving-block bootstrap
(block 16 rows, 500 resamples for the memory timescales table; block 8 rows,
1000 resamples for the centered layer tau table). The project canonical copies
of both tables carry the corrected intervals from an adjacency-preserving,
non-circular bootstrap with block 24 rows, 5000 resamples and layer-specific
seeds 42/43/44.

kopterix_unified_main.tex reports the corrected values. At line 1605 the report
gives per-layer confidence interval lower limits of about 14.8 to 21.1 h and
upper limits of about 11,931 to 16,324 h. The staged tables carry lower limits
of 12.60, 15.03 and 8.41 h and upper limits of 16849.56, 16375.52 and 13522.29 h.
The staged values therefore do not match the published report.

The deposit also stages the superseded implementations
scripts\kopterix_phase3_task2b.py and scripts\kopterix_phase3_task2_D.py rather
than the corrected project versions, and does not stage
kopterix_phase3_task2_perlayer_ci.py, which is the authoritative producer of the
corrected intervals. As it stands the deposit cannot reproduce the reported
values from its own contents.

Resolving this requires an explicit decision by the author about which table
version and which script versions the deposit should carry. It is recorded here
so that the deposit does not silently disagree with the report.

### Figures

The figures staged under figures\ are exactly those cited by active, uncommented
figure inclusions in kopterix_unified_main.tex, resolved relative to the TeX file
location. Seven of the fifteen active citations resolved to files on disk and
were copied unmodified. Eight cited figures, all Phase 3 figures, did not exist
at the cited path and are recorded as missing rather than substituted from
another location. They are listed in
_zenodo_prep\R4_PROVENANCE_AND_SCRIPT_CLASSIFICATION.md.



## Provenance completion, R4.5 (2026-07-20)

### Addendum: phase2_layer_order_permutation.csv, producer status closed as a gap

This completes the record opened under "Known provenance limitation" above. No
archived script in the deposit or in the project tree emits the exact schema of
the staged table. The related in-tree script scripts\kopterix_phase2_addendum.py
implements a five permutation null over the non-identity relabelings, whereas the
staged table reports an exact six permutation null that includes the identity, and
carries a ninth column, n_perms_null, that appears in no analysis script in the
tree. The two are therefore related but not the same computation, and
kopterix_phase2_addendum.py is not the producer of the staged file.

What is established is the numerical content rather than the producing code. The
staged values were verified by independent recomputation from kopterix_state.csv
during the shuffle reconciliation, and they agree with the six permutation
description given in the report. The exact historical implementation is not
recovered and is recorded as a permanent provenance gap. The table was not
modified.

### phase3_changepoint_crossref.csv, exact producer not identified; preserved unaltered

More than one staged source writes this filename, so no single exact producer is
identified. scripts\kopterix_phase3_task2.py builds a column named delta_hours,
while the staged file carries delta_h, which matches
scripts\kopterix_phase3_task2_D.py; neither can be shown to be the producer of the
staged bytes.

A preserved contained validation rerun, held read only under
_zenodo_prep\validation\perlayer_bootstrap\canonical_alignment\task2_D_run\, was
used as evidence. That rerun reproduced the changepoint rows, the changepoint
dates, the row order and the anomaly flags of the staged file. The changepoint
locations therefore do not indicate a ruptures version discrepancy. The
regenerated delta_h values differed from the staged values only in rounding or
display precision.

Two differences were not resolved. The staged surface_mood field is populated,
while the rerun left that column empty throughout. The surface mood and register
fallback that the staged header describes is present in no staged script. The
scientific changepoint results were therefore independently verified, while the
exact historical annotation step that populated the mood text was not recovered,
and the code that performed it is not present in the deposit.

The staged file is preserved unaltered as the publication record. Its SHA-256 is
A0CEA01BAB8C42B194B33A1929B1F6D995C08D307879182A5533145C7F74CD45. It was not regenerated and was not replaced.

### R4.5 session: deposit aligned to post-repair canonical state

The deposit tree was staged around 2026-07-15, before the per-layer bootstrap
repair concluded on 2026-07-17. This session aligned it to the corrected canonical
state. A SHA-256 sweep of every staged script and every staged Phase 3 table
against its project canonical counterpart found exactly four divergences, all four
of them anticipated. No unexpected divergence and no ambiguous canonical authority
case was found.

Restaged after verification against kopterix_unified_main.tex:

- phase3\tables\phase3_memory_timescales.csv, now SHA-256 840A1F9A654FAA5D2B85B1C4F669E8F53893EB7C1BDA1B524699CACC4987C6EE.
  The superseded staged copy carried per-layer exponential-fit intervals from a
  moving-block bootstrap with block length 16 rows and 500 resamples, giving lower
  limits of 12.60, 15.03 and 8.41 h and upper limits of 16849.56, 16375.52 and
  13522.29 h. The restaged copy carries the corrected intervals from an
  adjacency-preserving, non-circular moving-block bootstrap with block length 24
  rows, 5000 resamples, exponential fitting over lags 1 to 16 and layer-specific
  seeds 42, 43 and 44, giving lower limits of 16.80, 21.07 and 14.77 h and upper
  limits of 13110.76, 16323.50 and 11930.82 h. These are the values the report
  prints at line 1605 as lower limits of about 14.8 to 21.1 h and upper limits of
  about 11,931 to 16,324 h.
- phase3\tables\phase3_centered_layer_tau.csv, now SHA-256 DB3A7019A7178E647CE89D900A89162D9AC54B6BBD4F5A6E594E2F5D556020AF.
  The superseded staged copy applied a single bootstrap with block length 8 rows
  and 1000 resamples to both confidence-interval families. The restaged copy
  separates them: the exponential-fit intervals use block length 24 rows and 5000
  resamples with layer seeds 42, 43 and 44 for the raw series and 52, 53 and 54 for
  the centered series, while the AR(1) intervals remain at block length 8 rows and
  1000 resamples at lag 1. The centered exponential intervals now read 22.6 to
  220.5, 33.3 to 525.9 and 25.5 to 158.2 h, matching the report table.
- scripts\kopterix_phase3_task2b.py, now SHA-256 607004D94802E8C349AB62926A148FAC015B4BB0BD5309A5DC74BFFB6FD013E8.
- scripts\kopterix_phase3_task2_D.py, now SHA-256 088CC90CF4537F8C47B12407AF5F5B25433EB449B5C614E73901877A89FCCBB1.

Both scripts were replaced with the canonical copies that the 2026-07-16 CHANGELOG
entry records as reproducing the repair computation exactly. Verification before
replacement confirmed, on scratch copies, that each source declares block length 24
and 5000 resamples for the exponential family and names
kopterix_phase3_task2_perlayer_ci.py as the computation it follows. The repair
script itself is deliberately not staged: the canonical scripts are the sole
computational authority the deposit carries for the corrected intervals.

Path portability. Eleven staged scripts carried a hard-coded session mount path in
their BASE definition. All eleven were patched on scratch copies and then applied,
project canonical first and then restaged. The Phase 3 scripts anchor at
Path(__file__).resolve().parents[1], which resolves to the project root from
phase3\ and to the deposit root from the deposit scripts\ directory. The two
project-root scripts use the probe idiom already present in
kopterix_phase2_addendum.py. Every patch is confined to path definition lines, every
patched script passes py_compile, and static path resolution was checked in both
layouts. No script was left unresolved and no path layout conflict arose. No
formula, filter, constant, seed, bootstrap setting, filename, schema, numerical
precision, row ordering or analytical control flow was changed.

At the R4.5 stage, the deposit lacked several files required by the archived
scripts. These were `kopterix_state.csv`, `observations.csv`, the four files
under `analysis_intermediate\` (`real_clean.csv`, `shuffle_clean.csv`,
`quarantine_shuffle_metrics.csv`, and `phase1_status.json`), the five Phase 1
tables (`data_coverage.csv`, `temporal_structure.csv`,
`cross_month_distribution.csv`, `cross_month_autocorr_persistence.csv`, and
`diagnostic_flags.csv`), and the two monthly logs. The monthly logs remain
outside the deposit by author decision. The other files were staged in R5.

The earlier list incorrectly classified `kopterix_validation_report_2month.md`
as an input. It is written by `kopterix_phase2.py` and
`kopterix_phase2_corrected.py`, read by no archived script, and regenerated by
either script.

Figures. The report includes figures through the custom command
\safeincludegraphics, with \graphicspath set to figures/. Parsing the active,
uncommented content of kopterix_unified_main.tex found fifteen
\safeincludegraphics calls and no direct \includegraphics call outside the macro
definition, resolving to fifteen distinct cited figures, which matches the expected
count. Seven were already staged by R4 at citation-matching destinations and were
confirmed byte-identical to their canonical sources. Eight were added this session,
copied unmodified from phase3\figures\:

- phase3_hrare_vs_ntotal.png and phase3_hrare_timeseries.png, produced by
  kopterix_phase3_figures.py;
- phase3_task2_acf_fits.png, phase3_task2_changepoints_timeseries.png and
  phase3_task2_perlayer_tau.png, produced by kopterix_phase3_task2_figures.py;
- phase3_task2b_centered_acf.png, produced by kopterix_phase3_task2b.py;
- phase3_task2c_floor_acf.png, produced by kopterix_phase3_task2c.py;
- phase3_task2d_longlag_acf.png, produced by kopterix_phase3_task2d.py.

Two differing copies of phase3_task2_perlayer_tau.png exist in the project tree.
The copy under phase3\figures\ was staged because it carries the same SHA-256 as
the figure written by the contained per-layer bootstrap repair run and as the
canonical snapshot taken at the close of that repair, which settles which copy is
current. The differing copy under phase3\report_v2\figures\ predates the repair and
was left in place. No figure was regenerated and no similarly named file was
substituted. Every active figure citation now resolves inside the deposit figures
tree, and the staged count of fifteen matches the cited count.

Supersedes. This entry supersedes the R4 open item "superseded per-layer bootstrap
in two Phase 3 tables", which recorded the divergence without repairing it, and the
R4 figures note recording eight cited Phase 3 figures as missing.