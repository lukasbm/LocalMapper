# AAM comparison playbook

This document describes the comparison strategy used in this repository for
LocalMapper active-learning experiments and the constraints another agent should
preserve when comparing a different atom-to-atom mapping (AAM) system.

It is written as a practical handoff document, not as a paper summary.

## Goal

The goal is to compare AAM systems under a controlled active-learning setup
while keeping the evaluation aligned with chemistry-aware mapping equivalence.

There are two distinct comparison targets:

1. Compare LocalMapper configurations against each other.
2. Compare LocalMapper against another AAM system or equivalence backend.

Those are not the same experiment. The first changes training and annotation
budget. The second should change only the mapper or equivalence method while
holding everything else fixed.

## Core principle

Only one major factor should change at a time.

- If the question is "does fine-tuning from the released checkpoint help?", keep dataset,
  seed, budget, splits, and evaluation backend fixed.
- If the question is "does a different AAM system help?", keep dataset, seed,
  sampled reactions, splits, and metric definitions fixed.
- If the question is "does another equivalence backend help?", keep the mapped
  reactions fixed and re-score them under the alternate backend.

If multiple axes change at once, the comparison stops being interpretable.

## Current LocalMapper comparison design

The default sweep is implemented in
[scripts/experiments/comparison_sweep.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/comparison_sweep.sh:1).

The experiment varies these factors:

- Dataset: `USPTO_50K`, `Golden`, `ringreactions`, `metAMDB`
- Initialization mode: `scratch`, `finetune`
- Annotation budget:
  - most datasets: `low = 50 x 3`, `standard = 200 x 5`
  - `ringreactions`: `low = 5 x 5`, `standard = 10 x 10`
- Split policy: preserve source train/test files when available. `metAMDB` and
  `ringreactions` use their source train/test files; datasets without explicit
  source splits are split by `seed`, `val_fraction`, and `test_fraction`.
- Seed: currently `0` in the default comparison run. Use five seeds for the
  final targeted comparison when runtime permits; do not spend the budget on
  a 20-seed full matrix.

The active-learning loop is always:

1. `Sample`
2. `Train`
3. `Test`

That loop is orchestrated in
[scripts/experiments/run_active_learning.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/run_active_learning.sh:43).

## What must remain fixed for a fair comparison

Another agent should treat the following as controlled variables unless the
comparison explicitly targets them:

- Dataset and split policy
- Seed
- Sampled reactions per iteration
- Sample candidate factor
- Number of active-learning iterations
- Epoch budget and patience
- Training batch size
- Confidence selection logic
- Exact metric definitions
- Equivalence backend and its timeout / parser settings
- EEquAAM evaluable and unevaluable counts

In this repository, the run metadata already records these controls in
`run_metadata.json`, including:

- active-learning budget and total annotation budget
- initialization mode
- pretrained checkpoint
- training split and evaluation split
- split fractions
- equivalence backend
- EEquAAM timeout, chunking, and parser failure handling
- software versions
- git commit and dirty state

See [scripts/RunMetadata.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/RunMetadata.py:80).

## Annotation model used here

This repository does not use human corrections during comparison runs.

Instead, the active-learning workflow emulates annotation with dataset ground
truth:

- selected reactions are still chosen by the model
- the "correction" written back to training data is the dataset mapping
- the annotation source is recorded as `dataset_ground_truth`

This matters because any new AAM system must be compared under the same
annotation regime. Do not mix human-corrected rounds with dataset-ground-truth
rounds in the same comparison table.

## Exact AAM metric

The primary metric is:

- `aam.equiv_exact_match_accuracy`

This is the paper-aligned exact mapping metric. It does not use raw mapped-SMILES
string equality. Instead it checks whether the predicted and reference mappings
are chemically equivalent under the configured backend.

In the current implementation, the intended backend is `eequaam_its`.

Secondary AAM metrics are:

- `aam.atom_accuracy`
- `aam.raw_exact_match_accuracy`
- `aam.invalid_mapping_count`
- `aam.evaluable_count`
- `aam.unevaluable_count`
- `aam.eequaam_status_counts`
- `aam.total`

Interpretation:

- `equiv_exact_match_accuracy` is the main reported metric.
- `atom_accuracy` is useful when exact equivalence is too strict to show partial
  progress.
- `raw_exact_match_accuracy` is diagnostic only. It will underestimate quality
  whenever two mappings are equivalent but serialized differently.
- `invalid_mapping_count` is a stability signal. A method that wins only by
  failing on fewer rows should be understood differently from one that improves
  actual mapping quality.

Metric construction lives in
[scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py:117).

## Removed calibration metrics

The current benchmark reports EEquAAM exact accuracy and denominator diagnostics
only. It no longer reports AP, F1, MCC, score-threshold metrics, or confidence
confusion matrices. `mapping_score` and `confident` remain in the prediction CSV
as row-level diagnostics, but they are not benchmark metrics.

## Equivalence backend requirements

The repository currently uses EEquAAM in default ITS mode for the primary exact
metric.

Critical settings:

- `LOCALMAPPER_AAM_BACKEND=eequaam_its`
- `EEQUAAM_CHUNK_SIZE=100`
- `EEQUAAM_BASE_TIMEOUT_SECONDS=30`
- `EEQUAAM_TIMEOUT_SECONDS_PER_REACTION=5`
- `LOCALMAPPER_EEQUAAM_TMPDIR=outputs/eequaam_tmp`

These are exported in
[scripts/experiments/run_active_learning.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/run_active_learning.sh:27).

Why this matters:

- parser behavior changes whether noisy reactions are counted or marked unevaluable
- timeout behavior changes whether difficult reactions become failures or hangs
- chunking changes how quickly EEquAAM isolates bad comparisons
- temp-directory placement matters on machines where `/tmp` is small or full

If another agent swaps in another equivalence backend, it must report:

- backend name
- parser policy
- timeout policy
- failure behavior
- whether results are directly comparable to EEquAAM-based runs

Do not compare results across backends without making that explicit.

## Recommended comparison modes

### Mode 1: compare LocalMapper variants

Use the existing sweep and keep the equivalence backend fixed.

Questions answered:

- scratch vs fine-tuning
- low vs standard budget
- easier vs harder datasets

### Mode 2: compare another mapper under the same evaluation

Use the same datasets, same splits, same seed, and same exact-equivalence
metric. The new system should produce mapped reactions on the same evaluation
rows, then be scored through the same metric code or a byte-for-byte equivalent
reimplementation.

This is the cleanest way to compare AAM systems.

### Mode 3: compare another equivalence backend

Keep mapped reactions fixed and recompute `aam.equiv_exact_match_accuracy`
under the new backend. This isolates the effect of the evaluator itself.

Do not mix this with retraining unless the actual question is about the whole
system, not just equivalence checking.

## Minimal protocol for comparing another AAM system

Another agent should follow this protocol.

1. Choose the comparison target.
   `mapper`, `training mode`, or `equivalence backend`
2. Freeze all other axes.
3. Record the run metadata needed for reproducibility.
4. Score predictions with the same metric definitions.
5. Report the primary metric first:
   `aam.equiv_exact_match_accuracy`
6. Report secondary context:
   `aam.atom_accuracy`, `invalid_mapping_count`, and EEquAAM evaluable/unevaluable counts
7. Plot learning curves against cumulative annotation budget, not only final bars.
8. State explicitly if confidence outputs are not comparable across systems.

## Practical runtime policy

Do not run the full matrix for five seeds by default. A single full pass already
takes multiple days, and preserving source train/test splits removes one major
reason to repeat seeds for split robustness on `metAMDB` and `ringreactions`.

Use this cheaper policy:

1. Run the full matrix once with `SEEDS=0`.
2. Use source train/test files where available.
3. Add repeat seeds only for conclusions that are close or important.
4. Prefer a targeted repeat such as one dataset x one budget x two modes.
5. Report that seed replication is targeted rather than exhaustive.

Seeds still matter for model initialization, active-learning sampling,
pseudo-label sampling, and validation subsets inside the annotation set. They
matter less for datasets whose train/test split is fixed by source files.

## Files another agent should use

- Experiment launcher:
  [scripts/experiments/comparison_sweep.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/comparison_sweep.sh:1)
- Per-run loop:
  [scripts/experiments/run_active_learning.sh](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/experiments/run_active_learning.sh:1)
- Metric computation:
  [scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py:60)
- Metadata writing:
  [scripts/RunMetadata.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/RunMetadata.py:80)
- Aggregation / plotting helper:
  [outputs/analyze_metrics_summaries.py](/homes/biertank/lukas/Documents/repos/LocalMapper/outputs/analyze_metrics_summaries.py:1)

## Output artifacts to expect

For each run:

- `predictions/metrics_<split>_<iteration>.json`
- `metrics_summary.json`
- `metrics_summary.csv`
- `run_metadata.json`

For cross-run analysis:

- `<run_id>_metrics_all_iterations.csv`
- `<run_id>_metrics_final_iterations.csv`
- `<run_id>_metrics_deltas.csv`
- dashboard and trajectory PNGs

## Common failure modes

- Comparing runs with different equivalence backends as if they were the same
- Comparing systems with different sampled reactions
- Reporting raw exact match as the main AAM metric
- Ignoring invalid mapping counts
- Reporting EEquAAM accuracy without its evaluable/unevaluable denominator
- Comparing confidence-driven metrics when one system lacks a comparable score
- Changing epoch budget and active-learning budget at the same time
- Mixing pilot runs and final runs in one table
- Forgetting that `ringreactions` uses a different budget grid than other datasets

## Reporting order

For papers, notes, or agent handoff summaries, report in this order:

1. `aam.equiv_exact_match_accuracy`
2. `aam.atom_accuracy`
3. `aam.invalid_mapping_count`
4. `aam.evaluable_count`, `aam.unevaluable_count`, and `aam.eequaam_status_counts`
5. provenance:
   backend, timeout settings, run id, dataset, seed, budget, init mode

That ordering keeps the chemistry-valid exact metric primary while still
showing whether the model's confidence signals are useful and whether the run
was stable.
