# LocalMapper comparison experiment design

## Changes from `paper.md`

The original LocalMapper paper used a human-in-the-loop workflow: the model selected reactions for review, chemists corrected the atom-to-atom mapping (AAM), and the corrected mappings were added to the next training round. This repository keeps the active-learning structure but replaces the manual review step with dataset ground truth. For each sampled reaction, `scripts.Sample` stores the dataset mapping as the emulated correction and records the model prediction that would have been shown to a chemist in `model_mapped_rxn` and `model_template`. The annotation rows are marked with `annotation_source=dataset_ground_truth`.

The exact AAM comparison now uses the provided EEquAAM implementation in default ITS mode. `scripts.Test` writes each predicted/reference pair in EEquAAM input format, runs EEquAAM in chunks, and recursively isolates failing chunks so one bad reaction does not abort the full benchmark. Raw string equality is still saved separately as `raw_exact_match_accuracy`, but it is not used as the exact AAM metric.

## EEquAAM stabilization

EEquAAM terminates the whole process if it sees an unreadable or incomplete map, so the benchmark does not call it on the full prediction table directly. The implementation prefilters for complete bijective maps, runs EEquAAM in chunks, and recursively splits failing chunks until individual bad comparisons can be marked unevaluable.

The main knobs are `EEQUAAM_CHUNK_SIZE`, `EEQUAAM_BASE_TIMEOUT_SECONDS`, `EEQUAAM_TIMEOUT_SECONDS_PER_REACTION`, and `LOCALMAPPER_EEQUAAM_TMPDIR`. The temp directory defaults to `outputs/eequaam_tmp` because `/tmp` may be too small on shared machines.

## Metrics and metadata

Per-iteration evaluation metrics are saved in `predictions/metrics_<split>_<iteration>.json`. The run-level summaries `metrics_summary.json` and `metrics_summary.csv` flatten those metrics across iterations and are the main files needed for downstream tables and plots. The exact AAM metric to report is `aam.equiv_exact_match_accuracy`, with `aam.equivalence_backend=eequaam_its`. Also report `aam.evaluable_count`, `aam.unevaluable_count`, and `aam.eequaam_status_counts`.

Each run now also writes `run_metadata.json` in the run directory. This records dataset, model name, seed, training split, evaluation split, initialization mode, active-learning budget, training parameters, pretrained checkpoint, template library, split fractions, EEquAAM settings, software versions, and git state. This file is intended for provenance, while `metrics_summary.csv` remains the primary analysis table.

## Experimental design

The comparison should vary the factors that define the LocalMapper use case:

- Dataset: at minimum `USPTO_50K`, `Golden`, `ringreactions`, and `metAMDB`; additional datasets such as `NatComm` and `schneider` can be added when runtime permits.
- Initialization: train from scratch and fine-tune from the released LocalMapper checkpoint. Pure pretrained evaluation is a separate no-training protocol via `scripts/experiments/run_pretrained_eval.sh`.
- Annotation budget: compare a low-budget setting against the paper-style budget. The default sweep uses `50 x 3` and `200 x 5` annotations for most datasets, and `5 x 5` and `10 x 10` for `ringreactions`.
- Split policy: preserve source train/test files when the dataset provides them. `metAMDB` and `ringreactions` keep their source train/test files; datasets without source splits are split by `seed`, `val_fraction`, and `test_fraction`.
- Seed: use seed 0 for the full matrix. For final claims, five targeted seeds are a practical compromise; avoid a 20-seed full matrix unless runtime stops being a constraint.

The provided launcher is `scripts/experiments/comparison_sweep.sh`. It runs all selected dataset, mode, budget, and seed combinations through the same `Sample -> Train -> Test` loop and stores each run under `outputs/<dataset>/<model>_seed<seed>/`.

The argument for running the full default matrix is that each axis answers a different comparison question:

- Multiple datasets test whether the method generalizes beyond a single chemistry domain and whether EEquAAM evaluation remains stable on both standard and difficult datasets.
- Scratch versus fine-tuning tests whether gains come from the active-learning procedure itself or from starting from the released checkpoint.
- Low versus standard annotation budgets test sample efficiency, which is the central claim of the LocalMapper active-learning setup.

With the default settings and `SEEDS=0`, the launcher performs 16 top-level runs:

- `USPTO_50K`, `Golden`, and `metAMDB`: 2 modes x 2 budgets = 4 runs each.
- `ringreactions`: 2 modes x 2 budgets = 4 runs.

Because each run contains several active-learning iterations, the full default matrix expands to 78 `Sample -> Train -> Test` iterations in total:

- `USPTO_50K`, `Golden`, and `metAMDB`: `(3 + 5)` iterations x 2 modes x 3 datasets = 48 iterations.
- `ringreactions`: `(5 + 10)` iterations x 2 modes = 30 iterations.

This is a reasonable comparison matrix, but it is already expensive. A common workflow is:

- Pilot: `PLAN_ONLY=1` first, then `COMPARISON_NUM_EPOCHS=10` with the full matrix or a subset.
- Main comparison: keep the same matrix, restore the main epoch budget, and run `SEEDS=0`.
- Targeted repeats: run up to five seeds only for the smallest subset needed to answer uncertainty questions, for example a single dataset, one budget, and the two initialization modes.

Epochs should not be a primary sweep factor for the comparison experiment. They affect optimization quality and runtime, but they do not directly test the active-learning method. Fixing `NUM_EPOCHS` keeps the comparison interpretable. For pilot runs, use a smaller value such as `COMPARISON_NUM_EPOCHS=10`; for final runs, use the paper-like default of 100 unless validation shows that early stopping consistently ends earlier.

K-fold cross-validation is also not recommended as a default factor. The experiment already has a repeated active-learning procedure over several datasets, two initialization modes, and multiple annotation budgets. K-folds would multiply cost substantially and are less aligned with the paper, which uses defined dataset splits and out-of-distribution datasets. Prefer source splits where available, fraction splits only where unavoidable, and targeted repeat seeds instead of running the full matrix five times.

## Suggested figures

- Learning curves: exact EEquAAM-equivalent AAM accuracy versus cumulative annotation budget, faceted by dataset and initialization mode.
- Budget efficiency: final exact AAM accuracy at each budget, with paired bars for scratch and fine-tuning runs.
- Runtime or failure diagnostics: EEquAAM invalid, non-bijective, timeout, and unevaluable counts per dataset, useful because metAMDB has a visible long tail.
