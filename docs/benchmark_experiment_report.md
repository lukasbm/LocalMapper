# LocalMapper Benchmark Experiment Report

## Objective

This benchmark evaluates LocalMapper as an atom-to-atom mapping baseline for comparison with the GNN-based benchmark in the paper. The benchmark uses EEquAAM for mapping equivalence accuracy and avoids AP/F1/MCC-style confidence metrics, because the current comparison target is whole-reaction atom-map correctness rather than calibrated binary edge classification.

## Run Modes

The benchmark contains three run modes:

1. **Unchanged paper checkpoint (`paper_checkpoint`)**: evaluate the released LocalMapper checkpoint without any training or fine-tuning. This uses `data/checkpoints/LocalMapper_202403.pth` and the released template library `data/checkpoints/templates_202403.pkl`. Because no model fitting is performed, there is no train/test contamination risk from the local benchmark datasets. This mode evaluates `split=all` for every dataset.
2. **Pretrained / fine-tuned (`pretrained`)**: start from `data/checkpoints/LocalMapper_202403.pth` for the first active-learning iteration, then continue from the previous iteration checkpoint. This mode trains only on the dataset training split and evaluates on the dataset test split.
3. **From scratch (`scratch`)**: train LocalMapper with the same active-learning pipeline and hyperparameters, but initialize randomly instead of loading the released checkpoint. This mode also trains only on the dataset training split and evaluates on the dataset test split.

All trainable runs use five seeds: `0 1 2 3 4`.

## Datasets and Splits

The default benchmark datasets are `ringreactions` and `metAMDB`. `USPTO_50K` is available as an opt-in dataset by running the sweep with `INCLUDE_USPTO=1`.

When enabled, `USPTO_50K` is loaded from `data/USPTO_50K/raw_data.csv`, which contains 49,996 reactions plus one CSV header row. The current repository copy does not include an explicit original train/validation/test column or separate split files. For trainable runs, the pipeline therefore uses the repository's deterministic seed-based split: 10% test, 10% validation, and the remaining 80% training after LocalMapper normalization/filtering. The split is seed-dependent and is not k-fold cross-validation. For the unchanged paper checkpoint mode, USPTO is evaluated with `split=all`.

`ringreactions` uses the provided source split files: `data/ringreactions/train_ringreactions.csv` with 141 reactions and `data/ringreactions/test_ringreactions.csv` with 36 reactions. Trainable runs sample/train from the train file and evaluate on the test file. The unchanged paper checkpoint evaluates all 177 reactions by using `split=all`.

`metAMDB` uses the provided source split files: `data/metAMDB/train_metamdb_filtered.csv` with 8,455 reactions and `data/metAMDB/test_metamdb_filtered.csv` with 2,124 reactions. Trainable runs sample/train from the train file and evaluate on the test file. The unchanged paper checkpoint evaluates the union of train and test by using `split=all`.

The source counts above describe the input split files. The exact metric denominator is reported in every `metrics_<split>_<iteration>.json` file after LocalMapper's reaction normalization, demapping, and filtering.

## Reaction Preprocessing

Model inputs are demapped before prediction so that LocalMapper cannot read the reference atom-map numbers from the reactant/product strings. The ground-truth mapped reaction is retained separately for scoring. This is the intended setup for fair AAM evaluation.

The template-verification path remains in the LocalMapper pipeline. Trainable runs use templates accepted from the selected active-learning annotations and confident pseudo-labeling. The unchanged paper checkpoint additionally uses the released `templates_202403.pkl` library, matching the released model.

## Active-Learning Budgets

The default active-learning budget is chosen to be feasible while still giving multiple selection/training rounds:

| Dataset | Sample limit per iteration | Iterations | Total annotations per seed |
| --- | ---: | ---: | ---: |
| `USPTO_50K` | 200 | 5 | 1,000 |
| `ringreactions` | 10 | 10 | 100 |
| `metAMDB` | 200 | 5 | 1,000 |

Candidate selection uses `SAMPLE_CANDIDATE_FACTOR=20`, so each iteration considers up to `sample_limit * 20` candidates before selecting uncertain predicted templates and filling randomly if needed.

## Training Setup

Trainable modes use the original LocalMapper training defaults:

| Parameter | Default |
| --- | ---: |
| Epochs | 100 |
| Batch size | 16 |
| Early-stopping patience | 5 |
| Learning rate | 1e-3 |
| Weight decay | 1e-6 |
| Gradient clip | 20 |
| LR scheduler step parameter | 10 |
| Validation fraction inside selected annotations | 0.1 |
| Confident pseudo-labels per template | 100 |

No hyperparameter tuning is part of this benchmark. The intent is to compare the method under its established configuration, with only the initialization mode and dataset changing.

## Metrics

The primary metric is strict EEquAAM ITS equivalence exact-match accuracy:

`aam.strict_equiv_exact_match_accuracy = correct_equivalent_mappings / predicted_evaluation_reactions`

EEquAAM compares the predicted mapped reaction against the reference mapped reaction and treats chemically equivalent atom maps as correct. The pipeline records unevaluable cases and failed comparisons instead of stopping the run.

For backward compatibility, `aam.equiv_exact_match_accuracy` is still written with the same value as `aam.strict_equiv_exact_match_accuracy`. This is a conservative metric: EEquAAM failures and unevaluable rows remain in the denominator and count as incorrect. `aam.evaluable_accuracy` should be reported as a diagnostic, not as the headline metric, because it excludes failed comparisons.

The metric JSON files also include diagnostics:

- `aam.evaluable_accuracy`: exact-match accuracy over reactions where EEquAAM produced an evaluable comparison.
- `aam.evaluable_count` and `aam.unevaluable_count`: denominator diagnostics for EEquAAM.
- `aam.predicted_count`: number of reactions actually predicted and sent to metric aggregation after LocalMapper dataset filters.
- `aam.eequaam_failed_count`: number of rows where EEquAAM failed or omitted a comparison result.
- `aam.eequaam_status_counts`: status distribution from EEquAAM, including fallback or failure statuses.
- `aam.invalid_mapping_count`: predictions rejected before or during equivalence comparison.
- `aam.raw_exact_match_accuracy` and `aam.atom_accuracy`: retained only as debugging diagnostics, not as the benchmark headline metric.

The old AP/F1/MCC reporting is intentionally not used for this benchmark because EEquAAM provides whole-reaction mapping accuracy, not edge-level ranking or classification scores.

## EEquAAM Reliability Controls

EEquAAM is invoked in chunks to limit blast radius from brittle reactions and long-running comparisons. The default controls are:

| Parameter | Default |
| --- | ---: |
| `EEQUAAM_CHUNK_SIZE` | 25 |
| `EEQUAAM_BASE_TIMEOUT_SECONDS` | 10 |
| `EEQUAAM_TIMEOUT_SECONDS_PER_REACTION` | 2 |
| Temporary directory | `/scratch/lukas/tmp/localmapper/eequaam` |

If a chunk fails, the wrapper falls back to smaller comparisons and marks failed rows as unevaluable or incorrect according to the recorded status. If EEquAAM exits but omits a comparison, that comparison is explicitly marked as `eequaam_failed`. This prevents one bad reaction from aborting the full benchmark and makes the denominator auditable.

## Default Pipeline

The default command is:

```bash
RUN_ID=paper_benchmark scripts/experiments/comparison_sweep.sh
```

To resume the deterministic default sweep after already completing the first 15 runs, use:

```bash
RUN_ID=paper_benchmark SKIP_RUNS=15 scripts/experiments/comparison_sweep.sh
```

`SKIP_RUNS` skips complete top-level runs in the printed plan order. It is only valid when the rest of the sweep configuration is unchanged. With the default configuration, runs 1-5 are `ringreactions` paper-checkpoint evaluations, runs 6-10 are `metAMDB` paper-checkpoint evaluations, runs 11-15 are `ringreactions` scratch runs, and run 16 starts `ringreactions` pretrained.

To include USPTO_50K in addition to the default datasets:

```bash
INCLUDE_USPTO=1 RUN_ID=paper_benchmark scripts/experiments/comparison_sweep.sh
```

Preview the planned runs without executing:

```bash
PLAN_ONLY=1 scripts/experiments/comparison_sweep.sh
```

The default plan contains 30 runs:

- 10 unchanged paper-checkpoint evaluations: 2 datasets times 5 seeds.
- 20 trainable runs: 2 datasets times 2 modes (`scratch`, `pretrained`) times 5 seeds.
- 150 total active-learning training/evaluation iterations across all trainable runs.

With `INCLUDE_USPTO=1`, the plan contains 45 runs and 200 active-learning iterations.

The paper-checkpoint seed repetitions are kept for consistent output layout with the trainable modes. Since this mode uses `split=all` and does not train, the seed should not affect the model; repeated results can be sanity-checked or collapsed when reporting.

The sweep prints a large colored completion banner after each finished run, including the completed run count, total run count, percentage, dataset, mode, seed, and model name. Long-running Python stages also expose `tqdm` progress bars for dataset filtering, annotation/template extraction, prediction, training epochs, training batches, validation batches, and metrics-summary aggregation.

The environment pins PyTorch to `torch==2.7.1+cu128` on Linux x86_64. This avoids the CUDA 13 wheel, which requires a newer NVIDIA driver than the current machine provides. `scripts/Train.py` and `scripts/Test.py` fail fast when a CUDA device is requested but unavailable; set `GPU=cpu` or `ALLOW_CPU=1` only for an intentional CPU run. The sweep also exports `UV_CACHE_DIR=/scratch/lukas/tmp/uv-cache` by default to avoid writing into read-only uv cache locations.

## Outputs

Each run writes to:

```text
outputs/<dataset>/<model>_seed<seed>/
```

Important files:

- `run_metadata.json`: exact run configuration, split names, EEquAAM settings, checkpoint paths, software versions, and git state.
- `iteration_<n>.pth`: trained LocalMapper checkpoint for trainable modes.
- `training_history_iteration_<n>.csv` and `training_history_iteration_<n>.json`: per-epoch train loss, validation loss, learning rate, and early-stopping flag.
- `annotations/`: selected active-learning annotations.
- `predictions/pred_<split>_<iteration>.csv`: row-level predictions and EEquAAM result columns.
- `predictions/metrics_<split>_<iteration>.json`: headline metric and diagnostics for that evaluation.
- `metrics_summary.csv` and `metrics_summary.json`: per-run metric summaries across iterations.

`outputs/analyze_metrics_summaries.py` aggregates the per-run `metrics_summary.csv` files into paper-facing comparison tables and plots. It lives under `outputs/` because it scans the generated run directories and writes its derived artifacts to `outputs/analysis/`.

After the sweep, aggregate summaries can also be regenerated manually with:

```bash
uv run python outputs/analyze_metrics_summaries.py --run-id paper_benchmark
```

The sweep runs this analysis automatically by default after all runs finish. Set `ANALYZE_AFTER_RUN=0` to skip it.

## Current Readiness Notes

The pipeline is configured for the agreed benchmark protocol and the default sweep now matches the three-mode design. The local environment imports the CUDA 12.8 PyTorch build successfully. In the sandbox used for maintenance, GPU device files and NVML are not visible, so CUDA availability cannot be proven from that sandbox; on the normal experiment shell, the run should use the GPU if the CUDA 12.8 wheel is compatible with the installed driver.

The full sweep has not been executed because it is expensive. The dry-run plan and shell syntax checks should be used before launching or resuming the full experiment.
