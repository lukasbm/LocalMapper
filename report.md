# AAM Benchmark Experimental Design Report

## Scope

This report summarizes what the current BRMapper publication draft claims, what
the existing GNN/kernel experiments in `code/GNN` appear to have done, and a
recommended experimental design for benchmarking MetaMDB, RingReactions, and
USPTO against other atom-to-atom mapping (AAM) methods.

The central recommendation is:

- Use the existing fixed train/test splits for MetaMDB and RingReactions.
- Use repeated model-training seeds on those fixed splits only for stochastic,
  trainable methods.
- Use 5-fold cross-validation for datasets without an accepted split, such as
  USPTO in the current plan.
- Report mapping-level AAM accuracy with coverage/failure rate for all mappers.
- For BRMapper-style candidate selection, also report candidate-level binary
  classifier metrics.

## What The Publication Draft Currently Claims

The draft in `latex/publication_BRMapper.tex` presents BRMapper as a method that
does not generate atom maps directly. Instead, it generates chemically plausible
candidate AAMs using MCES/McSplit-derived options and trains a model to select
the correct candidate.

### Datasets

The draft states:

- MetaMDB starts from approximately 43,000 metabolic reactions. Only validated
  AAMs are used. This gives 10,579 reactions, then additional filtering gives a
  final dataset of 4,456 reactions.
- RingReactions are extracted from validated MetaMDB reactions. The draft says
  1,247 ring reactions are found, 884 remain after balanced-reaction filtering,
  443 mechanisms are identified, and clusters with more than 9 examples are
  retained, giving 225 reactions.

There is a mismatch with the current repository files:

- `data/Inputdata/RingReactions/Reaction_Lists/ringreactions_mainCluster_bereinigt.csv`
  has 177 reactions plus a header.
- The fixed RingReactions split files have 141 train reactions and 36 test
  reactions:
  `data/Inputdata/RingReactions/Reaction_Lists/train_ringreactions.csv` and
  `data/Inputdata/RingReactions/Reaction_Lists/test_ringreactions.csv`.
- The TU candidate datasets have 10,646 train candidate graphs from 141 train
  reaction ids and 2,602 test candidate graphs from 36 test reaction ids.

This should be resolved before publication. Either the paper is describing an
older filtering stage, or the code/results are using a stricter subset than the
text says.

### BRMapper/GNN/Kernel Setup

The draft claims:

- Six model families are evaluated: GIN, GAT, GraphSAGE, NH, NSPD, and WL-OA.
- Each dataset is split 80% train / 20% test, stratified by reaction mechanism.
- Hyperparameter optimization uses 100 TPE iterations.
- For GNNs, optimization uses an 80%/20% train/validation split inside the
  training data.
- For graph-kernel SVMs, optimization uses majority-class undersampling because
  the Gram matrix scales quadratically.
- Final models are retrained on the training split and evaluated on the held-out
  test split.
- For final mapping selection, the AAM candidate with the highest prediction
  score is selected.
- Reported BRMapper accuracy is the percentage of reactions for which the
  selected AAM is correct.

The draft's result table reports reaction-level top-1 candidate-selection
accuracy over 20 runs for RingReactions, MetaMDB, and k/k1/k2/k3 ITS
representations. This is different from the binary classifier metrics used in
the GNN report.

### Benchmarking Other Mappers

The active benchmark text currently covers RXNMapper only:

- RXNMapper version 0.4.2 is used.
- Original ground-truth mappings are removed before prediction.
- Predictions are compared to the ground truth with EEquAAM.
- The RingReactions RXNMapper result is reported as 39.55%.

Graphormer and LocalMapper are still marked as TODO in the active text.
Commented-out text contains candidate values:

- Graphormer: 61.67% without additional training, 61.13% with USPTO50K +
  RingReactions pretraining, 46.89% when trained only on RingReactions.
- LocalMapper: 47.44% without retraining, 72.62% with USPTO50K +
  RingReactions pretraining, 66.80% when trained only on RingReactions.

Do not cite these commented values as final until the exact experimental setup
is reconstructed and aligned with the fixed splits.

There is also a split-alignment concern. The RXNMapper RingReactions EEquAAM
summary contains 70 equivalent and 107 non-equivalent mappings, i.e. 177 total
reactions and 70/177 = 39.55%. That appears to evaluate the full 177-reaction
filtered RingReactions set, not the 36-reaction fixed test split used by the GNN
TU data. If BRMapper is evaluated on the 36-reaction test split but RXNMapper is
reported on all 177 reactions, the comparison is not clean.

For MetaMDB, the repository contains RXNMapper/EEquAAM files for 1,788 mapped
reactions, with the summary showing 1,353 equivalent and 426 non-equivalent
among the filtered EEquAAM input. That is 1,779 summarized comparisons and about
76.1% conditional accuracy on the summarized subset. This is not the full 4,456
reactions described in the draft, so coverage and filtering need to be reported.

## What The Existing GNN Work Did

The relevant files are:

- `code/GNN/train_test_gnn.py`
- `code/GNN/train_test_svm.py`
- `code/GNN/optimize_gnn.py`
- `code/GNN/optimize_svm.py`
- `code/GNN/src/utils.py`
- `code/GNN/src/metric.py`
- `code/GNN/results/results.tex`

### Task Definition

The GNN/kernel work is a binary candidate classifier, not a direct AAM mapper.
Each candidate ITS graph is labeled as correct or incorrect. The model predicts
a positive-class score. For BRMapper reaction-level evaluation, candidates are
grouped by reaction id and the highest-scoring candidate is selected.

This explains why two kinds of metrics appear in the project:

- Candidate-level binary metrics: accuracy, balanced accuracy, precision,
  recall, F1, AUROC, average precision, specificity, MCC, TP/TN/FP/FN.
- Reaction-level mapping accuracy: top-1 selected candidate is EEquAAM-equivalent
  to the ground truth.

Both are useful, but they answer different questions.

### Data Handling

The GNN code loads prebuilt TU datasets. By default, `load_dataset` applies:

- `RemoveZeroSamples`: removes graphs with no nodes or no edges.
- `RemoveIsomorphicSamples`: removes duplicate/isomorphic candidate graphs using
  a Weisfeiler-Lehman graph hash.

For GNN optimization, `split_dataset` performs a group split by reaction id. It
splits unique reaction ids, then assigns all candidate graphs from a reaction to
the same side. That prevents candidate leakage between train and validation.

For final train/test, the code uses pre-materialized train and test TU datasets,
for example `train_Ringreactions_TUdata` and `test_Ringreactions_TUdata`. The
training script itself does not create the external train/test split.

The preprocessing script `code/PreprocessData/split_train_testdata.py` performs
an 80/20 split with `random_state = 42`, stratified by the `cluster` column.
This is consistent with the paper's mechanism-stratified split claim.

### Hyperparameter Optimization

GNN optimization:

- Config: `code/GNN/cfg/optimize_gnn.yaml`
- 100 trials.
- 100 epochs.
- Batch size 1024.
- 20% validation split inside the training dataset.
- Objective in code: minimize validation loss, while logging balanced accuracy
  and other metrics.
- Search spaces include layer count, hidden size, dropout, heads for GAT, pool
  type, and learning rate. Current hparam YAMLs fix `pool_type: max`.

Kernel/SVM optimization:

- Config: `code/GNN/cfg/optimize_svm.yaml`
- Optimization score: `AveragePrecision`.
- 5 evaluation runs per hyperparameter configuration.
- 20% validation split inside the training dataset.
- Grid search is configured for NH, NSPD, and WL-OA.
- Majority-class undersampling is supported and is described as necessary for
  kernel scalability.
- The median validation score across the 5 runs is used.

### Final Runs

Final train/test configs specify 20 runs:

- `code/GNN/cfg/train_test_gnn.yaml`: `runs: 20`, `epochs: 100`.
- `code/GNN/cfg/train_test_svm.yaml`: `runs: 20`.

GNN final training:

- Trains on the fixed training TU dataset.
- Evaluates on the fixed test TU dataset.
- Uses class-weighted NLL loss.
- Uses shuffled batches.
- Logs predictions for each candidate graph.

SVM final training:

- Converts TU graphs to kernel input.
- Uses `CalibratedClassifierCV(SVC(kernel="precomputed", class_weight="balanced"))`.
- Supports undersampling via `undersampling_ratio`.
- Logs candidate predictions.

Important reproducibility caveat: the final train/test scripts do not set a
global seed per run. Therefore, the 20 runs sample stochastic variation, but the
exact run sequence may not be reproducible unless seeds are added and logged.

### Metrics Used By The GNN Work

`code/GNN/src/metric.py` computes:

- Accuracy
- BalancedAccuracy
- Precision
- Recall
- F1
- AUROC
- AveragePrecision
- Specificity
- MatthewsCorrCoef
- TP, TN, FP, FN
- sample count

`code/GNN/results/results.tex` explicitly argues that MCC and AP are more
reliable than balanced accuracy for the candidate-classification problem because
the data are highly imbalanced and GNN results have high variance. It also notes
that false positives are the limiting factor, because one high-scoring wrong
candidate can cause the reaction-level AAM selection to fail.

## Does The Proposed Design Make Sense?

Yes, with two refinements.

For MetaMDB and RingReactions, using the existing fixed split and running
several training seeds is the right design for trainable stochastic methods. The
test set should stay fixed. The seeds should affect model initialization,
dataloader order, dropout, and any stochastic undersampling, not the train/test
split.

For USPTO or any dataset without an accepted split, 5-fold cross-validation is
reasonable. However, do not make each fold from a different random split unless
you intentionally want repeated cross-validation. A cleaner design is:

1. Generate one 5-fold split with a recorded splitter seed.
2. Use the same fold assignment for every method.
3. For each outer fold, train on 4 folds and test on the held-out fold.
4. If hyperparameters are tuned, split the training portion again or use inner
   cross-validation, never the outer test fold.
5. For stochastic methods, run multiple model seeds inside each fold if compute
   allows.

If compute is limited, prefer 5 folds x 1 seed over 1 fold x 5 seeds. If compute
allows, use 5 folds x 5 seeds. Report the design clearly as either "5-fold CV"
or "5-fold CV with 5 training seeds per fold".

## Recommended Benchmark Protocol

### Dataset Splits

MetaMDB:

- Use the same fixed train/test split as BRMapper/GNN if it exists and can be
  reconstructed.
- If benchmarking a pretrained-only mapper, evaluate only the fixed test split.
- If retraining Graphormer/LocalMapper or similar methods, train/tune only on
  the fixed training split and evaluate once on the fixed test reactions.
- Report any reactions excluded because prediction failed, sanitization failed,
  or EEquAAM comparison failed.

RingReactions:

- Use the fixed 141/36 reaction split if that is the BRMapper/GNN split.
- Do not compare BRMapper test accuracy on 36 reactions against RXNMapper
  accuracy on all 177 reactions.
- If you also want a "full filtered RingReactions" diagnostic, report it as a
  separate non-split benchmark.

USPTO:

- Use 5-fold cross-validation if no official split is available.
- Prefer stratification by reaction class/mechanism if reliable labels exist.
- If near-duplicate or template-equivalent reactions exist, use grouped folds to
  prevent leakage.
- Save fold assignments to disk and reuse them for every method.

### Seeds

Use explicit, logged seeds such as `0, 1, 2, 3, 4` or
`42, 43, 44, 45, 46`.

For fixed-split datasets:

- One split seed/provenance value for the dataset split, ideally already fixed
  and not varied.
- Five or more training seeds for stochastic trainable methods.
- No seed repetition needed for deterministic pretrained inference methods.

For 5-fold CV:

- One fold-generation seed.
- Optional training seeds within each fold.
- If using stochastic undersampling, seed and log the undersampling separately
  or derive it deterministically from fold id and training seed.

The existing GNN work used 20 final runs, so 20 seeds would align best with it.
If benchmarking time is the limiting factor, 5 seeds is acceptable, but state
that it is a smaller repeat count than the GNN experiments.

### Hyperparameter Tuning

For trainable baselines:

- Tune hyperparameters using training data only.
- For fixed-split MetaMDB/RingReactions, use an internal validation split from
  the training data, grouped by reaction id and preferably stratified by
  mechanism.
- For USPTO 5-fold CV, tune within each outer training fold or use a fixed
  hyperparameter configuration selected from prior literature. Do not tune on
  the outer test fold.

For pretrained-only mappers:

- Do not tune on benchmark test reactions.
- Report version, checkpoint, default settings, and any decoding parameters.

### Handling Failures And Filtering

For every mapper and dataset, report:

- `N_total`: reactions in the intended evaluation split.
- `N_predicted`: reactions for which the mapper returned a parseable mapped
  reaction.
- `N_compared`: reactions that could be compared by EEquAAM.
- `N_correct`: EEquAAM-equivalent predictions.
- Coverage: `N_predicted / N_total`.
- EEquAAM coverage: `N_compared / N_total`.
- Conditional accuracy: `N_correct / N_compared`.
- Strict accuracy: `N_correct / N_total`, treating failures as wrong.

The strict accuracy should be the primary headline number. Conditional accuracy
is still useful, but it can hide failure modes.

### Primary Metrics

For external AAM mappers, use reaction-level metrics:

- Primary: strict top-1 AAM accuracy by EEquAAM equivalence.
- Secondary: conditional EEquAAM accuracy on compared reactions.
- Coverage/failure rate.
- Runtime per reaction, at least median and interquartile range if available.

For BRMapper candidate selection, use both reaction-level and candidate-level
metrics:

- Primary reaction-level metric: top-1 AAM accuracy by EEquAAM equivalence.
- Optional reaction-level metrics: top-k candidate contains correct AAM, mean
  rank of the correct candidate, number of candidates per reaction.
- Candidate-level metrics: AP, MCC, F1, precision, recall, balanced accuracy,
  AUROC, specificity, confusion matrix counts.

For the candidate classifier, AP and MCC should be emphasized because the
candidate data are highly imbalanced and false positives directly harm top-1
mapping selection.

### Aggregation And Uncertainty

For fixed-split repeated-seed experiments:

- Report mean +/- standard deviation across seeds.
- Also consider a 95% confidence interval over reactions via bootstrap for the
  headline AAM accuracy.

For 5-fold CV:

- Report mean +/- standard deviation across folds.
- If using multiple seeds per fold, aggregate per fold first, then report fold
  means, or use a mixed summary that clearly separates fold variance and seed
  variance.

For comparing two mappers on the same reactions:

- Use paired reporting: count reactions both correct, only A correct, only B
  correct, both wrong.
- A paired bootstrap or McNemar test is more meaningful than comparing two
  independent standard deviations.

## Concrete Action Items

1. Decide which RingReactions evaluation set is official: the 36-reaction fixed
   test split, the full 177-reaction filtered set, or both as separate analyses.
2. Recompute RXNMapper, Graphormer, and LocalMapper on exactly the same official
   test reactions used for BRMapper.
3. For every benchmark run, produce a per-reaction CSV with reaction id,
   ground-truth mapped reaction, unmapped input, predicted mapped reaction,
   mapper status, EEquAAM status, correctness, runtime, dataset split, fold id,
   and seed.
4. Add explicit seed setting and seed logging to any trainable baseline scripts.
5. Update the paper to distinguish:
   - candidate-level classifier metrics, and
   - reaction-level AAM accuracy.
6. Fix the dataset-count mismatch in the draft before adding final benchmark
   numbers.
7. Keep Graphormer/LocalMapper values out of the active manuscript until their
   split, training data, seeds, and EEquAAM filtering are documented.


