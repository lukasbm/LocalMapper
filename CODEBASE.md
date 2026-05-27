# LocalMapper Codebase Overview

This repository contains two layers:

- A reusable Python package in `localmapper/`
- Three CLI entrypoints in `scripts/` that drive the active-learning workflow

Run the CLIs from the repository root with module execution:

```bash
python -m scripts.Sample --iteration=1
python -m scripts.Train --iteration=1
python -m scripts.Test --iteration=1
```

## Core Workflow

The project is organized as a human-in-the-loop loop:

1. `Sample.py` picks reactions to review.
2. A human corrects the sampled reactions and saves the corrected file.
3. `Train.py` trains a model on the corrected data plus confident predictions.
4. `Test.py` runs the trained model on the full dataset and writes predictions.
5. The next iteration uses those outputs as input for more sampling.

The key driver is `iteration`. It is a 1-based active-learning round number and is embedded in file names.

## What Each Script Does

### `scripts/Sample.py`

- Reads `data/<dataset>/raw_data.csv`.
- Uses the current `chemist_name` to build a per-user working directory under `data/<dataset>/<chemist_name>/`.
- If there is no previous fixed training file, it samples random reactions.
- If previous rounds exist, it reads previous prediction files and previous template files to choose new candidates.
- Writes:
  - `data/<dataset>/<chemist_name>/pred_train_<iteration>.csv`
  - `data/<dataset>/<chemist_name>/conf_pred_<iteration>.csv`

Relevant code:
- [scripts/Sample.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Sample.py#L61)
- [scripts/Sample.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Sample.py#L126)

### `scripts/Train.py`

- Loads the raw dataset and the per-user manual correction files.
- Trains `LocalMapper` using:
  - manual corrections from `fixed_train_1.csv` through `fixed_train_<iteration>.csv`
  - confident predictions from `conf_pred_<iteration>.csv`
- Saves the model checkpoint to:
  - `models/<dataset>/<chemist_name>/LocalMapper_<iteration>.pth`

Relevant code:
- [scripts/Train.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Train.py#L85)
- [scripts/Train.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Train.py#L123)

### `scripts/Test.py`

- Loads the trained checkpoint for the requested iteration.
- Runs the model on the full dataset.
- Converts predictions into mapped reactions and templates.
- Writes:
  - `outputs/<dataset>/<chemist_name>/pred_<iteration>.txt`

Relevant code:
- [scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py#L22)
- [scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py#L48)

## What Templates Are

Templates are extracted from mapped reactions.

The conversion happens in `prediction2map`:

- the model outputs an atom-mapping score matrix
- the mapper turns that into a mapped reaction string
- `extract_from_reaction(mapped_rxn)` derives the template string

That template is the main unit used for:

- filtering invalid predictions
- deciding whether a prediction is "accepted"
- grouping confident reactions in training

Relevant code:
- [localmapper/atom_mapper.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/atom_mapper.py#L15)

## What Iteration Means

`iteration` is the active-learning round number.

It appears in almost every file name:

- `pred_train_<iteration>.csv`
- `conf_pred_<iteration>.csv`
- `fixed_train_<iteration>.csv`
- `pred_<iteration>.txt`
- `LocalMapper_<iteration>.pth`

It also controls which files are loaded:

- `Sample.py` looks for `fixed_train_<iteration - 1>.csv` to decide whether it can build a new round from previous predictions.
- `ReactionDataset` loads `fixed_train_1.csv` through `fixed_train_<iteration>.csv`.
- `Test.py` loads `LocalMapper_<iteration>.pth`.

Relevant code:
- [scripts/Sample.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Sample.py#L64)
- [localmapper/dataset.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/dataset.py#L88)
- [scripts/Test.py](/homes/biertank/lukas/Documents/repos/LocalMapper/scripts/Test.py#L76)

## Package Modules

### `localmapper/models.py`

User-facing model API and neural network definition.

- Creates scratch models with constructor defaults.
- Loads checkpoints with `LocalMapper.from_checkpoint(...)`.
- Scores reactions for training or inference.
- Maps raw reaction strings with `LocalMapper.map_rxns(...)`.

### `localmapper/models.py`

Defines the neural network:

- DGL MPNN for graph embeddings
- cross-reactivity attention
- atom-mapping attention head

Relevant code:
- [localmapper/models.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/models.py#L1)

### `localmapper/model_utils.py`

Contains the attention blocks used by `LocalMapper`:

- `MultiHeadAttention`
- `FeedForward`
- `CrossReactivityAttention`

Relevant code:
- [localmapper/model_utils.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/model_utils.py#L1)

### `localmapper/atom_mapper.py`

Converts model scores into a mapped reaction string.

- masks impossible atom matches by atom type
- greedily picks the best product/reactant atom pairing
- extracts a template from the final mapped reaction

Relevant code:
- [localmapper/atom_mapper.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/atom_mapper.py#L15)

### `localmapper/dataset.py`

Provides the dataset used by training and testing.

- reads `raw_data.csv`
- injects manual corrections from `fixed_train_*.csv`
- injects confident predictions from `conf_pred_*.csv`
- builds DGL graphs on demand

Relevant code:
- [localmapper/dataset.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/dataset.py#L73)

### `localmapper/cli_utils.py`

Shared glue for the scripts.

- resolves `chemist_name`
- builds the featurizer
- creates dataloaders
- loads models
- loads fixed and accepted templates

Relevant code:
- [localmapper/cli_utils.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/cli_utils.py#L41)
- [localmapper/cli_utils.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/cli_utils.py#L71)
- [localmapper/cli_utils.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/cli_utils.py#L97)
- [localmapper/cli_utils.py](/homes/biertank/lukas/Documents/repos/LocalMapper/localmapper/cli_utils.py#L158)

## Expected Input Files

The code expects these repository-local files and directories:

- `data/<dataset>/raw_data.csv`
- `data/<dataset>/<chemist_name>/fixed_train_<n>.csv`
- `data/<dataset>/<chemist_name>/pred_train_<n>.csv`
- `data/<dataset>/<chemist_name>/conf_pred_<n>.csv`
- `manual/*.user` for the current chemist name

The raw and intermediate CSV files are expected to contain at least:

- `mapped_rxn`
- `template`

For sampled rows:

- `data_idx`
- `freq` for `pred_train_<n>.csv`

## Outputs

- Training checkpoint:
  - `models/<dataset>/<chemist_name>/LocalMapper_<iteration>.pth`
- Test predictions:
  - `outputs/<dataset>/<chemist_name>/pred_<iteration>.txt`
- Sampling outputs:
  - `data/<dataset>/<chemist_name>/pred_train_<iteration>.csv`
  - `data/<dataset>/<chemist_name>/conf_pred_<iteration>.csv`

## Gotchas

- `iteration` is not cosmetic. It must match across sampling, training, and testing.
- `Sample.py` is incremental. Round `N` assumes round `N-1` has already produced `fixed_train_<N-1>.csv`.
- Templates are not arbitrary labels. They are extracted from mapped reactions and then used as the confidence filter.
- `chemist_name` comes from `manual/*.user` if present, otherwise it falls back to the OS username. That name is part of every output path.
- The code assumes you run from the repository root with `python -m scripts.<name>`. Do not `cd` into `scripts/`.
- `Test.py --try_twice` only changes mappings whose first template is not already accepted.
- `ReactionDataset` is driven by file naming conventions. Renaming the CSVs will break loading.
- Some older files in the repo still reflect earlier layouts, but the active path for the CLIs is now the package-based code in `localmapper/`.


## Dataset Stuff

1. Dataset formats differ substantially:

- USPTO_50K/raw_data.csv: mapped_rxn only, 49,996 rows.
- USPTO_50K/pretrained/fixed_train_1..5.csv: data_idx, mapped_rxn, template; this is the 5-iteration chemist-fixed training set.
- USPTO_FULL/pretrained/fixed_train_0..2.csv: same fixed format, but starts at 0 and has 3 files.
- Golden/raw_data.csv: mapped_rxn only, 1,851 rows.
- Golden/test_data.csv: original_id, mapped_rxn, filtered to 1,758 rows.
- NatComm/original_data/*.csv: reaction, mapped_reaction, split by source: patent, typical, complex.
- NatComm/test_data.csv: mapped_rxn, source, plus stray unnamed columns.
- comparison/*.csv: evaluation artifacts, not clean training datasets.

2. Paper/README training and evaluation:

- Pretrained first on USPTO_50K using 5 active-learning iterations of 200 sampled reactions each.
- Further trained/adapted on USPTO_FULL for 2 additional iterations of 500 sampled reactions each.
- Evaluated on filtered USPTO_50K, the full filtered Golden dataset, and the extracted Jaworski/NatComm-style subsets: USPTO, typical, complex.
- README says USPTO raw reactions come from RXNMapper sources and mapped USPTO outputs are Figshare artifacts.

3. Current code issue:

- scripts/Sample.py:24 expects raw_data.csv to contain template.
- Actual raw_data.csv files only contain mapped_rxn.
- So the sampler is incompatible with the checked-in data unless templates are generated from mapped_rxn before sampling or a separate normalized file is created.

4. Meaning of “fixed”:

- “fixed” means human-reviewed/manual-corrected active-learning samples.
- fixed_train_*.csv is not a fixed train split. It is accepted/corrected AAM supervision.
- The current name is misleading and should probably become something like annotations/iteration_*.csv or labels/manual_*.csv.
