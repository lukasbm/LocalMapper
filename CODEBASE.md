# LocalMapper Codebase Overview

The repository contains a reusable package in `localmapper/` and CLI entrypoints in
`scripts/` for the emulated active-learning workflow.

Run CLIs from the repository root:

```bash
python -m scripts.Sample --dataset=USPTO_50K --model=LocalMapper --seed=0 --iteration=1
python -m scripts.Train --dataset=USPTO_50K --model=LocalMapper --seed=0 --iteration=1
python -m scripts.Test --dataset=USPTO_50K --model=LocalMapper --seed=0 --iteration=1 --split=train
```

## Active-Learning Workflow

The paper loop is emulated with dataset ground truth instead of a human chemist:

1. `Sample.py` selects reactions for the next annotation round.
2. The selected reactions are immediately “annotated” with the dataset `mapped_rxn`.
3. `Train.py` trains on annotations through the requested iteration, plus optional confident pseudo-labels from the previous prediction round.
4. `Test.py` writes mapped predictions, predicted templates, and confidence.
5. The next `Sample.py` round samples uncertain predictions first.

Run state is grouped by dataset, model name, and seed:

- annotations: `outputs/<dataset>/<model>_seed<seed>/annotations/annotations_<iteration>.csv`
- verified templates: `outputs/<dataset>/<model>_seed<seed>/templates/verified_templates_<iteration>.csv`
- predictions: `outputs/<dataset>/<model>_seed<seed>/predictions/pred_<split>_<iteration>.csv`
- checkpoints: `models/<dataset>/<model>_seed<seed>/iteration_<iteration>.pth`

## Main Modules

### `localmapper/models.py`

Defines the neural model and public API:

- `LocalMapper(...)` creates a scratch model with constructor defaults.
- `LocalMapper.from_checkpoint(...)` loads weights.
- `score_graphs(...)` and `score_rxns(...)` return product-atom by reactant-atom scores.
- `map_rxns(...)` / `map_scores(...)` convert scores to mapped reactions and templates.

### `localmapper/active_learning.py`

Owns active-learning file layout and state:

- samples random first-round annotations
- samples uncertain predicted-template groups in later rounds
- loads verified templates from emulated annotations only
- loads annotation and confident-pseudo-label training items

### `localmapper/dataset.py`

Normalizes dataset files into `ReactionDataset` items with stable `id`, `rxn`,
`split`, `source`, and `num_mappings` fields.

### `localmapper/atom_mapper.py`

Converts score matrices into mapped reactions, extracts templates, and applies the
neighbor-weight retry behavior.

## Important Gotchas

- Verified template confidence must come only from annotations through the current iteration, not from the full ground-truth dataset.
- `Sample.py` iteration 2+ expects previous predictions from `Test.py` for the same dataset/model/seed/split.
- `Train.py --init=auto` loads the previous iteration checkpoint if it exists, otherwise starts from scratch.
- The dataset ground truth is used only to emulate manual annotation for sampled IDs.
