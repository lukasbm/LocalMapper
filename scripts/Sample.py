from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]

from localmapper.active_learning import (
    load_annotations,
    load_dataset_split,
    sample_annotations,
    save_annotations,
)


def main(
    dataset="USPTO_50K",
    model="LocalMapper",
    iteration=1,
    split="train",
    sample_limit=200,
    seed=0,
    val_fraction=0.1,
    test_fraction=0.1,
):
    split_items = load_dataset_split(
        dataset,
        split,
        data_root=ROOT / "data",
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    annotations = load_annotations(ROOT, dataset, model, seed, iteration - 1)
    annotated_ids = (
        set(annotations["data_idx"].astype(str)) if not annotations.empty else set()
    )
    pool_size = len(split_items)
    working_size = sum(str(item["id"]) not in annotated_ids for item in split_items)
    print(
        "Iteration %s: dataset %s split %s has %d reactions (%d remaining)"
        % (iteration, dataset, split, pool_size, working_size)
    )
    annotations = sample_annotations(
        ROOT,
        dataset,
        model,
        seed=seed,
        iteration=iteration,
        split=split,
        sample_limit=sample_limit,
        data_root=ROOT / "data",
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    output_path = save_annotations(ROOT, dataset, model, seed, iteration, annotations)
    print(f"Saved {len(annotations)} emulated annotations to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
