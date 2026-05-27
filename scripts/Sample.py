from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]

from localmapper.active_learning import sample_annotations, save_annotations


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
