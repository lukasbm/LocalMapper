from pathlib import Path

import fire
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from localmapper.LocalTemplate.template_extractor import extract_from_reaction
from localmapper.dataset import load_reactions, mkdir_p


def _template_for(rxn):
    try:
        return extract_from_reaction(rxn)
    except Exception:
        return None


def main(
    dataset="USPTO_50K",
    split="train",
    sample_limit=200,
    seed=0,
    val_fraction=0.1,
    test_fraction=0.1,
):
    data_root = ROOT / "data"
    items = load_reactions(
        dataset,
        split=split,
        data_root=data_root,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if sample_limit > len(items):
        sample_limit = len(items)

    rng = np.random.default_rng(seed)
    sampled_positions = sorted(rng.choice(len(items), sample_limit, replace=False))
    sampled_items = [items[i] for i in sampled_positions]

    output_dir = ROOT / "outputs" / dataset
    mkdir_p(output_dir)
    output_path = output_dir / f"sample_{split}.csv"
    pd.DataFrame(
        {
            "data_idx": [item["id"] for item in sampled_items],
            "split": [item["split"] for item in sampled_items],
            "mapped_rxn": [item["rxn"] for item in sampled_items],
            "template": [_template_for(item["rxn"]) for item in sampled_items],
            "source": [item["source"] for item in sampled_items],
            "num_mappings": [item["num_mappings"] for item in sampled_items],
        }
    ).to_csv(output_path, index=False)
    print(f"Sampled {len(sampled_items)} reactions to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
