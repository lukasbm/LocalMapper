from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import torch

from localmapper.active_learning import (
    checkpoint_path,
    prediction_path,
    verified_templates,
)
from localmapper.cli_utils import init_featurizer, load_dataloader, load_test_model
from localmapper.dataset import mkdir_p


def write_predictions(
    output_path,
    accepted_templates,
    model,
    data_loader,
    try_twice,
):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_data in tqdm(
            data_loader, total=len(data_loader), desc="Predicting AAM..."
        ):
            idxs, rxns, rbg, pbg, _, _, _, items = batch_data
            logits_list = model.score_graphs(rbg, pbg)
            results = model.map_scores(
                rxns,
                logits_list,
                accepted_templates=accepted_templates,
                try_twice=try_twice,
                return_dict=True,
            )
            for result, item in zip(results, items):
                rows.append(
                    {
                        "data_idx": item["id"],
                        "split": item["split"],
                        "mapped_rxn": result["mapped_rxn"],
                        "template": result["template"],
                        "confident": result["confident"],
                        "source": item["source"],
                        "num_mappings": item["num_mappings"],
                    }
                )
    mkdir_p(Path(output_path).parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main(
    gpu="cuda:0",
    batch_size=20,
    dataset="USPTO_50K",
    model="LocalMapper",
    iteration=1,
    split="train",
    try_twice=True,
    seed=0,
    val_fraction=0.1,
    test_fraction=0.1,
    checkpoint=None,
):
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device("cpu")
    print(
        "Testing with device %s, dataset %s, split %s, iteration %s"
        % (device, dataset, split, iteration)
    )

    model_path = checkpoint or str(
        checkpoint_path(ROOT, dataset, model, seed, iteration)
    )
    output_path = prediction_path(
        ROOT, dataset, model, seed, split, iteration
    )

    node_featurizer, edge_featurizer, mol_to_graph = init_featurizer()
    test_loader = load_dataloader(
        dataset,
        split,
        mol_to_graph,
        batch_size=batch_size,
        data_root=ROOT / "data",
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        include_labels=False,
    )
    mapper_model = load_test_model(
        node_featurizer, edge_featurizer, mol_to_graph, device, model_path
    )
    accepted_templates = verified_templates(
        ROOT, dataset, model, seed, through_iteration=iteration
    )
    write_predictions(
        output_path, accepted_templates, mapper_model, test_loader, try_twice
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
