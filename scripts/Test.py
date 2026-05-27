from pathlib import Path

import fire
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import torch

from localmapper.cli_utils import (
    init_featurizer,
    load_dataloader,
    load_dataset_templates,
    load_test_model,
)
from localmapper.dataset import mkdir_p


def get_atom_map(
    output_dir,
    output_name,
    try_twice,
    accepted_templates,
    model,
    device,
    data_loader,
):
    model.eval()
    file_path = Path(output_dir) / f"pred_{output_name}.txt"
    with open(file_path, "w") as f:
        f.write("Reaction_id\tMapped_reaction\tTemplate\n")
        with torch.no_grad():
            for batch_data in tqdm(
                data_loader, total=len(data_loader), desc="Predicting AAM..."
            ):
                idxs, rxns, rbg, pbg, _, _, _, items = batch_data
                results = model.map_rxns(
                    rxns,
                    accepted_templates=accepted_templates,
                    try_twice=try_twice,
                    return_dict=True,
                )
                for result, item in zip(results, items):
                    f.write(
                        f"{item['id']}\t{result['mapped_rxn']}\t{result['template']}\n"
                    )
    return


def main(
    gpu="cuda:0",
    batch_size=20,
    dataset="USPTO_50K",
    split="test",
    try_twice=False,
    seed=0,
    val_fraction=0.1,
    test_fraction=0.1,
):
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device("cpu")
    print(
        "Testing with device %s, dataset %s, split %s"
        % (device, dataset, split)
    )

    data_root = str(ROOT / "data")
    output_dir = str(ROOT / "outputs" / dataset)
    model_path = str(ROOT / "models" / dataset / "LocalMapper.pth")
    mkdir_p(output_dir)

    node_featurizer, edge_featurizer, mol_to_graph = init_featurizer()
    test_loader = load_dataloader(
        dataset,
        split,
        mol_to_graph,
        batch_size=batch_size,
        data_root=data_root,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        include_labels=False,
    )
    model = load_test_model(
        node_featurizer, edge_featurizer, mol_to_graph, device, model_path
    )
    accepted_templates = (
        load_dataset_templates(dataset, data_root=data_root, split="train")
        if try_twice
        else set()
    )
    get_atom_map(
        output_dir,
        split,
        try_twice,
        accepted_templates,
        model,
        device,
        test_loader,
    )


if __name__ == "__main__":
    fire.Fire(main)
