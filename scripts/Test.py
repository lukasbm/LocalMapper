from pathlib import Path

import fire
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import torch

from localmapper.cli_utils import (
    get_user_name,
    init_featurizer,
    load_dataloader,
    load_fixed_templates,
    load_test_model,
)
from localmapper.dataset import mkdir_p
from localmapper.utils import predict
from localmapper.atom_mapper import prediction2map


def get_atom_map(
    output_dir,
    iteration,
    try_twice,
    accepted_templates,
    model,
    device,
    data_loader,
):
    model.eval()
    file_path = Path(output_dir) / f"pred_{iteration}.txt"
    with open(file_path, "w") as f:
        f.write("Reaction_id\tMapped_reaction\tTemplate\n")
        with torch.no_grad():
            for batch_data in tqdm(
                data_loader, total=len(data_loader), desc="Predicting AAM..."
            ):
                idxs, rxns, rbg, pbg, _, _, _ = batch_data
                logits_list = predict(model, device, rbg, pbg)
                for rxn, logits, idx in zip(rxns, logits_list, idxs):
                    prediction = torch.softmax(logits, dim=1).cpu().numpy()
                    result = prediction2map(rxn, prediction)
                    if (
                        try_twice
                        and result["template"] not in accepted_templates
                    ):
                        result = prediction2map(rxn, prediction, neighbor_weight=90)
                    f.write(f"{idx}\t{result['mapped_rxn']}\t{result['template']}\n")
    return


def main(
    gpu="cuda:0",
    config="default_config.json",
    batch_size=20,
    iteration=1,
    dataset="USPTO_50K",
    try_twice=False,
):
    chemist_name = get_user_name()
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device("cpu")
    print(
        "Testing with device %s, chemist name %s"
        % (device, chemist_name)
    )

    data_dir = str(ROOT / "data" / dataset)
    sample_dir = Path(data_dir) / chemist_name
    output_dir = str(ROOT / "outputs" / dataset / chemist_name)
    model_path = str(
        ROOT
        / "models"
        / dataset
        / chemist_name
        / f"LocalMapper_{iteration}.pth"
    )
    config_path = str(ROOT / "data" / "configs" / config)
    mkdir_p(output_dir)

    node_featurizer, edge_featurizer, mol_to_graph = init_featurizer()
    test_loader = load_dataloader(
        data_dir,
        "test",
        mol_to_graph,
        batch_size=batch_size,
        iteration=iteration,
    )
    model = load_test_model(
        config_path, node_featurizer, edge_featurizer, device, model_path
    )
    accepted_templates, _ = load_fixed_templates(sample_dir, iteration)
    get_atom_map(
        output_dir,
        iteration,
        try_twice,
        accepted_templates,
        model,
        device,
        test_loader,
    )


if __name__ == "__main__":
    fire.Fire(main)
