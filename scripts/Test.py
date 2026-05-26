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


def get_atom_map(args, model, data_loader):
    model.eval()
    file_path = str(Path(args["output_dir"]) / f"pred_{args['iteration']}.txt")
    accepted_templates, _ = load_fixed_templates(args)
    with open(file_path, "w") as f:
        f.write("Reaction_id\tMapped_reaction\tTemplate\n")
        with torch.no_grad():
            for batch_data in tqdm(
                data_loader, total=len(data_loader), desc="Predicting AAM..."
            ):
                idxs, rxns, rbg, pbg, _, _, _ = batch_data
                logits_list = predict(args, model, rbg, pbg)
                for rxn, logits, idx in zip(rxns, logits_list, idxs):
                    prediction = torch.softmax(logits, dim=1).cpu().numpy()
                    result = prediction2map(rxn, prediction)
                    if (
                        args["try_twice"]
                        and result["template"] not in accepted_templates
                    ):
                        result = prediction2map(rxn, prediction, neighbor_weight=90)
                    f.write(
                        "%s\t%s\t%s\n" % (idx, result["mapped_rxn"], result["template"])
                    )
    return


def main(
    gpu="cuda:0",
    config="default_config.json",
    batch_size=20,
    iteration=1,
    dataset="USPTO_50K",
    try_twice=False,
):
    args = {
        "gpu": gpu,
        "config": config,
        "batch_size": batch_size,
        "iteration": iteration,
        "dataset": dataset,
        "try_twice": try_twice,
    }
    args["mode"] = "test"
    args["chemist_name"] = get_user_name(args)
    args["device"] = (
        torch.device(args["gpu"]) if torch.cuda.is_available() else torch.device("cpu")
    )
    print(
        "Testing with device %s, chemist name %s"
        % (args["device"], args["chemist_name"])
    )

    args["data_dir"] = str(ROOT / "data" / args["dataset"])
    args["output_dir"] = str(ROOT / "outputs" / args["dataset"] / args["chemist_name"])
    args["model_path"] = str(
        ROOT
        / "models"
        / args["dataset"]
        / args["chemist_name"]
        / f"LocalMapper_{args['iteration']}.pth"
    )
    args["config_path"] = str(ROOT / "data" / "configs" / args["config"])
    mkdir_p(args["output_dir"])

    args = init_featurizer(args)
    test_loader = load_dataloader(args, test=True)
    model = load_test_model(args)
    get_atom_map(args, model, test_loader)


if __name__ == "__main__":
    fire.Fire(main)
