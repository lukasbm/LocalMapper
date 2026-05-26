from pathlib import Path
from collections import defaultdict
import glob
import os

import fire
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from localmapper.cli_utils import get_user_name, load_fixed_templates
from localmapper.dataset import mkdir_p


def reject_template(template):
    return not isinstance(template, str) or template == "mapped" or template == "None"


def load_raw_data(data_dir, fixed=False):
    file_name = "fixed_data.csv" if fixed else "raw_data.csv"
    df = pd.read_csv(Path(data_dir) / file_name)
    trues = []
    for i, (rxn, temp) in enumerate(zip(df["mapped_rxn"], df["template"])):
        trues.append([rxn, temp])
    return trues


def load_train_data(sample_dir, iteration):
    train_rxns = []
    train_temps = []
    for i, file in enumerate(sorted(glob.glob(str(Path(sample_dir) / "fixed_train_*.csv")))):
        if i >= iteration:
            break
        df = pd.read_csv(file)
        train_rxns += df["mapped_rxn"].tolist()
        train_temps += df["template"].tolist()
    return train_rxns, train_temps


def load_prediction(output_dir, iteration, load_prev=False, skip=False):
    predictions = []
    if load_prev:
        load_iter = iteration - 1
    else:
        load_iter = iteration
    if skip:
        file = Path(output_dir) / f"pred_{load_iter}_skip.txt"
    else:
        file = Path(output_dir) / f"pred_{load_iter}_full.txt"
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            predictions.append(line.split("\n")[0].split("\t"))
    return predictions


def sample_reactions(
    data_dir,
    sample_dir,
    output_dir,
    iteration,
    sample_n,
    sample_limit,
    skip=False,
):
    trues = load_raw_data(data_dir)
    mkdir_p(sample_dir)
    fixed_data = Path(sample_dir) / f"fixed_train_{iteration - 1}.csv"
    if os.path.exists(fixed_data):
        current_fixed = Path(sample_dir) / f"fixed_train_{iteration}.csv"
        if os.path.exists(current_fixed):
            print(f"Train data for iteration {iteration} is already sampled and fixed.")
            return
        elif len(glob.glob(str(Path(sample_dir) / "fixed_train_*.csv"))) > 0:
            predictions = load_prediction(output_dir, iteration, load_prev=True, skip=skip)
            accepted_templates, rejected_templates = load_fixed_templates(
                sample_dir, iteration, load_prev=True
            )
            print(f"{len(rejected_templates)} rejected templates:", rejected_templates)
            new_templates = defaultdict(list)
            conf_idxs, conf_rxns, conf_temps = [], [], []
            for prediction in predictions:
                idx, mapped_rxn, temp = prediction
                if reject_template(temp) or temp in rejected_templates:
                    continue
                elif temp in accepted_templates:
                    conf_idxs.append(int(idx))
                    conf_rxns.append(mapped_rxn)
                    conf_temps.append(temp)
                else:
                    new_templates[temp].append(int(idx))

            sorted_templates = {
                k: v for k, v in sorted(new_templates.items(), key=lambda x: -len(x[1]))
            }
            sampled_idxs = []
            template_freqs = []
            freq = 0
            for template, rxn_idxs in sorted_templates.items():
                freq = len(rxn_idxs)
                sampled_idx = np.random.choice(
                    rxn_idxs, min([freq, sample_n]), replace=False
                )
                sampled_idxs += list(sampled_idx)
                template_freqs += [freq] * len(sampled_idx)
                if len(sampled_idxs) >= sample_limit:
                    break
            sampled_rxns = [trues[i][0] for i in sampled_idxs]
            sampled_temps = [trues[i][1] for i in sampled_idxs]
            print(f"Sampled {len(sampled_idxs)} reactions showing rxns >= {freq} times")

    else:
        conf_idxs, conf_rxns, conf_temps = [], [], []
        sampled_idxs = list(
            np.random.choice(np.arange(len(trues)), sample_limit, replace=False)
        )
        sampled_rxns = [trues[i][0] for i in sampled_idxs]
        sampled_temps = [trues[i][1] for i in sampled_idxs]
        template_freqs = [0 for i in sampled_idxs]
        print(f"Sampled {len(sampled_idxs)} random rxns")

    conf_df = pd.DataFrame(
        {"data_idx": conf_idxs, "mapped_rxn": conf_rxns, "template": conf_temps}
    )
    conf_df.to_csv(Path(sample_dir) / f"conf_pred_{iteration}.csv", index=None)
    sample_df = pd.DataFrame(
        {
            "data_idx": sampled_idxs,
            "mapped_rxn": sampled_rxns,
            "template": sampled_temps,
            "freq": template_freqs,
        }
    )
    sample_df.to_csv(Path(sample_dir) / f"pred_train_{iteration}.csv", index=None)
    return


def main(
    dataset="USPTO_50K",
    iteration=1,
    skip=False,
    sample_n=1,
    sample_limit=200,
):
    chemist_name = get_user_name()
    print("Sampling... chemist name: %s" % chemist_name)

    data_dir = str(ROOT / "data" / dataset)
    sample_dir = Path(data_dir) / chemist_name
    output_dir = str(ROOT / "outputs" / dataset / chemist_name)
    sample_reactions(
        data_dir=data_dir,
        sample_dir=sample_dir,
        output_dir=output_dir,
        iteration=iteration,
        sample_n=sample_n,
        sample_limit=sample_limit,
        skip=skip,
    )


if __name__ == "__main__":
    fire.Fire(main)
