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


def load_raw_data(args, fixed=False):
    if fixed:
        df = pd.read_csv("%s/fixed_data.csv" % args["data_dir"])
    else:
        df = pd.read_csv("%s/raw_data.csv" % args["data_dir"])
    trues = []
    for i, (rxn, temp) in enumerate(zip(df["mapped_rxn"], df["template"])):
        trues.append([rxn, temp])
    return trues


def load_train_data(args):
    train_rxns = []
    train_temps = []
    for i, file in enumerate(glob.glob("%s/fixed_train_*.csv" % args["sample_dir"])):
        if i >= args["iteration"]:
            break
        df = pd.read_csv(file)
        train_rxns += df["mapped_rxn"].tolist()
        train_temps += df["template"].tolist()
    return train_rxns, train_temps


def load_prediction(args, load_prev=False, skip=False):
    predictions = []
    if load_prev:
        load_iter = args["iteration"] - 1
    else:
        load_iter = args["iteration"]
    if skip:
        file = "%s/pred_%d_skip.txt" % (args["output_dir"], load_iter)
    else:
        file = "%s/pred_%d_full.txt" % (args["output_dir"], load_iter)
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            predictions.append(line.split("\n")[0].split("\t"))
    return predictions


def sample_reactions(args):
    trues = load_raw_data(args)
    mkdir_p(args["sample_dir"])
    fixed_data = "%s/fixed_train_%d.csv" % (args["sample_dir"], args["iteration"] - 1)
    if os.path.exists(fixed_data):
        if os.path.exists(
            "%s/fixed_train_%d.csv" % (args["sample_dir"], args["iteration"])
        ):
            print(
                "Train data for iteration %d is already sampled and fixed."
                % args["iteration"]
            )
            return
        elif len(glob.glob("%s/fixed_train_*.csv" % args["sample_dir"])) > 0:
            predictions = load_prediction(args, True)
            accepted_templates, rejected_templates = load_fixed_templates(args, True)
            print(
                "%d rejected tempaltes:" % len(rejected_templates), rejected_templates
            )
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
            for template, rxn_idxs in sorted_templates.items():
                freq = len(rxn_idxs)
                sampled_idx = np.random.choice(
                    rxn_idxs, min([freq, args["sample_n"]]), replace=False
                )
                sampled_idxs += list(sampled_idx)
                template_freqs += [freq] * len(sampled_idx)
                if len(sampled_idxs) >= args["sample_limit"]:
                    break
            sampled_rxns = [trues[i][0] for i in sampled_idxs]
            sampled_preds = [predictions[i][1] for i in sampled_idxs]
            sampled_temps = [trues[i][1] for i in sampled_idxs]
            print(
                "Sampled %d reactions showing rxns >= %d times"
                % (len(sampled_idxs), freq)
            )

    else:
        conf_idxs, conf_rxns, conf_temps = [], [], []
        sampled_idxs = list(
            np.random.choice(np.arange(len(trues)), args["sample_limit"], replace=False)
        )
        sampled_preds = ["" for i in sampled_idxs]
        sampled_rxns = [trues[i][0] for i in sampled_idxs]
        sampled_temps = [trues[i][1] for i in sampled_idxs]
        template_freqs = [0 for i in sampled_idxs]
        print("Sampled %d random rxns" % len(sampled_idxs))

    conf_df = pd.DataFrame(
        {"data_idx": conf_idxs, "mapped_rxn": conf_rxns, "template": conf_temps}
    )
    conf_df.to_csv(
        "%s/conf_pred_%d.csv" % (args["sample_dir"], args["iteration"]), index=None
    )
    sample_df = pd.DataFrame(
        {
            "data_idx": sampled_idxs,
            "mapped_rxn": sampled_rxns,
            "template": sampled_temps,
            "freq": template_freqs,
        }
    )
    sample_df.to_csv(
        "%s/pred_train_%d.csv" % (args["sample_dir"], args["iteration"]), index=None
    )
    return


def main(
    dataset="USPTO_50K",
    iteration=1,
    skip=0,
    sample_n=1,
    sample_limit=200,
):
    args = {
        "dataset": dataset,
        "iteration": iteration,
        "skip": skip,
        "sample_n": sample_n,
        "sample_limit": sample_limit,
    }
    args["chemist_name"] = get_user_name(args)
    print("Sampling... chemist name: %s" % (args["chemist_name"]))

    args["data_dir"] = str(ROOT / "data" / args["dataset"])
    args["sample_dir"] = "%s/%s" % (args["data_dir"], args["chemist_name"])
    args["output_dir"] = str(ROOT / "outputs" / args["dataset"] / args["chemist_name"])
    sample_reactions(args)


if __name__ == "__main__":
    fire.Fire(main)
