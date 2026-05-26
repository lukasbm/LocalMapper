from __future__ import annotations

import getpass
import glob
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Subset

from .dataset import ReactionDataset, mkdir_p
from .models import LocalMapper
from .utils import get_configure, init_featurizer as _init_featurizer

ROOT = Path(__file__).resolve().parents[1]


class EarlyStopping:
    def __init__(self, mode="lower", patience=5, filename="checkpoint.pth"):
        self.mode = mode
        self.patience = patience
        self.filename = filename
        self.best_score = None
        self.num_bad_epochs = 0
        self.sign = 1 if mode == "higher" else -1

    def step(self, value, model):
        score = self.sign * value
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.num_bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, self.filename)
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def get_user_name(args=None):
    manual_dir = ROOT / "manual"
    if manual_dir.exists():
        user_files = sorted(manual_dir.glob("*.user"))
        if user_files:
            return user_files[0].stem
    return getpass.getuser()


def init_featurizer(args):
    node_featurizer, edge_featurizer, graph_function = _init_featurizer()
    args["node_featurizer"] = node_featurizer
    args["edge_featurizer"] = edge_featurizer
    args["mol_to_graph"] = graph_function
    return args


def _collate_batch(batch):
    idxs, rxns, rgraphs, pgraphs, labels, weights = zip(*batch)
    labels_list = [
        torch.as_tensor(label, dtype=torch.long)
        if not torch.is_tensor(label)
        else label
        for label in labels
    ]
    masks_list = [torch.ones_like(label, dtype=torch.long) for label in labels_list]
    weight_list = [float(weight) for weight in weights]
    return list(idxs), list(rxns), list(rgraphs), list(pgraphs), labels_list, masks_list, weight_list


def load_dataloader(args, test=False):
    dataset = ReactionDataset(args)
    batch_size = int(args.get("batch_size", 16))

    if test:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
        )

    train_subset = Subset(dataset, dataset.train_idx)
    val_subset = Subset(dataset, dataset.val_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    return train_loader, val_loader


def load_train_components(args):
    exp_config = get_configure(
        args["config_path"], args["node_featurizer"], args["edge_featurizer"]
    )
    model = LocalMapper(
        node_in_feats=exp_config["in_node_feats"],
        edge_in_feats=exp_config["in_edge_feats"],
        node_out_feats=exp_config["node_out_feats"],
        edge_hidden_feats=exp_config["edge_hidden_feats"],
        num_step_message_passing=exp_config["num_step_message_passing"],
        attention_heads=exp_config["attention_heads"],
        attention_layers=exp_config["attention_layers"],
    ).to(args["device"])

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args["learning_rate"],
        weight_decay=args["weight_decay"],
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-4
    )
    stopper = EarlyStopping(
        mode="lower", patience=args["patience"], filename=args["model_path"]
    )
    return model, loss_criterion, optimizer, scheduler, stopper


def load_test_model(args):
    exp_config = get_configure(
        args["config_path"], args["node_featurizer"], args["edge_featurizer"]
    )
    model = LocalMapper(
        node_in_feats=exp_config["in_node_feats"],
        edge_in_feats=exp_config["in_edge_feats"],
        node_out_feats=exp_config["node_out_feats"],
        edge_hidden_feats=exp_config["edge_hidden_feats"],
        num_step_message_passing=exp_config["num_step_message_passing"],
        attention_heads=exp_config["attention_heads"],
        attention_layers=exp_config["attention_layers"],
    ).to(args["device"])
    checkpoint = torch.load(args["model_path"], map_location=args["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_templates(args, files, load_prev):
    loaded_templates = set()
    for file in files:
        iteration = int(file.split("_")[-1].split(".")[0])
        if iteration > args["iteration"] or (
            load_prev and iteration == args["iteration"]
        ):
            continue
        df = pd.read_csv(file)
        for template in df.template:
            loaded_templates.add(template)
    return loaded_templates


def load_fixed_templates(args, load_prev=False):
    if "sample_dir" not in args:
        args["sample_dir"] = "%s/%s" % (args["data_dir"], args["chemist_name"])
    pred_templates = load_templates(
        args, glob.glob("%s/pred_train_*.csv" % args["sample_dir"]), load_prev
    )
    conf_templates = load_templates(
        args, glob.glob("%s/conf_pred_*.csv" % args["sample_dir"]), load_prev
    )
    accepted_templates = load_templates(
        args, glob.glob("%s/fixed_train_*.csv" % args["sample_dir"]), load_prev
    )
    accepted_templates = accepted_templates.union(conf_templates)
    rejected_templates = set(
        [template for template in pred_templates if template not in accepted_templates]
    )
    return accepted_templates, rejected_templates
