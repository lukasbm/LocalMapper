from __future__ import annotations

from pathlib import Path
import glob

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Subset

from .dataset import ReactionDataset
from .models import LocalMapper
from .utils import get_configure, init_featurizer as _init_featurizer


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


def init_featurizer():
    return _init_featurizer()


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
    return (
        list(idxs),
        list(rxns),
        list(rgraphs),
        list(pgraphs),
        labels_list,
        masks_list,
        weight_list,
    )


def load_dataloader(
    data_dir,
    mode,
    mol_to_graph,
    batch_size=16,
    iteration=1,
    sample_dir=None,
):
    dataset = ReactionDataset(
        data_dir=data_dir,
        mode=mode,
        mol_to_graph=mol_to_graph,
        iteration=iteration,
        sample_dir=sample_dir,
    )

    if mode == "test":
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


def _build_model(config_path, node_featurizer, edge_featurizer, device):
    exp_config = get_configure(config_path, node_featurizer, edge_featurizer)
    return LocalMapper(
        node_in_feats=exp_config["in_node_feats"],
        edge_in_feats=exp_config["in_edge_feats"],
        node_out_feats=exp_config["node_out_feats"],
        edge_hidden_feats=exp_config["edge_hidden_feats"],
        num_step_message_passing=exp_config["num_step_message_passing"],
        attention_heads=exp_config["attention_heads"],
        attention_layers=exp_config["attention_layers"],
    ).to(device)


def load_train_components(
    config_path,
    node_featurizer,
    edge_featurizer,
    device,
    learning_rate,
    weight_decay,
    patience,
    model_path,
):
    model = _build_model(config_path, node_featurizer, edge_featurizer, device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-4
    )
    stopper = EarlyStopping(mode="lower", patience=patience, filename=model_path)
    return model, loss_criterion, optimizer, scheduler, stopper


def load_test_model(config_path, node_featurizer, edge_featurizer, device, model_path):
    model = _build_model(config_path, node_featurizer, edge_featurizer, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_templates(sample_dir, iteration, files, load_prev=False):
    loaded_templates = set()
    for file in files:
        file_iteration = int(Path(file).stem.split("_")[-1])
        if file_iteration > iteration or (load_prev and file_iteration == iteration):
            continue
        df = pd.read_csv(file)
        loaded_templates.update(df.template.tolist())
    return loaded_templates


def load_fixed_templates(sample_dir, iteration, load_prev=False):
    sample_dir = Path(sample_dir)
    pred_templates = load_templates(
        sample_dir,
        iteration,
        glob.glob(str(sample_dir / "pred_train_*.csv")),
        load_prev,
    )
    conf_templates = load_templates(
        sample_dir,
        iteration,
        glob.glob(str(sample_dir / "conf_pred_*.csv")),
        load_prev,
    )
    accepted_templates = load_templates(
        sample_dir,
        iteration,
        glob.glob(str(sample_dir / "fixed_train_*.csv")),
        load_prev,
    )
    accepted_templates = accepted_templates.union(conf_templates)
    rejected_templates = {
        template for template in pred_templates if template not in accepted_templates
    }
    return accepted_templates, rejected_templates
