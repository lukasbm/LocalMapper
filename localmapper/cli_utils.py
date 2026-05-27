from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from .dataset import (
    ReactionDataset,
    collate_reaction_batch,
    load_reactions,
)
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


def load_dataloader(
    dataset_name,
    split,
    mol_to_graph,
    batch_size=16,
    data_root="data",
    val_fraction=0.1,
    test_fraction=0.1,
    seed=0,
    include_labels=True,
):
    dataset = ReactionDataset.from_name(
        dataset=dataset_name,
        mol_to_graph=mol_to_graph,
        split=split,
        data_root=data_root,
        include_labels=include_labels,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=collate_reaction_batch,
    )


def load_train_val_dataloaders(
    dataset_name,
    mol_to_graph,
    batch_size=16,
    data_root="data",
    val_fraction=0.1,
    test_fraction=0.1,
    seed=0,
):
    train_dataset = ReactionDataset(
        load_reactions(
            dataset_name,
            "train",
            data_root=data_root,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        ),
        mol_to_graph,
        include_labels=True,
    )
    val_dataset = ReactionDataset(
        load_reactions(
            dataset_name,
            "val",
            data_root=data_root,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        ),
        mol_to_graph,
        include_labels=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_reaction_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_batch,
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


def load_dataset_templates(dataset_name, data_root="data", split="train"):
    from .LocalTemplate.template_extractor import extract_from_reaction

    templates = set()
    for item in load_reactions(dataset_name, split, data_root=data_root):
        try:
            templates.add(extract_from_reaction(item["rxn"]))
        except Exception:
            pass
    return templates
