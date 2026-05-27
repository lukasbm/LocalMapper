from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from .dataset import (
    ReactionDataset,
    collate_reaction_batch,
    create_reaction_dataset,
    select_split,
)
from .models import LocalMapper
from .utils import init_featurizer as _init_featurizer


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
    full_dataset = create_reaction_dataset(
        dataset=dataset_name,
        mol_to_graph=mol_to_graph,
        data_root=data_root,
        include_labels=include_labels,
    )
    dataset = ReactionDataset.from_items(
        select_split(
            full_dataset.items,
            split,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        ),
        mol_to_graph,
        include_labels=include_labels,
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
    full_dataset = create_reaction_dataset(
        dataset_name,
        mol_to_graph,
        data_root=data_root,
        include_labels=True,
    )
    train_dataset = ReactionDataset.from_items(
        select_split(
            full_dataset.items,
            "train",
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        ),
        mol_to_graph,
        include_labels=True,
    )
    val_dataset = ReactionDataset.from_items(
        select_split(
            full_dataset.items,
            "val",
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


def load_train_val_dataloaders_from_items(
    items,
    mol_to_graph,
    batch_size=16,
    val_fraction=0.1,
    seed=0,
):
    import numpy as np

    items = list(items)
    if not items:
        raise ValueError("No training items available")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_val = int(round(len(items) * val_fraction))
    val_ids = set(order[:n_val])

    train_items = [item for i, item in enumerate(items) if i not in val_ids]
    val_items = [item for i, item in enumerate(items) if i in val_ids]
    if not train_items:
        train_items, val_items = items, []

    train_dataset = ReactionDataset.from_items(
        train_items,
        mol_to_graph,
        include_labels=True,
    )
    val_dataset = ReactionDataset.from_items(
        val_items,
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


def _build_model(node_featurizer, edge_featurizer, mol_to_graph, device):
    return LocalMapper(
        node_in_feats=node_featurizer.feat_size(),
        edge_in_feats=edge_featurizer.feat_size(),
        graph_function=mol_to_graph,
        device=device,
    )


def load_train_components(
    node_featurizer,
    edge_featurizer,
    mol_to_graph,
    device,
    learning_rate,
    weight_decay,
    patience,
    model_path,
    init="scratch",
    checkpoint_path=None,
):
    model = _build_model(node_featurizer, edge_featurizer, mol_to_graph, device)
    if init == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when init='checkpoint'")
        model.load_checkpoint(checkpoint_path, device=device)
    elif init != "scratch":
        raise ValueError("init must be 'scratch' or 'checkpoint'")
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


def load_test_model(node_featurizer, edge_featurizer, mol_to_graph, device, model_path):
    model = _build_model(node_featurizer, edge_featurizer, mol_to_graph, device)
    model.load_checkpoint(model_path, device=device)
    return model
