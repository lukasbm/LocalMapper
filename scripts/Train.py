from pathlib import Path

import fire

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from localmapper.cli_utils import (
    init_featurizer,
    load_train_val_dataloaders_from_items,
    load_train_components,
)
from localmapper.dataset import mkdir_p
from localmapper.active_learning import (
    checkpoint_path,
    confident_pseudo_items,
    training_items_from_annotations,
)


def run_a_train_epoch(
    epoch,
    model,
    data_loader,
    loss_criterion,
    optimizer,
    device,
    learning_rate,
    max_clip,
    print_every,
    num_epochs,
):
    if epoch < 0:  # warmup
        optimizer.param_groups[0]["lr"] = learning_rate * 0.001
    model.train()
    train_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        idxs, rxns, rbg, pbg, labels_list, masks_list, weight_list, records = (
            batch_data
        )
        labels_list, masks_list = (
            [labels.to(device) for labels in labels_list],
            [masks.to(device) for masks in masks_list],
        )
        logits_list = model.score_graphs(rbg, pbg)
        loss = 0
        total_weights = 0
        for logits, labels, masks, weight in zip(
            logits_list, labels_list, masks_list, weight_list
        ):
            loss += (
                weight * (loss_criterion(logits, labels) * (masks != 0)).float().mean()
            )
            total_weights += weight
        loss = loss / total_weights
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_clip)
        optimizer.step()
        train_loss += loss.item()
        if batch_id % print_every == 0:
            print(
                "\repoch %d/%d, batch %d/%d, loss %.4f"
                % (epoch + 1, num_epochs, batch_id + 1, len(data_loader), loss),
                end="",
                flush=True,
            )
    if epoch < 0:
        optimizer.param_groups[0]["lr"] = learning_rate
    return


def run_an_val_epoch(model, data_loader, loss_criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            idxs, rxns, rbg, pbg, labels_list, masks_list, weight_list, records = (
                batch_data
            )
            labels_list, masks_list = (
                [labels.to(device) for labels in labels_list],
                [masks.to(device) for masks in masks_list],
            )
            logits_list = model.score_graphs(rbg, pbg)
            loss = 0
            total_weights = 0
            for logits, labels, masks, weight in zip(
                logits_list, labels_list, masks_list, weight_list
            ):
                loss += (
                    weight
                    * (loss_criterion(logits, labels) * (masks != 0)).float().mean()
                )
                total_weights += weight
            loss = loss / total_weights
            val_loss += loss.item()
    return val_loss / (batch_id + 1)


def main(
    gpu="cuda:0",
    dataset="USPTO_50K",
    model="LocalMapper",
    iteration=1,
    split="train",
    batch_size=16,
    num_epochs=100,
    patience=5,
    max_clip=20,
    learning_rate=1e-3,
    weight_decay=1e-6,
    schedule_step=10,
    print_every=20,
    seed=0,
    val_fraction=0.1,
    init="auto",
    checkpoint=None,
    confident_per_template=100,
):
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device("cpu")
    print(
        "Training with device %s, dataset: %s, iteration: %s"
        % (device, dataset, iteration)
    )

    model_dir = checkpoint_path(ROOT, dataset, model, seed, iteration).parent
    model_path = str(checkpoint_path(ROOT, dataset, model, seed, iteration))
    mkdir_p(model_dir)

    node_featurizer, edge_featurizer, mol_to_graph = init_featurizer()
    train_items = training_items_from_annotations(
        ROOT, dataset, model, seed, iteration
    )
    pseudo_items = confident_pseudo_items(
        ROOT,
        dataset,
        mapper_model,
        seed,
        iteration,
        split,
        confident_per_template,
    )
    all_train_items = train_items + pseudo_items
    if not all_train_items:
        raise ValueError(
            "No active-learning annotations found. Run scripts.Sample first."
        )
    print(
        "Loaded %d annotated and %d confident pseudo-labeled training reactions"
        % (len(train_items), len(pseudo_items))
    )
    train_loader, val_loader = load_train_val_dataloaders_from_items(
        all_train_items,
        mol_to_graph,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
    )
    checkpoint_path_to_load = checkpoint
    init_mode = init
    if init == "auto":
        previous_checkpoint = checkpoint_path(
            ROOT, dataset, model, seed, iteration - 1
        )
        if checkpoint_path_to_load is None and iteration > 1 and previous_checkpoint.exists():
            checkpoint_path_to_load = str(previous_checkpoint)
            init_mode = "checkpoint"
        else:
            init_mode = "scratch"
    mapper_model, loss_criterion, optimizer, scheduler, stopper = load_train_components(
        node_featurizer,
        edge_featurizer,
        mol_to_graph,
        device,
        learning_rate,
        weight_decay,
        patience,
        model_path,
        init=init_mode,
        checkpoint_path=checkpoint_path_to_load,
    )
    run_a_train_epoch(
        -1,
        mapper_model,
        train_loader,
        loss_criterion,
        optimizer,
        device,
        learning_rate,
        max_clip,
        print_every,
        num_epochs,
    )
    for epoch in range(num_epochs):
        run_a_train_epoch(
            epoch,
            mapper_model,
            train_loader,
            loss_criterion,
            optimizer,
            device,
            learning_rate,
            max_clip,
            print_every,
            num_epochs,
        )
        if len(val_loader) == 0:
            val_loss = 1 / (epoch + 1)
        else:
            val_loss = run_an_val_epoch(
                mapper_model, val_loader, loss_criterion, device
            )
            print(", validation loss: %.4f" % val_loss)
        early_stop = stopper.step(val_loss, mapper_model)
        scheduler.step(val_loss)
        if early_stop:
            print("Model is Early stopped!!")
            break


if __name__ == "__main__":
    fire.Fire(main)
