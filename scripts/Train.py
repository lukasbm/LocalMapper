from pathlib import Path

import fire

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from localmapper.cli_utils import (
    init_featurizer,
    load_train_val_dataloaders,
    load_train_components,
)
from localmapper.dataset import mkdir_p


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
    test_fraction=0.1,
    init="scratch",
    checkpoint=None,
):
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device("cpu")
    print("Training with device %s, dataset: %s" % (device, dataset))

    data_root = str(ROOT / "data")
    model_dir = str(ROOT / "models" / dataset)
    model_path = str(Path(model_dir) / "LocalMapper.pth")
    mkdir_p(model_dir)

    node_featurizer, edge_featurizer, mol_to_graph = init_featurizer()
    train_loader, val_loader = load_train_val_dataloaders(
        dataset,
        mol_to_graph,
        batch_size=batch_size,
        data_root=data_root,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    model, loss_criterion, optimizer, scheduler, stopper = load_train_components(
        node_featurizer,
        edge_featurizer,
        mol_to_graph,
        device,
        learning_rate,
        weight_decay,
        patience,
        model_path,
        init=init,
        checkpoint_path=checkpoint,
    )
    run_a_train_epoch(
        -1,
        model,
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
            model,
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
            val_loss = run_an_val_epoch(model, val_loader, loss_criterion, device)
            print(", validation loss: %.4f" % val_loss)
        early_stop = stopper.step(val_loss, model)
        scheduler.step(val_loss)
        if early_stop:
            print("Model is Early stopped!!")
            break


if __name__ == "__main__":
    fire.Fire(main)
