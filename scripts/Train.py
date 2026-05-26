from pathlib import Path

import fire

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from localmapper.cli_utils import (
    get_user_name,
    init_featurizer,
    load_dataloader,
    load_train_components,
)
from localmapper.dataset import mkdir_p
from localmapper.utils import predict


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    if epoch < 0:  # warmup
        optimizer.param_groups[0]["lr"] = args["learning_rate"] * 0.001
    model.train()
    train_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        idxs, rxns, rbg, pbg, labels_list, masks_list, weight_list = batch_data
        labels_list, masks_list = (
            [labels.to(args["device"]) for labels in labels_list],
            [masks.to(args["device"]) for masks in masks_list],
        )
        logits_list = predict(args, model, rbg, pbg)
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
        nn.utils.clip_grad_norm_(model.parameters(), args["max_clip"])
        optimizer.step()
        train_loss += loss.item()
        if batch_id % args["print_every"] == 0:
            print(
                "\repoch %d/%d, batch %d/%d, loss %.4f"
                % (epoch + 1, args["num_epochs"], batch_id + 1, len(data_loader), loss),
                end="",
                flush=True,
            )
    if epoch < 0:
        optimizer.param_groups[0]["lr"] = args["learning_rate"]
    return


def run_an_val_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            idxs, rxns, rbg, pbg, labels_list, masks_list, weight_list = batch_data
            labels_list, masks_list = (
                [labels.to(args["device"]) for labels in labels_list],
                [masks.to(args["device"]) for masks in masks_list],
            )
            logits_list = predict(args, model, rbg, pbg)
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
    config="default_config.json",
    batch_size=16,
    num_epochs=100,
    patience=5,
    iteration=1,
    max_clip=20,
    learning_rate=1e-3,
    weight_decay=1e-6,
    schedule_step=10,
    print_every=20,
):
    args = {
        "gpu": gpu,
        "dataset": dataset,
        "config": config,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "iteration": iteration,
        "max_clip": max_clip,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "schedule_step": schedule_step,
        "print_every": print_every,
    }
    args["mode"] = "train"
    args["chemist_name"] = get_user_name(args)
    args["device"] = (
        torch.device(args["gpu"]) if torch.cuda.is_available() else torch.device("cpu")
    )
    print(
        "Trianing with device %s, chemist name: %s"
        % (args["device"], args["chemist_name"])
    )

    model_name = "LocalMapper_%d.pth" % (args["iteration"])
    args["data_dir"] = str(ROOT / "data" / args["dataset"])
    args["model_dir"] = str(ROOT / "models" / args["dataset"] / args["chemist_name"])
    args["model_path"] = str(Path(args["model_dir"]) / model_name)
    mkdir_p(args["model_dir"])

    args = init_featurizer(args)
    args["config_path"] = str(ROOT / "data" / "configs" / args["config"])
    train_loader, val_loader = load_dataloader(args)
    model, loss_criterion, optimizer, scheduler, stopper = load_train_components(args)
    run_a_train_epoch(args, -1, model, train_loader, loss_criterion, optimizer)
    for epoch in range(args["num_epochs"]):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        if len(val_loader) == 0:
            val_loss = 1 / (epoch + 1)
        else:
            val_loss = run_an_val_epoch(args, model, val_loader, loss_criterion)
            print(", validation loss: %.4f" % val_loss)
        early_stop = stopper.step(val_loss, model)
        scheduler.step(val_loss)
        if early_stop:
            print("Model is Early stopped!!")
            break


if __name__ == "__main__":
    fire.Fire(main)
