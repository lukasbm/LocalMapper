from pathlib import Path
import json

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import torch

from localmapper.active_learning import (
    checkpoint_path,
    prediction_path,
    verified_templates,
)
from localmapper.cli_utils import init_featurizer, load_dataloader, load_test_model
from localmapper.dataset import mkdir_p


def write_predictions(
    output_path,
    metrics_path,
    accepted_templates,
    model,
    data_loader,
    try_twice,
):
    model.eval()
    rows = []
    correctness_labels = []
    mapping_scores = []
    confidence_predictions = []
    with torch.no_grad():
        for batch_data in tqdm(
            data_loader, total=len(data_loader), desc="Predicting AAM..."
        ):
            idxs, rxns, rbg, pbg, _, _, _, items = batch_data
            logits_list = model.score_graphs(rbg, pbg)
            results = model.map_scores(
                rxns,
                logits_list,
                accepted_templates=accepted_templates,
                try_twice=try_twice,
                return_dict=True,
            )
            for result, item in zip(results, items):
                correct = result["mapped_rxn"] == item["rxn"]
                mapping_score = _mapping_score(result)
                confidence = bool(result["confident"])
                correctness_labels.append(bool(correct))
                mapping_scores.append(mapping_score)
                confidence_predictions.append(confidence)
                rows.append(
                    {
                        "data_idx": item["id"],
                        "split": item["split"],
                        "ground_truth_mapped_rxn": item["rxn"],
                        "mapped_rxn": result["mapped_rxn"],
                        "template": result["template"],
                        "confident": result["confident"],
                        "mapping_score": mapping_score,
                        "is_correct": correct,
                        "source": item["source"],
                        "num_mappings": item["num_mappings"],
                    }
                )
    mkdir_p(Path(output_path).parent)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    metrics = _compute_metrics(
        np.asarray(correctness_labels, dtype=bool),
        np.asarray(mapping_scores, dtype=float),
    )
    template_confidence_metrics = _compute_confidence_metrics(
        np.asarray(correctness_labels, dtype=bool),
        np.asarray(confidence_predictions, dtype=bool),
    )
    metrics["template_confidence"] = template_confidence_metrics
    mkdir_p(Path(metrics_path).parent)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def _mapping_score(result):
    mapper = result.get("mapper")
    map_steps = getattr(mapper, "map_steps", None)
    if map_steps is None or map_steps.empty or "score" not in map_steps.index:
        return 0.0
    scores = pd.to_numeric(map_steps.loc["score"], errors="coerce").dropna()
    if scores.empty:
        return 0.0
    return float(scores.mean())


def _compute_confidence_metrics(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    total = int(len(y_true))

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(mcc_denom) if mcc_denom else 0.0

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "mcc": float(mcc),
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total": total,
        },
    }


def _compute_metrics(y_true, y_score):
    y_true = y_true.astype(bool)
    y_score = y_score.astype(float)
    threshold, y_pred = _best_threshold(y_true, y_score)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    total = int(len(y_true))

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(mcc_denom) if mcc_denom else 0.0

    order = np.argsort(-y_score)
    sorted_true = y_true[order].astype(int)
    positives = int(sorted_true.sum())
    if positives == 0:
        ap = 0.0
    else:
        cumulative_tp = np.cumsum(sorted_true)
        precision_at_k = cumulative_tp / (np.arange(len(sorted_true)) + 1)
        ap = float((precision_at_k * sorted_true).sum() / positives)

    return {
        "ap": float(ap),
        "mcc": float(mcc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total": total,
        },
        "exact_match_accuracy": float(y_true.mean()) if total else 0.0,
    }


def _best_threshold(y_true, y_score):
    if len(y_true) == 0:
        return 0.5, np.zeros(0, dtype=bool)

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order].astype(bool)

    positives = int(sorted_true.sum())
    if positives == 0:
        return float(sorted_scores[0]) if len(sorted_scores) else 0.5, np.zeros(
            len(y_true), dtype=bool
        )

    tp_cum = np.cumsum(sorted_true)
    fp_cum = np.cumsum(~sorted_true)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / positives
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )

    best_idx = int(np.argmax(f1))
    threshold = float(sorted_scores[best_idx])
    y_pred = y_score >= threshold
    return threshold, y_pred


def main(
    gpu="cuda:0",
    batch_size=20,
    dataset="USPTO_50K",
    model="LocalMapper",
    iteration=1,
    split="train",
    try_twice=True,
    seed=0,
    val_fraction=0.1,
    test_fraction=0.1,
    checkpoint=None,
):
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device("cpu")
    print(
        "Testing with device %s, dataset %s, split %s, iteration %s"
        % (device, dataset, split, iteration)
    )

    model_path = checkpoint or str(
        checkpoint_path(ROOT, dataset, model, seed, iteration)
    )
    output_path = prediction_path(ROOT, dataset, model, seed, split, iteration)
    metrics_path = output_path.with_name(f"metrics_{split}_{iteration}.json")

    node_featurizer, edge_featurizer, mol_to_graph = init_featurizer()
    test_loader = load_dataloader(
        dataset,
        split,
        mol_to_graph,
        batch_size=batch_size,
        data_root=ROOT / "data",
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        include_labels=False,
    )
    mapper_model = load_test_model(
        node_featurizer, edge_featurizer, mol_to_graph, device, model_path
    )
    accepted_templates = verified_templates(
        ROOT, dataset, model, seed, through_iteration=iteration
    )
    metrics = write_predictions(
        output_path,
        metrics_path,
        accepted_templates,
        mapper_model,
        test_loader,
        try_twice,
    )
    print(
        "AP: {ap:.4f}, MCC: {mcc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Exact: {exact_match_accuracy:.4f}".format(
            **metrics
        )
    )
    print(f"Saved predictions to {output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    fire.Fire(main)
