from pathlib import Path
import json
import os
import pickle

import fire
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import torch

from localmapper.active_learning import (
    checkpoint_path,
    output_dir,
    prediction_path,
    verified_templates,
)
from localmapper.cli_utils import (
    init_featurizer,
    load_dataloader,
    load_test_model,
    set_global_seed,
)
from localmapper.dataset import (
    mapping_signature,
    mkdir_p,
)
from localmapper.eequaam import (
    EEQUAAM_BACKEND,
    EEquAAMComparison,
    EEquAAMResult,
    evaluate_eequaam,
    summarize_row_results,
)


def resolve_device(gpu: str) -> torch.device:
    requested = str(gpu)
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if os.environ.get("ALLOW_CPU", "0") == "1":
            print("CUDA requested but unavailable; ALLOW_CPU=1 so testing on CPU.")
            return torch.device("cpu")
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is false. "
            "Fix the CUDA/PyTorch environment, set GPU=cpu, or set ALLOW_CPU=1 for an intentional CPU run."
        )
    return torch.device(requested)


def write_predictions(
    output_path,
    metrics_path,
    accepted_templates,
    model,
    data_loader,
    try_twice,
    eequaam_chunk_size,
    eequaam_base_timeout_seconds,
    eequaam_timeout_seconds_per_reaction,
):
    model.eval()
    selected_count = len(data_loader.dataset) if hasattr(data_loader, "dataset") else None
    rows = []
    correctness_labels = []
    raw_correctness_labels = []
    atom_correct = 0
    atom_total = 0
    invalid_mapping_count = 0
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
            for result, item, input_rxn in zip(results, items, rxns):
                reference_rxns = item.get("mapped_rxns", [item["rxn"]])
                raw_correct = result["mapped_rxn"] in reference_rxns
                atom_metrics = _atom_correspondence_metrics(
                    result["mapped_rxn"], reference_rxns
                )
                atom_correct += atom_metrics["correct"]
                atom_total += atom_metrics["total"]
                invalid_mapping_count += int(atom_metrics["invalid"])
                mapping_score = _mapping_score(result)
                confidence = bool(result["confident"])
                raw_correctness_labels.append(bool(raw_correct))
                rows.append(
                    {
                        "data_idx": item["id"],
                        "split": item["split"],
                        "input_rxn": input_rxn,
                        "ground_truth_mapped_rxn": item["rxn"],
                        "mapped_rxn": result["mapped_rxn"],
                        "_reference_rxns": reference_rxns,
                        "template": result["template"],
                        "confident": result["confident"],
                        "mapping_score": mapping_score,
                        "raw_is_correct": raw_correct,
                        "atom_correct": atom_metrics["correct"],
                        "atom_total": atom_metrics["total"],
                        "atom_accuracy": atom_metrics["accuracy"],
                        "invalid_mapping": atom_metrics["invalid"],
                        "source": item["source"],
                        "num_mappings": item["num_mappings"],
                    }
                )
    eequaam_stats = _apply_eequaam_metrics(
        rows,
        chunk_size=eequaam_chunk_size,
        base_timeout_seconds=eequaam_base_timeout_seconds,
        timeout_seconds_per_reaction=eequaam_timeout_seconds_per_reaction,
    )
    correctness_labels = [bool(row["is_correct"]) for row in rows]
    eequaam_evaluable_labels = [
        bool(row["is_correct"]) for row in rows if row["eequaam_evaluable"]
    ]
    invalid_mapping_count = sum(
        row["eequaam_status"] == "predicted_not_complete_bijective" for row in rows
    )
    status_counts = summarize_row_results(rows)
    strict_accuracy = (
        float(sum(correctness_labels) / len(correctness_labels))
        if correctness_labels
        else 0.0
    )

    mkdir_p(Path(output_path).parent)
    df = pd.DataFrame(rows)
    if "_reference_rxns" in df:
        df = df.drop(columns=["_reference_rxns"])
    df.to_csv(output_path, index=False)

    metrics = {
        "aam": {
            "equiv_exact_match_accuracy": strict_accuracy,
            "strict_equiv_exact_match_accuracy": strict_accuracy,
            "raw_exact_match_accuracy": (
                float(sum(raw_correctness_labels) / len(raw_correctness_labels))
                if raw_correctness_labels
                else 0.0
            ),
            "atom_accuracy": float(atom_correct / atom_total) if atom_total else 0.0,
            "atom_correct": int(atom_correct),
            "atom_total": int(atom_total),
            "invalid_mapping_count": int(invalid_mapping_count),
            "evaluable_count": int(sum(row["eequaam_evaluable"] for row in rows)),
            "unevaluable_count": int(
                sum(not row["eequaam_evaluable"] for row in rows)
            ),
            "eequaam_failed_count": int(status_counts.get("eequaam_failed", 0)),
            "evaluable_accuracy": (
                float(sum(eequaam_evaluable_labels) / len(eequaam_evaluable_labels))
                if eequaam_evaluable_labels
                else 0.0
            ),
            "eequaam_status_counts": status_counts,
            "selected_count": int(selected_count) if selected_count is not None else None,
            "predicted_count": int(len(rows)),
            "total": int(len(correctness_labels)),
            **eequaam_stats,
            "equivalence_backend": EEQUAAM_BACKEND,
        },
        "template_library": {
            "accepted_template_count": (
                int(len(accepted_templates))
                if accepted_templates is not None
                else None
            ),
            "has_template_filter": accepted_templates is not None,
        },
    }
    mkdir_p(Path(metrics_path).parent)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def _atom_correspondence_metrics(predicted_rxn, reference_rxns):
    predicted = mapping_signature(predicted_rxn)
    references = [
        signature for rxn in reference_rxns if (signature := mapping_signature(rxn))
    ]
    if not references:
        return {"correct": 0, "total": 0, "accuracy": 0.0, "invalid": True}
    fallback_total = max(len(reference_pairs) for _, reference_pairs in references)
    if predicted is None:
        return {
            "correct": 0,
            "total": int(fallback_total),
            "accuracy": 0.0,
            "invalid": True,
        }

    predicted_pattern, predicted_pairs = predicted
    predicted_map = dict(predicted_pairs)
    best_correct = 0
    best_total = 0
    for reference_pattern, reference_pairs in references:
        if reference_pattern != predicted_pattern:
            continue
        reference_map = dict(reference_pairs)
        total = len(reference_map)
        correct = sum(
            predicted_map.get(product_idx) == reactant_idx
            for product_idx, reactant_idx in reference_map.items()
        )
        if correct > best_correct:
            best_correct = correct
            best_total = total

    if best_total == 0:
        best_total = fallback_total

    return {
        "correct": int(best_correct),
        "total": int(best_total),
        "accuracy": float(best_correct / best_total) if best_total else 0.0,
        "invalid": False,
    }


def _apply_eequaam_metrics(
    rows,
    *,
    chunk_size,
    base_timeout_seconds,
    timeout_seconds_per_reaction,
):
    requested_reference_comparison_count = 0
    signature_short_circuit_count = 0
    precomputed_results: dict[str, EEquAAMResult] = {}
    comparisons = []
    comparison_to_row = {}
    for row_index, row in enumerate(tqdm(rows, desc="Preparing EEquAAM refs")):
        predicted_signature = mapping_signature(row["mapped_rxn"])
        unique_references = []
        seen_reference_keys = set()
        for reference_rxn in row["_reference_rxns"]:
            requested_reference_comparison_count += 1
            reference_signature = mapping_signature(reference_rxn)
            if (
                predicted_signature is not None
                and reference_signature is not None
                and predicted_signature == reference_signature
            ):
                comparison_id = f"row{row_index}__signature_match"
                comparison_to_row[comparison_id] = row_index
                precomputed_results[comparison_id] = EEquAAMResult(
                    True,
                    "ok",
                    "exact mapping-signature match",
                )
                signature_short_circuit_count += 1
                unique_references = []
                break

            reference_key = reference_signature or reference_rxn
            if reference_key in seen_reference_keys:
                continue
            seen_reference_keys.add(reference_key)
            unique_references.append(reference_rxn)

        for reference_index, reference_rxn in enumerate(unique_references):
            comparison_id = f"row{row_index}__ref{reference_index}"
            comparison_to_row[comparison_id] = row_index
            comparisons.append(
                EEquAAMComparison(
                    comparison_id=comparison_id,
                    predicted_rxn=row["mapped_rxn"],
                    reference_rxn=reference_rxn,
                    row_index=row_index,
                )
            )

    comparison_results = dict(precomputed_results)
    comparison_results.update(
        evaluate_eequaam(
            comparisons,
            eequaam_script=ROOT / "EEquAAM.py",
            chunk_size=chunk_size,
            base_timeout_seconds=base_timeout_seconds,
            timeout_seconds_per_reaction=timeout_seconds_per_reaction,
        )
    )

    row_results = [[] for _ in rows]
    for comparison_id, result in comparison_results.items():
        row_results[comparison_to_row[comparison_id]].append(result)

    for row, results in zip(rows, row_results):
        equivalent_results = [result for result in results if result.equivalent]
        ok_results = [result for result in results if result.status == "ok"]
        status_counts: dict[str, int] = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        row["is_correct"] = bool(equivalent_results)
        row["eequaam_is_correct"] = bool(equivalent_results)
        row["eequaam_evaluable"] = bool(ok_results)
        row["eequaam_status"] = _row_eequaam_status(results)
        row["eequaam_status_counts"] = json.dumps(status_counts, sort_keys=True)
        row["eequaam_details"] = json.dumps(
            [
                {"status": result.status, "detail": result.detail}
                for result in results
                if result.status != "ok" and result.detail
            ],
            sort_keys=True,
        )

    return {
        "eequaam_requested_reference_comparison_count": int(
            requested_reference_comparison_count
        ),
        "eequaam_subprocess_comparison_count": int(len(comparisons)),
        "eequaam_signature_short_circuit_count": int(signature_short_circuit_count),
    }


def _row_eequaam_status(results):
    if not results:
        return "no_reference_comparison"
    if any(result.equivalent for result in results):
        return "ok_equivalent"
    if any(result.status == "ok" for result in results):
        return "ok_non_equivalent"
    if any(result.status == "predicted_not_complete_bijective" for result in results):
        return "predicted_not_complete_bijective"
    if any(result.status == "eequaam_failed" for result in results):
        return "eequaam_failed"
    return results[0].status


def _mapping_score(result):
    mapper = result.get("mapper")
    map_steps = getattr(mapper, "map_steps", None)
    if map_steps is None or map_steps.empty or "score" not in map_steps.index:
        return 0.0
    scores = pd.to_numeric(map_steps.loc["score"], errors="coerce").dropna()
    if scores.empty:
        return 0.0
    return float(scores.mean())


def _flatten_metrics(prefix, value, out):
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_metrics(next_prefix, nested, out)
    else:
        out[prefix] = value


def save_run_metrics_summary(run_dir):
    run_dir = Path(run_dir)
    predictions_dir = run_dir / "predictions"
    rows = []
    for metrics_path in sorted(predictions_dir.glob("metrics_*.json")):
        parts = metrics_path.stem.split("_")
        split = parts[1] if len(parts) >= 3 else None
        iteration = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        row = {
            "split": split,
            "iteration": iteration,
            "metrics_path": str(metrics_path),
        }
        _flatten_metrics("", metrics, row)
        rows.append(row)

    mkdir_p(run_dir)
    summary_json = run_dir / "metrics_summary.json"
    summary_csv = run_dir / "metrics_summary.csv"
    with summary_json.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    return summary_json, summary_csv


def _valid_template_values(values) -> set[str]:
    return {
        str(value)
        for value in values
        if not pd.isna(value) and str(value).strip() and str(value) != "<NA>"
    }


def load_template_library(path: str | Path | None) -> set[str]:
    if path is None or str(path).strip() == "":
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Template library not found: {path}")

    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            values = pickle.load(handle)
        if isinstance(values, dict):
            values = values.keys()
        return _valid_template_values(values)

    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        if "template" in frame:
            return _valid_template_values(frame["template"])
        if frame.shape[1] == 0:
            return set()
        return _valid_template_values(frame.iloc[:, 0])

    with path.open() as handle:
        return _valid_template_values(line.strip() for line in handle)


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
    template_library=None,
    eequaam_chunk_size=100,
    eequaam_base_timeout_seconds=30.0,
    eequaam_timeout_seconds_per_reaction=5.0,
):
    set_global_seed(seed)
    device = resolve_device(gpu)
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
    print(
        "Evaluation split %s has %d reactions after LocalMapper dataset filters"
        % (split, len(test_loader.dataset))
    )
    mapper_model = load_test_model(
        node_featurizer, edge_featurizer, mol_to_graph, device, model_path
    )
    accepted_templates = verified_templates(
        ROOT, dataset, model, seed, through_iteration=iteration
    )
    accepted_templates.update(load_template_library(template_library))
    metrics = write_predictions(
        output_path,
        metrics_path,
        accepted_templates,
        mapper_model,
        test_loader,
        try_twice,
        int(eequaam_chunk_size),
        float(eequaam_base_timeout_seconds),
        float(eequaam_timeout_seconds_per_reaction),
    )
    summary_json, summary_csv = save_run_metrics_summary(
        output_dir(ROOT, dataset, model, seed)
    )
    print(
        "AAM {equivalence_backend} StrictEquivExact: {strict_equiv_exact_match_accuracy:.4f}, EvaluableExact: {evaluable_accuracy:.4f}, AtomAcc: {atom_accuracy:.4f}, RawExact: {raw_exact_match_accuracy:.4f}, Invalid: {invalid_mapping_count}/{predicted_count}".format(
            **metrics["aam"]
        )
    )
    print(
        "EEquAAM denominator: predicted={predicted_count}, evaluable={evaluable_count}, unevaluable={unevaluable_count}, failed={eequaam_failed_count}".format(
            **metrics["aam"]
        )
    )
    print(f"EEquAAM statuses: {metrics['aam']['eequaam_status_counts']}")
    print(f"Saved predictions to {output_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved run metrics summary to {summary_csv} and {summary_json}")


if __name__ == "__main__":
    fire.Fire(main)
