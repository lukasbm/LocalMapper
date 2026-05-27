from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .LocalTemplate.template_extractor import extract_from_reaction
from .dataset import load_reactions, mkdir_p


def run_name(model_name: str, seed: int) -> str:
    return f"{model_name}_seed{seed}"


def output_dir(root: str | Path, dataset: str, model_name: str, seed: int) -> Path:
    return Path(root) / "outputs" / dataset / run_name(model_name, seed)


def checkpoint_path(
    root: str | Path, dataset: str, model_name: str, seed: int, iteration: int
) -> Path:
    return output_dir(root, dataset, model_name, seed) / f"iteration_{iteration}.pth"


def annotations_dir(root: str | Path, dataset: str, model_name: str, seed: int) -> Path:
    return output_dir(root, dataset, model_name, seed) / "annotations"


def predictions_dir(root: str | Path, dataset: str, model_name: str, seed: int) -> Path:
    return output_dir(root, dataset, model_name, seed) / "predictions"


def templates_dir(root: str | Path, dataset: str, model_name: str, seed: int) -> Path:
    return output_dir(root, dataset, model_name, seed) / "templates"


def annotation_path(
    root: str | Path, dataset: str, model_name: str, seed: int, iteration: int
) -> Path:
    return (
        annotations_dir(root, dataset, model_name, seed)
        / f"annotations_{iteration}.csv"
    )


def prediction_path(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    split: str,
    iteration: int,
) -> Path:
    return (
        predictions_dir(root, dataset, model_name, seed)
        / f"pred_{split}_{iteration}.csv"
    )


def template_library_path(
    root: str | Path, dataset: str, model_name: str, seed: int, iteration: int
) -> Path:
    return (
        templates_dir(root, dataset, model_name, seed)
        / f"verified_templates_{iteration}.csv"
    )


def template_for(rxn: str) -> str | None:
    try:
        return extract_from_reaction(rxn)
    except Exception:
        return None


def load_dataset_split(
    dataset: str,
    split: str,
    *,
    data_root: str | Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> list[dict[str, Any]]:
    return load_reactions(
        dataset,
        split=split,
        data_root=data_root,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def load_annotations(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    through_iteration: int,
) -> pd.DataFrame:
    frames = []
    for iteration in range(1, through_iteration + 1):
        path = annotation_path(root, dataset, model_name, seed, iteration)
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"data_idx": str}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def verified_templates(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    through_iteration: int,
) -> set[str]:
    annotations = load_annotations(root, dataset, model_name, seed, through_iteration)
    if annotations.empty or "template" not in annotations:
        return set()
    return set(annotations["template"].dropna().astype(str))


def _item_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_idx": [item["id"] for item in items],
            "split": [item["split"] for item in items],
            "mapped_rxn": [item["rxn"] for item in items],
            "template": [template_for(item["rxn"]) for item in items],
            "source": [item["source"] for item in items],
            "num_mappings": [item["num_mappings"] for item in items],
        }
    )


def sample_annotations(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    iteration: int,
    split: str,
    sample_limit: int,
    *,
    data_root: str | Path,
    val_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    items = load_dataset_split(
        dataset,
        split,
        data_root=data_root,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    all_items = _item_frame(items)
    if all_items.empty:
        return all_items

    previous_annotations = load_annotations(
        root, dataset, model_name, seed, iteration - 1
    )
    annotated_ids = (
        set(previous_annotations["data_idx"].astype(str))
        if not previous_annotations.empty
        else set()
    )
    available = all_items[~all_items["data_idx"].astype(str).isin(annotated_ids)].copy()
    if available.empty:
        return available

    rng = np.random.default_rng(seed + iteration)
    sample_limit = min(sample_limit, len(available))
    selected_ids: list[str] = []
    selection_reasons: dict[str, str] = {}
    predicted_templates: dict[str, str | None] = {}

    if iteration > 1:
        pred_path = prediction_path(
            root, dataset, model_name, seed, split, iteration - 1
        )
        known_templates = verified_templates(
            root, dataset, model_name, seed, iteration - 1
        )
        if pred_path.exists() and known_templates:
            predictions = pd.read_csv(pred_path, dtype={"data_idx": str})
            predictions = predictions[
                predictions["data_idx"].isin(set(available["data_idx"].astype(str)))
            ].copy()
            predictions["template"] = predictions["template"].astype("string")
            uncertain = predictions[
                ~predictions["template"].fillna("").isin(known_templates)
            ].copy()
            if not uncertain.empty:
                groups = uncertain.groupby("template", dropna=False)
                group_keys = sorted(
                    groups.groups.keys(),
                    key=lambda key: len(groups.groups[key]),
                    reverse=True,
                )
                for key in group_keys:
                    if len(selected_ids) >= sample_limit:
                        break
                    group = groups.get_group(key)
                    chosen = group.sample(
                        n=1, random_state=int(rng.integers(0, 2**31 - 1))
                    ).iloc[0]
                    data_idx = str(chosen["data_idx"])
                    selected_ids.append(data_idx)
                    selection_reasons[data_idx] = "uncertain_template"
                    predicted_templates[data_idx] = (
                        None if pd.isna(chosen["template"]) else str(chosen["template"])
                    )

    if len(selected_ids) < sample_limit:
        remaining = available[~available["data_idx"].astype(str).isin(selected_ids)]
        fill = remaining.sample(
            n=sample_limit - len(selected_ids),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        fill_ids = fill["data_idx"].astype(str).tolist()
        selected_ids.extend(fill_ids)
        for data_idx in fill_ids:
            selection_reasons[data_idx] = "random"

    selected = available[available["data_idx"].astype(str).isin(selected_ids)].copy()
    order = {data_idx: i for i, data_idx in enumerate(selected_ids)}
    selected["_order"] = selected["data_idx"].astype(str).map(order)
    selected = selected.sort_values("_order").drop(columns=["_order"])
    selected.insert(0, "iteration", iteration)
    selected["selection_reason"] = [
        selection_reasons.get(str(data_idx), "random")
        for data_idx in selected["data_idx"]
    ]
    selected["predicted_template"] = [
        predicted_templates.get(str(data_idx)) for data_idx in selected["data_idx"]
    ]
    return selected


def save_annotations(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    iteration: int,
    annotations: pd.DataFrame,
) -> Path:
    path = annotation_path(root, dataset, model_name, seed, iteration)
    mkdir_p(path.parent)
    annotations.to_csv(path, index=False)
    save_verified_template_library(root, dataset, model_name, seed, iteration)
    return path


def save_verified_template_library(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    through_iteration: int,
) -> Path:
    annotations = load_annotations(root, dataset, model_name, seed, through_iteration)
    path = template_library_path(root, dataset, model_name, seed, through_iteration)
    mkdir_p(path.parent)
    if annotations.empty:
        pd.DataFrame(columns=["template", "count", "first_iteration"]).to_csv(
            path, index=False
        )
        return path

    library = (
        annotations.dropna(subset=["template"])
        .groupby("template", as_index=False)
        .agg(count=("template", "size"), first_iteration=("iteration", "min"))
        .sort_values(["first_iteration", "template"])
    )
    library.to_csv(path, index=False)
    return path


def training_items_from_annotations(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    through_iteration: int,
) -> list[dict[str, Any]]:
    annotations = load_annotations(root, dataset, model_name, seed, through_iteration)
    if annotations.empty:
        return []
    annotations = annotations.drop_duplicates("data_idx", keep="last")
    return [
        {
            "id": str(row["data_idx"]),
            "rxn": row["mapped_rxn"],
            "split": row.get("split"),
            "source": row.get("source"),
            "num_mappings": row.get("num_mappings", 1),
            "weight": 1.0,
            "label_source": "annotation",
        }
        for _, row in annotations.iterrows()
    ]


def confident_pseudo_items(
    root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    iteration: int,
    split: str,
    per_template: int,
) -> list[dict[str, Any]]:
    if iteration <= 1 or per_template <= 0:
        return []

    pred_path = prediction_path(root, dataset, model_name, seed, split, iteration - 1)
    if not pred_path.exists():
        return []

    known_templates = verified_templates(root, dataset, model_name, seed, iteration - 1)
    if not known_templates:
        return []

    predictions = pd.read_csv(pred_path, dtype={"data_idx": str})
    if predictions.empty:
        return []

    annotations = load_annotations(root, dataset, model_name, seed, iteration)
    annotated_ids = (
        set(annotations["data_idx"].astype(str)) if not annotations.empty else set()
    )
    predictions = predictions[
        predictions["template"].fillna("").isin(known_templates)
        & ~predictions["data_idx"].astype(str).isin(annotated_ids)
    ].copy()
    if predictions.empty:
        return []

    sampled = []
    rng = np.random.default_rng(seed + 10_000 + iteration)
    for _, group in predictions.groupby("template"):
        n = min(per_template, len(group))
        sampled.append(group.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    pseudo = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    return [
        {
            "id": str(row["data_idx"]),
            "rxn": row["mapped_rxn"],
            "split": row.get("split", split),
            "source": row.get("source"),
            "num_mappings": row.get("num_mappings", 1),
            "weight": 1.0,
            "label_source": "confident_prediction",
        }
        for _, row in pseudo.iterrows()
    ]
