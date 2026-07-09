from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[1]

from localmapper.dataset import mapping_comparison_backend, mkdir_p


def _output_dir(root: Path, dataset: str, model_name: str, seed: int) -> Path:
    return root / "outputs" / dataset / f"{model_name}_seed{seed}"


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _package_version(module_name: str, *distribution_names: str) -> str | None:
    for distribution_name in distribution_names:
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def main(
    dataset: str,
    model: str,
    init_mode: str,
    seed: int = 0,
    split: str | None = None,
    train_split: str | None = None,
    eval_split: str = "test",
    sample_limit: int = 200,
    sample_candidate_factor: int = 20,
    iterations: int = 5,
    batch_size: int = 16,
    num_epochs: int = 100,
    patience: int = 5,
    confident_per_template: int = 100,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    gpu: str = "cuda:0",
    pretrained_checkpoint: str | None = None,
    template_library: str | None = None,
    eequaam_chunk_size: int = 100,
    eequaam_base_timeout_seconds: float = 30.0,
    eequaam_timeout_seconds_per_reaction: float = 5.0,
    run_id: str | None = None,
    budget_label: str | None = None,
):
    train_split = train_split or split or "train"
    run_dir = _output_dir(ROOT, dataset, model, seed)
    mkdir_p(run_dir)

    metadata = {
        "dataset": dataset,
        "model": model,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "init_mode": init_mode,
        "seed": int(seed),
        "split": train_split,
        "train_split": train_split,
        "eval_split": eval_split,
        "active_learning": {
            "sample_limit_per_iteration": int(sample_limit),
            "sample_candidate_factor": int(sample_candidate_factor),
            "iterations": int(iterations),
            "total_annotation_budget": int(sample_limit) * int(iterations),
            "budget_label": budget_label,
            "annotation_source": "dataset_ground_truth",
            "selection_after_iteration_1": "uncertain_predicted_templates_then_random_fill",
        },
        "training": {
            "batch_size": int(batch_size),
            "num_epochs": int(num_epochs),
            "patience": int(patience),
            "confident_per_template": int(confident_per_template),
            "pretrained_checkpoint": pretrained_checkpoint
            if init_mode
            in {"pretrained", "finetune", "pretrained_eval", "paper_checkpoint"}
            else None,
            "device_request": gpu,
        },
        "splits": {
            "train": train_split,
            "eval": eval_split,
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
        },
        "evaluation": {
            "aam_equivalence_backend": mapping_comparison_backend(),
            "template_library": template_library,
            "eequaam_chunk_size": int(eequaam_chunk_size),
            "eequaam_base_timeout_seconds": float(eequaam_base_timeout_seconds),
            "eequaam_timeout_seconds_per_reaction": float(
                eequaam_timeout_seconds_per_reaction
            ),
            "eequaam_tmpdir": os.environ.get(
                "LOCALMAPPER_EEQUAAM_TMPDIR", "outputs/eequaam_tmp"
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _package_version("torch", "torch"),
        },
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
    }

    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Saved run metadata to {metadata_path}")
    return str(metadata_path)


if __name__ == "__main__":
    fire.Fire(main)
