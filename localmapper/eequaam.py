from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from tqdm import tqdm


EEQUAAM_BACKEND = "eequaam_its"


@dataclass(frozen=True)
class EEquAAMComparison:
    comparison_id: str
    predicted_rxn: str
    reference_rxn: str
    row_index: int


@dataclass(frozen=True)
class EEquAAMResult:
    equivalent: bool
    status: str
    detail: str = ""


def reaction_is_complete_bijective(rxn: str) -> bool:
    mols = _reaction_mols(rxn)
    if mols is None:
        return False
    reactant_mol, product_mol = mols
    reactant_maps = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
    product_maps = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]
    if 0 in reactant_maps or 0 in product_maps:
        return False
    return sorted(reactant_maps) == sorted(product_maps) and len(reactant_maps) == len(
        set(reactant_maps)
    )


def _reaction_mols(rxn: str):
    if not isinstance(rxn, str) or rxn.count(">>") != 1:
        return None
    reactants, products = rxn.split(">>")
    if not reactants or not products:
        return None
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(products)
    if reactant_mol is None or product_mol is None:
        return None
    return reactant_mol, product_mol


def evaluate_eequaam(
    comparisons: list[EEquAAMComparison],
    *,
    eequaam_script: str | Path,
    chunk_size: int = 100,
    base_timeout_seconds: float = 30.0,
    timeout_seconds_per_reaction: float = 5.0,
) -> dict[str, EEquAAMResult]:
    eequaam_script = Path(eequaam_script)
    if not eequaam_script.exists():
        raise FileNotFoundError(f"EEquAAM script not found: {eequaam_script}")
    chunk_size = max(int(chunk_size), 1)

    results: dict[str, EEquAAMResult] = {}
    valid_comparisons = []
    for comparison in tqdm(
        comparisons,
        desc="Checking EEquAAM inputs",
        disable=_progress_disabled(),
    ):
        if not reaction_is_complete_bijective(comparison.predicted_rxn):
            results[comparison.comparison_id] = EEquAAMResult(
                False, "predicted_not_complete_bijective"
            )
        elif not reaction_is_complete_bijective(comparison.reference_rxn):
            results[comparison.comparison_id] = EEquAAMResult(
                False, "reference_not_complete_bijective"
            )
        else:
            valid_comparisons.append(comparison)

    chunk_starts = range(0, len(valid_comparisons), chunk_size)
    for start in tqdm(
        chunk_starts,
        total=(len(valid_comparisons) + chunk_size - 1) // chunk_size,
        desc="Running EEquAAM chunks",
        disable=_progress_disabled(),
    ):
        chunk = valid_comparisons[start : start + chunk_size]
        results.update(
            _evaluate_chunk_with_fallback(
                chunk,
                eequaam_script=eequaam_script,
                base_timeout_seconds=base_timeout_seconds,
                timeout_seconds_per_reaction=timeout_seconds_per_reaction,
            )
        )
    return results


def _evaluate_chunk_with_fallback(
    comparisons: list[EEquAAMComparison],
    *,
    eequaam_script: Path,
    base_timeout_seconds: float,
    timeout_seconds_per_reaction: float,
) -> dict[str, EEquAAMResult]:
    if not comparisons:
        return {}

    result, failure_detail = _run_eequaam_chunk(
        comparisons,
        eequaam_script=eequaam_script,
        timeout_seconds=base_timeout_seconds
        + timeout_seconds_per_reaction * len(comparisons),
    )
    if result is not None:
        missing_ids = {
            comparison.comparison_id for comparison in comparisons
        } - set(result)
        for comparison_id in missing_ids:
            result[comparison_id] = EEquAAMResult(
                False,
                "eequaam_failed",
                "EEquAAM did not emit a result for this comparison",
            )
        return result

    if len(comparisons) == 1:
        comparison = comparisons[0]
        return {
            comparison.comparison_id: EEquAAMResult(
                False,
                "eequaam_failed",
                failure_detail or "EEquAAM failed or timed out for this comparison",
            )
        }

    midpoint = len(comparisons) // 2
    return {
        **_evaluate_chunk_with_fallback(
            comparisons[:midpoint],
            eequaam_script=eequaam_script,
            base_timeout_seconds=base_timeout_seconds,
            timeout_seconds_per_reaction=timeout_seconds_per_reaction,
        ),
        **_evaluate_chunk_with_fallback(
            comparisons[midpoint:],
            eequaam_script=eequaam_script,
            base_timeout_seconds=base_timeout_seconds,
            timeout_seconds_per_reaction=timeout_seconds_per_reaction,
        ),
    }


def _run_eequaam_chunk(
    comparisons: list[EEquAAMComparison],
    *,
    eequaam_script: Path,
    timeout_seconds: float,
) -> tuple[dict[str, EEquAAMResult] | None, str]:
    temp_root = _eequaam_temp_root().resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="localmapper-eequaam-", dir=temp_root) as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "comparisons.smiles"
        summary_path = tmp_dir / "comparisons_summary.txt"
        input_path.write_text(_eequaam_input(comparisons))
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        try:
            subprocess.run(
                [sys.executable, str(eequaam_script.resolve()), str(input_path.resolve())],
                cwd=tmp_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return None, _subprocess_detail(exc)
        except subprocess.CalledProcessError as exc:
            return None, _subprocess_detail(exc)
        except OSError as exc:
            return None, str(exc)
        if not summary_path.exists():
            return None, "EEquAAM did not write a summary file"
        return _parse_summary(summary_path), ""


def _subprocess_detail(exc: subprocess.SubprocessError) -> str:
    parts = [exc.__class__.__name__]
    stdout = getattr(exc, "stdout", None)
    stderr = getattr(exc, "stderr", None)
    if stdout:
        parts.append(str(stdout)[-1000:])
    if stderr:
        parts.append(str(stderr)[-1000:])
    return "\n".join(parts)


def _eequaam_temp_root() -> Path:
    return Path(os.environ.get("LOCALMAPPER_EEQUAAM_TMPDIR", "outputs/eequaam_tmp"))


def _progress_disabled() -> bool:
    return os.environ.get("LOCALMAPPER_PROGRESS", "1") == "0"


def _eequaam_input(comparisons: list[EEquAAMComparison]) -> str:
    lines = []
    for comparison in comparisons:
        lines.extend(
            [
                f"#,{comparison.comparison_id}",
                f"pred,{comparison.predicted_rxn}",
                f"ref,{comparison.reference_rxn}",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_summary(path: Path) -> dict[str, EEquAAMResult]:
    results: dict[str, EEquAAMResult] = {}
    current_id = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("#,"):
                current_id = line[2:]
                continue
            if current_id is None or not line.startswith("result ITS\t"):
                continue
            value = line.split("\t", 1)[1]
            results[current_id] = EEquAAMResult(
                value == "equivalent maps",
                "ok",
                value,
            )
    return results


def summarize_row_results(rows: list[dict[str, Any]]) -> dict[str, int]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("eequaam_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return status_counts
