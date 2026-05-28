from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import argparse

from rdkit import Chem

from localmapper.LocalTemplate.template_extractor import extract_from_reaction
from localmapper.dataset import DATASETS, get_mapping_label


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DATASETS = {
    "uspto_50k": "USPTO_50K",
    "golden": "Golden",
    "natcomm": "NatComm",
    "schneider": "Schneider",
    "ringreactions": "ringreactions",
    "metamdb": "metAMDB",
}


def _canonical_side(side: str) -> str | None:
    mol = Chem.MolFromSmiles(side)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def reaction_pattern(rxn: str) -> str | None:
    if rxn.count(">>") != 1:
        return None
    reactants, products = rxn.split(">>")
    reactants = _canonical_side(reactants)
    products = _canonical_side(products)
    if reactants is None or products is None:
        return None
    return f"{reactants}>>{products}"


def atom_counts(rxn: str) -> tuple[int, int] | None:
    if rxn.count(">>") != 1:
        return None
    counts = []
    for side in rxn.split(">>"):
        mol = Chem.MolFromSmiles(side)
        if mol is None:
            return None
        counts.append(mol.GetNumAtoms())
    return counts[0], counts[1]


def _fmt_ratio(value: float) -> str:
    return f"{value:.3f}"


def _fmt_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def _safe_mean(values: list[int | float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_median(values: list[int | float]) -> float:
    return float(median(values)) if values else 0.0


def _top(counter: Counter[str], n: int) -> list[tuple[str, int]]:
    return counter.most_common(n) if n > 0 else []


def _print_top(title: str, rows: list[tuple[str, int]], total: int) -> None:
    if not rows:
        return
    print(f"  {title}:")
    for value, count in rows:
        share = count / total if total else 0.0
        print(f"    {count:>6}  {_fmt_ratio(share):>6}  {value}")


def summarize_dataset(
    dataset_key: str,
    *,
    data_root: Path,
    max_items: int | None,
    top_n: int,
) -> dict[str, Any]:
    dataset = DATASETS[dataset_key](lambda mol: mol, data_root=data_root)
    raw_items = list(dataset.items)
    if max_items is not None:
        raw_items = raw_items[:max_items]

    split_counts = Counter(item.get("split") or "unsplit" for item in raw_items)
    source_counts = Counter(
        str(item.get("source")) for item in raw_items if item.get("source") is not None
    )

    valid_items = []
    invalid_mapping = 0
    for item in raw_items:
        if get_mapping_label(item["rxn"]) is None:
            invalid_mapping += 1
        else:
            valid_items.append(item)

    mapped_rxns = [item["rxn"] for item in valid_items]
    mapping_alternatives = [int(item.get("num_mappings", 1) or 1) for item in raw_items]

    pattern_counter: Counter[str] = Counter()
    pattern_failures = 0
    reactant_atoms = []
    product_atoms = []
    for rxn in mapped_rxns:
        pattern = reaction_pattern(rxn)
        if pattern is None:
            pattern_failures += 1
        else:
            pattern_counter[pattern] += 1

        counts = atom_counts(rxn)
        if counts is not None:
            reactant_atoms.append(counts[0])
            product_atoms.append(counts[1])

    template_counter: Counter[str] = Counter()
    template_failures = 0
    for rxn in mapped_rxns:
        try:
            template = extract_from_reaction(rxn)
        except Exception:
            template = None
        if template:
            template_counter[template] += 1
        else:
            template_failures += 1

    valid_count = len(valid_items)
    summary = {
        "dataset": CANONICAL_DATASETS[dataset_key],
        "raw_items": len(raw_items),
        "valid_items": valid_count,
        "invalid_mapping": invalid_mapping,
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "mapped_unique": len(set(mapped_rxns)),
        "mapped_unique_ratio": len(set(mapped_rxns)) / valid_count if valid_count else 0.0,
        "pattern_unique": len(pattern_counter),
        "pattern_unique_ratio": len(pattern_counter) / valid_count if valid_count else 0.0,
        "pattern_failures": pattern_failures,
        "template_unique": len(template_counter),
        "template_unique_ratio": len(template_counter) / valid_count if valid_count else 0.0,
        "template_failures": template_failures,
        "multi_mapping_rows": sum(1 for count in mapping_alternatives if count > 1),
        "max_mappings_per_row": max(mapping_alternatives, default=0),
        "mean_mappings_per_row": _safe_mean(mapping_alternatives),
        "mean_reactant_atoms": _safe_mean(reactant_atoms),
        "mean_product_atoms": _safe_mean(product_atoms),
        "median_reactant_atoms": _safe_median(reactant_atoms),
        "median_product_atoms": _safe_median(product_atoms),
        "top_templates": _top(template_counter, top_n),
        "top_patterns": _top(pattern_counter, top_n),
        "top_sources": _top(source_counts, top_n),
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== {summary['dataset']} ===")
    print(
        "  items: raw={raw_items}, valid_mapping={valid_items}, invalid_mapping={invalid_mapping}".format(
            **summary
        )
    )
    print(f"  splits: {summary['split_counts']}")
    if summary["source_counts"]:
        print(f"  sources: {summary['source_counts']}")
    print(
        "  mapping alternatives: multi_rows={multi_mapping_rows}, mean_per_row={mean_mappings_per_row}, max_per_row={max_mappings_per_row}".format(
            multi_mapping_rows=summary["multi_mapping_rows"],
            mean_mappings_per_row=_fmt_number(summary["mean_mappings_per_row"]),
            max_mappings_per_row=summary["max_mappings_per_row"],
        )
    )
    print(
        "  unique mapped reactions: {mapped_unique} ({mapped_unique_ratio})".format(
            mapped_unique=summary["mapped_unique"],
            mapped_unique_ratio=_fmt_ratio(summary["mapped_unique_ratio"]),
        )
    )
    print(
        "  unique demapped patterns: {pattern_unique} ({pattern_unique_ratio}), failures={pattern_failures}".format(
            pattern_unique=summary["pattern_unique"],
            pattern_unique_ratio=_fmt_ratio(summary["pattern_unique_ratio"]),
            pattern_failures=summary["pattern_failures"],
        )
    )
    print(
        "  unique templates: {template_unique} ({template_unique_ratio}), failures={template_failures}".format(
            template_unique=summary["template_unique"],
            template_unique_ratio=_fmt_ratio(summary["template_unique_ratio"]),
            template_failures=summary["template_failures"],
        )
    )
    print(
        "  atoms: reactants mean/median={}/{}; products mean/median={}/{}".format(
            _fmt_number(summary["mean_reactant_atoms"]),
            _fmt_number(summary["median_reactant_atoms"]),
            _fmt_number(summary["mean_product_atoms"]),
            _fmt_number(summary["median_product_atoms"]),
        )
    )
    _print_top("top templates", summary["top_templates"], summary["valid_items"])
    _print_top("top demapped patterns", summary["top_patterns"], summary["valid_items"])
    _print_top("top sources", summary["top_sources"], summary["raw_items"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print mapping, pattern, and template diversity stats for LocalMapper datasets."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated canonical dataset names, or 'all'.",
    )
    parser.add_argument("--data_root", default=str(ROOT / "data"))
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Optional cap per dataset for quick inspection.",
    )
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    if args.datasets == "all":
        dataset_keys = list(CANONICAL_DATASETS)
    else:
        requested = [name.strip().lower().replace("-", "_") for name in args.datasets.split(",")]
        dataset_keys = []
        for name in requested:
            if name not in CANONICAL_DATASETS:
                raise SystemExit(
                    f"Unknown dataset '{name}'. Choose from: {', '.join(CANONICAL_DATASETS)}"
                )
            dataset_keys.append(name)

    print("Dataset diversity report")
    if args.max_items is not None:
        print(f"Using first {args.max_items} raw items per dataset.")
    print(f"Data root: {Path(args.data_root)}")

    summaries = [
        summarize_dataset(
            dataset_key,
            data_root=Path(args.data_root),
            max_items=args.max_items,
            top_n=args.top_n,
        )
        for dataset_key in dataset_keys
    ]

    for summary in summaries:
        print_summary(summary)

    print("\n=== Compact Comparison ===")
    header = (
        "dataset",
        "valid",
        "mapped_u",
        "pattern_u",
        "template_u",
        "template_fail",
        "multi_rows",
        "max_alts",
    )
    print("{:<16} {:>8} {:>9} {:>10} {:>10} {:>13} {:>10} {:>8}".format(*header))
    for summary in summaries:
        print(
            "{:<16} {:>8} {:>9} {:>10} {:>10} {:>13} {:>10} {:>8}".format(
                summary["dataset"],
                summary["valid_items"],
                summary["mapped_unique"],
                summary["pattern_unique"],
                summary["template_unique"],
                summary["template_failures"],
                summary["multi_mapping_rows"],
                summary["max_mappings_per_row"],
            )
        )


if __name__ == "__main__":
    main()
