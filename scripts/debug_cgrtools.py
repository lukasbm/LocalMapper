from __future__ import annotations

import csv
import time
from pathlib import Path

import fire

from localmapper.dataset import cgr_signature, mapping_comparison_backend


def _iter_reactions(path: Path | None, column: str, limit: int | None):
    if path is None:
        yield "[CH3:1][OH:2]>>[CH3:1][Cl:2]"
        return

    with path.open() as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rxn = row.get(column)
            if rxn:
                yield rxn


def main(path=None, column="mapped_rxn", limit=20):
    path = Path(path) if path else None
    print(f"backend={mapping_comparison_backend()}")
    for i, rxn in enumerate(_iter_reactions(path, column, limit)):
        start = time.perf_counter()
        signature = cgr_signature(rxn)
        elapsed = time.perf_counter() - start
        status = "ok" if signature is not None else "none"
        print(f"{i}\t{status}\t{elapsed:.3f}s\t{rxn[:160]}")


if __name__ == "__main__":
    fire.Fire(main)
