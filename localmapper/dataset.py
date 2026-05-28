from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return
        raise


def get_adm(mol, max_distance=4):
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1  # remote (different molecule)
    dm[dm > max_distance] = max_distance + 1  # remote (same molecule)
    dm[dm == -1] = max_distance + 2  # remote (different molecule)
    return dm


def product_is_unmapped(rxn):
    r, p = [Chem.MolFromSmiles(smi) for smi in rxn.split(">>")]
    rmaps = [atom.GetAtomMapNum() for atom in r.GetAtoms()]
    pmaps = [atom.GetAtomMapNum() for atom in p.GetAtoms()]
    return 0 in pmaps or sum([m not in rmaps for m in pmaps]) > 0


def get_mapping_label(rxn):
    rsmi, psmi = rxn.split(">>")
    rmol = Chem.MolFromSmiles(rsmi)
    pmol = Chem.MolFromSmiles(psmi)
    r_atom_dict = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rmol.GetAtoms()}
    labels = []
    for atom in pmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num not in r_atom_dict:
            return None
        labels.append(r_atom_dict[map_num])
    return labels


def clean_reactant_map(rxn):
    r, p = rxn.split(">>")
    r_mol = Chem.MolFromSmiles(r)
    [atom.SetAtomMapNum(0) for atom in r_mol.GetAtoms()]
    r = Chem.MolToSmiles(r_mol, canonical=False)
    return ">>".join([r, p])


def canonicalize_map_rxn(rxn):
    new_rxn = []
    for smi in rxn.split(">>"):
        mol = Chem.MolFromSmiles(smi)
        index2mapnums = {}
        for atom in mol.GetAtoms():
            index2mapnums[atom.GetIdx()] = atom.GetAtomMapNum()
        mol_cano = Chem.RWMol(mol)
        [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
        smi_cano = Chem.MolToSmiles(mol_cano)
        mol_cano = Chem.MolFromSmiles(smi_cano)
        matches = mol.GetSubstructMatches(mol_cano)
        if matches:
            for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
                atom.SetAtomMapNum(index2mapnums[mat])
            smi = Chem.MolToSmiles(mol_cano, canonical=False)
        new_rxn.append(smi)
    return ">>".join(new_rxn)


def mapping_signature(rxn: str) -> tuple[tuple[str, str], tuple[tuple[int, int], ...]] | None:
    try:
        rxn = canonicalize_map_rxn(rxn)
        rsmi, psmi = rxn.split(">>")
        rmol = Chem.MolFromSmiles(rsmi)
        pmol = Chem.MolFromSmiles(psmi)
    except Exception:
        return None
    if rmol is None or pmol is None:
        return None

    r_demapped = Chem.RWMol(rmol)
    p_demapped = Chem.RWMol(pmol)
    for mol in (r_demapped, p_demapped):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    pattern = (
        Chem.MolToSmiles(r_demapped, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(p_demapped, canonical=True, isomericSmiles=True),
    )

    reactant_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in rmol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    pairs = []
    for product_idx, atom in enumerate(pmol.GetAtoms()):
        map_num = atom.GetAtomMapNum()
        if map_num <= 0 or map_num not in reactant_map_to_idx:
            return None
        pairs.append((product_idx, reactant_map_to_idx[map_num]))
    return pattern, tuple(pairs)


def mappings_are_equivalent(predicted_rxn: str, reference_rxn: str) -> bool:
    predicted = mapping_signature(predicted_rxn)
    reference = mapping_signature(reference_rxn)
    return predicted is not None and predicted == reference


def mapping_matches_any(predicted_rxn: str, reference_rxns: list[str]) -> bool:
    predicted = mapping_signature(predicted_rxn)
    if predicted is None:
        return False
    return any(predicted == mapping_signature(reference) for reference in reference_rxns)


def normalize_mapped_rxn(rxn: str) -> str | None:
    try:
        normalized = canonicalize_map_rxn(rxn)
    except Exception:
        return None
    return normalized if get_mapping_label(normalized) is not None else None


def _valid_rxn(rxn: Any) -> bool:
    if not isinstance(rxn, str) or rxn.count(">>") != 1:
        return False
    reactants, products = rxn.split(">>")
    return bool(reactants and products)


def _alternatives(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [
        part.strip() for part in value.strip().split(",") if _valid_rxn(part.strip())
    ]


def _item(
    idx: Any,
    rxn: Any,
    *,
    split: str | None = None,
    source: Any = None,
    alternatives: list[str] | None = None,
) -> dict[str, Any] | None:
    rxns = alternatives if alternatives is not None else _alternatives(rxn)
    if not rxns:
        return None
    return {
        "id": str(idx),
        "rxn": rxns[0],
        "mapped_rxns": rxns,
        "split": split,
        "original_split": split,
        "source": None if pd.isna(source) else source,
        "num_mappings": len(rxns),
    }


def _from_frame(
    df: pd.DataFrame,
    rxn_col: str,
    *,
    split: str | None = None,
    id_col: str | None = None,
    source_col: str | None = None,
) -> list[dict[str, Any]]:
    items = []
    for i, row in df.iterrows():
        item = _item(
            row[id_col] if id_col and id_col in df.columns else i,
            row[rxn_col],
            split=split,
            source=row[source_col] if source_col and source_col in df.columns else None,
        )
        if item is not None:
            items.append(item)
    return items


def _from_line_file(path: Path, *, split: str, id_prefix: str | None = None) -> list[dict[str, Any]]:
    items = []
    with path.open() as f:
        for i, line in enumerate(f):
            rxns = _alternatives(line.rstrip("\n\r"))
            idx = f"{id_prefix}:{i}" if id_prefix else i
            item = _item(idx, None, split=split, alternatives=rxns)
            if item is not None:
                items.append(item)
    return items


def _from_semicolon_file(path: Path, *, split: str, id_prefix: str | None = None) -> list[dict[str, Any]]:
    items = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n\r")
            if not line:
                continue
            idx, payload = line.split(";", 1) if ";" in line else (i, line)
            idx = f"{id_prefix}:{idx}" if id_prefix else idx
            item = _item(idx, None, split=split, alternatives=_alternatives(payload))
            if item is not None:
                items.append(item)
    return items


class ReactionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mol_to_graph,
        *,
        data_root: str | Path = "data",
        items: list[dict[str, Any]] | None = None,
        include_labels: bool = True,
    ):
        self.mol_to_graph = mol_to_graph
        self.include_labels = include_labels
        self.data_root = Path(data_root)
        self.items = self.load_items() if items is None else items

    def load_items(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def from_items(
        cls,
        items: list[dict[str, Any]],
        mol_to_graph,
        *,
        include_labels: bool = True,
    ) -> "ReactionDataset":
        return cls(mol_to_graph, items=items, include_labels=include_labels)

    def __getitem__(self, item):
        data = self.items[item]
        rxn = data["rxn"]
        r, p = rxn.split(">>")
        rgraph = self.mol_to_graph(Chem.MolFromSmiles(r))
        pgraph = self.mol_to_graph(Chem.MolFromSmiles(p))
        label = get_mapping_label(rxn) if self.include_labels else []
        if self.include_labels and label is None:
            raise ValueError(f"Invalid mapped reaction for training item: {data['id']}")
        return data["id"], rxn, rgraph, pgraph, label, data.get("weight", 1.0), data

    def __len__(self):
        return len(self.items)


class USPTO50KDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "USPTO_50K" / "raw_data.csv")
        return _from_frame(df, "mapped_rxn")


class GoldenDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "Golden" / "raw_data.csv")
        return _from_frame(df, "mapped_rxn")


class NatCommDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "NatComm" / "test_data.csv")
        return _from_frame(df, "mapped_rxn", source_col="source")


class SchneiderDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        df = pd.read_csv(self.data_root / "schneider" / "schneider50k.tsv", sep="\t")
        return _from_frame(df, "clean_rxn", id_col="Unnamed: 0", source_col="source")


class RingReactionsDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        return _from_line_file(
            self.data_root / "ringreactions" / "train_ringreactions.csv",
            split=None,
            id_prefix="train",
        ) + _from_line_file(
            self.data_root / "ringreactions" / "test_ringreactions.csv",
            split=None,
            id_prefix="test",
        )


class MetAMDBDataset(ReactionDataset):
    def load_items(self) -> list[dict[str, Any]]:
        return _from_semicolon_file(
            self.data_root / "metAMDB" / "train_metamdb_filtered.csv",
            split=None,
            id_prefix="train",
        ) + _from_semicolon_file(
            self.data_root / "metAMDB" / "test_metamdb_filtered.csv",
            split=None,
            id_prefix="test",
        )


DATASETS = {
    "uspto_50k": USPTO50KDataset,
    "uspto50k": USPTO50KDataset,
    "golden": GoldenDataset,
    "natcomm": NatCommDataset,
    "jaworski": NatCommDataset,
    "schneider": SchneiderDataset,
    "schneider50k": SchneiderDataset,
    "ringreactions": RingReactionsDataset,
    "ring_reactions": RingReactionsDataset,
    "metamdb": MetAMDBDataset,
    "metamdb_filtered": MetAMDBDataset,
}


def _dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def create_reaction_dataset(
    dataset: str,
    mol_to_graph,
    *,
    data_root: str | Path = "data",
    include_labels: bool = True,
) -> ReactionDataset:
    dataset_cls = DATASETS.get(_dataset_key(dataset))
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    dataset_obj = dataset_cls(
        mol_to_graph,
        data_root=data_root,
        include_labels=include_labels,
    )
    dataset_obj.items = [item for item in dataset_obj.items if item_has_valid_mapping(item)]
    return dataset_obj


def normalize_item_target(item: dict[str, Any]) -> dict[str, Any] | None:
    reference_rxns = item.get("mapped_rxns", [item["rxn"]])
    normalized_rxn = next(
        (
            normalized
            for reference_rxn in reference_rxns
            if (normalized := normalize_mapped_rxn(reference_rxn))
        ),
        None,
    )
    if normalized_rxn is None:
        return None
    return {**item, "rxn": normalized_rxn}


def normalize_item_targets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in (normalize_item_target(item) for item in items) if item]


def item_has_valid_mapping(item: dict[str, Any]) -> bool:
    return any(
        get_mapping_label(reference_rxn) is not None
        for reference_rxn in item.get("mapped_rxns", [item["rxn"]])
    )


def select_split(
    items: list[dict[str, Any]],
    split: str,
    *,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> list[dict[str, Any]]:
    split = split.lower()
    explicit = [item for item in items if item["split"] == split]
    if explicit:
        return normalize_item_targets(explicit)

    unsplit = [item for item in items if item["split"] is None]
    if not unsplit:
        return []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unsplit))
    n_test = int(round(len(unsplit) * test_fraction))
    n_val = int(round(len(unsplit) * val_fraction))
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test : n_test + n_val])

    selected = []
    for i, item in enumerate(unsplit):
        item_split = "test" if i in test_ids else "val" if i in val_ids else "train"
        if item_split == split:
            selected.append({**item, "split": item_split})
    return normalize_item_targets(selected)


def load_dataset_items(
    dataset: str,
    *,
    data_root: str | Path = "data",
) -> list[dict[str, Any]]:
    # For code that only needs the reaction list, not graph construction.
    items = create_reaction_dataset(dataset, lambda mol: mol, data_root=data_root).items
    return [item for item in items if item_has_valid_mapping(item)]


def load_reactions(
    dataset: str,
    split: str,
    *,
    data_root: str | Path = "data",
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> list[dict[str, Any]]:
    return select_split(
        load_dataset_items(dataset, data_root=data_root),
        split,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def collate_reaction_batch(batch):
    batch = [row for row in batch if row is not None]
    idxs, rxns, rgraphs, pgraphs, labels, weights, data = zip(*batch)
    labels_list = [torch.as_tensor(label, dtype=torch.long) for label in labels]
    masks_list = [torch.ones_like(label, dtype=torch.long) for label in labels_list]
    return (
        list(idxs),
        list(rxns),
        list(rgraphs),
        list(pgraphs),
        labels_list,
        masks_list,
        list(weights),
        list(data),
    )
