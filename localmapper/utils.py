import numpy as np

import torch
import dgl
from rdkit import Chem
from functools import partial


from dgllife.utils import (
    WeaveAtomFeaturizer,
    CanonicalBondFeaturizer,
    mol_to_bigraph,
)

atom_types = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "W",
    "Ru",
    "Nb",
    "Re",
    "Te",
    "Rh",
    "Ta",
    "Tc",
    "Ba",
    "Bi",
    "Hf",
    "Mo",
    "U",
    "Sm",
    "Os",
    "Ir",
    "Ce",
    "Gd",
    "Ga",
    "Cs",
]


def clean_reactant_map(rxn):
    r, p = rxn.split(">>")
    r_mol = Chem.MolFromSmiles(r)
    [atom.SetAtomMapNum(0) for atom in r_mol.GetAtoms()]
    r = Chem.MolToSmiles(r_mol, canonical=False)
    return ">>".join([r, p])


def get_adm(mol, max_distance=4):
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1  # remote (different molecule)
    dm[dm > max_distance] = max_distance + 1  # remote (same molecule)
    dm[dm == -1] = max_distance + 2  # remote (different molecule)
    return dm


def pad_atom_distance_matrix(adm_list):
    max_size = max([adm.shape[0] for adm in adm_list])
    adm_list = [
        torch.LongTensor(
            np.pad(adm, (0, max_size - adm.shape[0]), "maximum")
        ).unsqueeze(0)
        for adm in adm_list
    ]
    return torch.cat(adm_list, dim=0)


def init_featurizer():
    node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function = partial(
        mol_to_bigraph,
        add_self_loop=True,
        node_featurizer=node_featurizer,
        edge_featurizer=edge_featurizer,
        canonical_atom_order=False,
    )
    return node_featurizer, edge_featurizer, graph_function


def predict(model, device, rgraphs, pgraphs, grad=False):
    if hasattr(model, "score_graphs"):
        if grad:
            return model.score_graphs(rgraphs, pgraphs)
        with torch.no_grad():
            return model.score_graphs(rgraphs, pgraphs)

    rbg, pbg = dgl.batch(rgraphs), dgl.batch(pgraphs)
    (
        rbg.set_n_initializer(dgl.init.zero_initializer),
        pbg.set_n_initializer(dgl.init.zero_initializer),
    )
    (
        rbg.set_e_initializer(dgl.init.zero_initializer),
        pbg.set_e_initializer(dgl.init.zero_initializer),
    )
    rbg, pbg = rbg.to(device), pbg.to(device)
    rnode_feats, pnode_feats = (
        rbg.ndata.pop("h").to(device),
        pbg.ndata.pop("h").to(device),
    )
    redge_feats, pedge_feats = (
        rbg.edata.pop("e").to(device),
        pbg.edata.pop("e").to(device),
    )
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        predicitons = model(
            rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats
        )
    return predicitons


def demap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def is_valid_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_maps = [
        atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0
    ]
    return len(atom_maps) == len(set(atom_maps))


def save_reaction(rxn, path="mol.png"):
    img = Chem.Draw.MolsToGridImage(
        [Chem.MolFromSmiles(s) for s in rxn.split(">>")],
        returnPNG=False,
        molsPerRow=2,
        subImgSize=(400, 300),
    )
    path = "mol.png"
    img.save(path)
    return


def clean_map(rxn):
    r_mol, p_mol = [Chem.MolFromSmiles(s) for s in rxn.split(">>")]
    p_maps = [atom.GetAtomMapNum() for atom in p_mol.GetAtoms()]
    [
        atom.SetAtomMapNum(0)
        for atom in r_mol.GetAtoms()
        if atom.GetAtomMapNum() not in p_maps
    ]
    return ">>".join([Chem.MolToSmiles(m) for m in [r_mol, p_mol]])
