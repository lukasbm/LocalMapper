from pathlib import Path
import getpass
import json
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

from .models import LocalMapper

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

def get_configure(config_path, node_featurizer, edge_featurizer):
    with open(config_path, "r") as f:
        config = json.load(f)
    config["in_node_feats"] = node_featurizer.feat_size()
    config["in_edge_feats"] = edge_featurizer.feat_size()
    return config


def load_model(exp_config, model_path, device):
    model = LocalMapper(
        node_in_feats=exp_config["in_node_feats"],
        edge_in_feats=exp_config["in_edge_feats"],
        node_out_feats=exp_config["node_out_feats"],
        edge_hidden_feats=exp_config["edge_hidden_feats"],
        num_step_message_passing=exp_config["num_step_message_passing"],
        attention_heads=exp_config["attention_heads"],
        attention_layers=exp_config["attention_layers"],
    )
    model = model.to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    return model


def load_model2(
    config_path,
    node_featurizer,
    edge_featurizer,
    device,
    mode,
    learning_rate=None,
    weight_decay=None,
    patience=None,
    model_path=None,
):
    exp_config = get_configure(config_path, node_featurizer, edge_featurizer)
    model = LocalMapper(
        node_in_feats=exp_config["in_node_feats"],
        edge_in_feats=exp_config["in_edge_feats"],
        node_out_feats=exp_config["node_out_feats"],
        edge_hidden_feats=exp_config["edge_hidden_feats"],
        num_step_message_passing=exp_config["num_step_message_passing"],
        attention_heads=exp_config["attention_heads"],
        attention_layers=exp_config["attention_layers"],
    )
    model = model.to(device)

    if mode == "train":
        if model_path is None:
            raise ValueError("model_path is required when mode='train'")
        if learning_rate is None or weight_decay is None or patience is None:
            raise ValueError(
                "learning_rate, weight_decay, and patience are required when mode='train'"
            )
        from torch import nn
        from torch.optim import Adam, lr_scheduler
        from .cli_utils import EarlyStopping

        loss_criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-4
        )
        stopper = EarlyStopping(
            mode="lower", patience=patience, filename=model_path
        )
        return model, loss_criterion, optimizer, scheduler, stopper

    if model_path is None:
        raise ValueError("model_path is required when mode!='train'")

    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    return model


def predict(model, device, rgraphs, pgraphs):
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
    with torch.no_grad():
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


def trash_loop():
    chemist_name = get_user_name()
    dataset = "USPTO_50K"

    samp_iter = 1
    sampled_data = load_sampled_data(dataset, chemist_name, samp_iter)
    accepted_templates, rejected_templates = load_fixed_templates(
        dataset, chemist_name, samp_iter
    )
    remapped_rxn_dict = {}
    remapped_temp_dict = {}

    ## Manually check AAM
    # 0: remap, 1: accept, 2: reject reaction
    for i, (idx, rxn, temp, freq) in enumerate(
        zip(
            sampled_data["data_idx"],
            sampled_data["mapped_rxn"],
            sampled_data["template"],
            sampled_data["freq"],
        )
    ):  # remap: reject, 1: accept, 2: reject
        if idx in remapped_rxn_dict:
            continue
        rxn = clean_map(rxn)
        r, p = rxn.split(">>")
        temp = extract_from_reaction(rxn)
        answer = "1"

        while True:
            if temp in accepted_templates:
                answer = "1"
                break
            print(rxn)
            print("Reactant: \n", r)
            print("Template: \n", temp)
            print("Frequency: \n", freq)
            save_reaction(rxn)
            display(Image.open("mol.png"))
            answer = input("Correct (%d/%d)?" % (i, len(sampled_data)))
            if answer in ["1", "2"]:
                break
            remap = input("Remap (%d/%d)..." % (i, len(sampled_data)))
            if not is_valid_mapping(remap):
                print("Not valid mapping!")
                continue
            else:
                r = remap
            rxn = "%s>>%s" % (r, p)
            temp = extract_from_reaction(rxn)

        save_reaction(rxn)
        display(Image.open("mol.png"))
        if answer == "1":
            remapped_rxn_dict[idx] = rxn
            remapped_temp_dict[idx] = temp
            accepted_templates.add(temp)

        clear_output(wait=True)

    # Sort the reaction idex before exporting
    remapped_idxs, remapped_rxns, remapped_temps = [], [], []
    for idx in sorted(list(remapped_temp_dict.keys())):
        remapped_idxs.append(idx)
        remapped_rxns.append(remapped_rxn_dict[idx])
        remapped_temps.append(remapped_temp_dict[idx])
    df = pd.DataFrame(
        {
            "data_idx": remapped_idxs,
            "mapped_rxn": remapped_rxns,
            "template": remapped_temps,
        }
    )
    save_fixed_data(df, dataset, chemist_name, samp_iter)
