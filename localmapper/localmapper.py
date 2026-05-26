import pickle
import torch


from rdkit import Chem

from .utils import *
from .mapper import prediction2map
import os


class localmapper:
    def __init__(self, device=torch.device("cpu"), model_version="202403"):
        self.device = device
        config_path = "default_config.json"
        template_path = os.path.join("data", f"templates_{model_version}.pkl")
        model_path = os.path.join("data", f"LocalMapper_{model_version}.pth")
        with open(template_path, "rb") as f:
            self.accepted_templates = pickle.load(f)
        node_featurizer, edge_featurizer, self.graph_function = init_featurizer()
        exp_config = get_configure(config_path, node_featurizer, edge_featurizer)
        self.model = load_model(exp_config, model_path, device)
        self.model.eval()

    def pred_pxr(self, rxns):
        rgraphs, pgraphs = [], []
        for rxn in rxns:
            reactant, product = rxn.split(">>")
            reactant, product = (
                Chem.MolFromSmiles(reactant),
                Chem.MolFromSmiles(product),
            )
            rgraph, pgraph = self.graph_function(reactant), self.graph_function(product)
            rgraphs.append(rgraph)
            pgraphs.append(pgraph)
        predictions = predict(self.model, self.device, rgraphs, pgraphs)
        return [torch.softmax(pred, dim=1).cpu().numpy() for pred in predictions]

    def get_atom_map(self, rxns, return_dict=False):
        single_input = isinstance(rxns, str)
        if single_input:
            rxns = [rxns]
        results = []
        predictions = self.pred_pxr(rxns)
        for i, (rxn, prediction) in enumerate(zip(rxns, predictions)):
            mapped_result = prediction2map(rxn, prediction)
            result = {"rxn": rxn}
            result.update(mapped_result)
            confident = result["template"] in self.accepted_templates
            if not confident:
                mapped_result = prediction2map(rxn, prediction, 90)
                result.update(mapped_result)
                confident = result["template"] in self.accepted_templates
            result["confident"] = confident
            if not return_dict:
                result = mapped_result["mapped_rxn"]
            results.append(result)
        if single_input:
            return results[0]
        else:
            return results

    def plot_rxn(self, rxn):
        rdkit_rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
        return Chem.Draw.ReactionToImage(rdkit_rxn)
