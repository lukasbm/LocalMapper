from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import dgl
from dgllife.model import MPNNGNN
from rdkit import Chem

from localmapper.atom_mapper import prediction2map
from localmapper.model_utils import MultiHeadAttention, CrossReactivityAttention


class LocalMapper(nn.Module):
    def __init__(
        self,
        node_in_feats=None,
        edge_in_feats=None,
        node_out_feats=256,
        edge_hidden_feats=32,
        num_step_message_passing=3,
        attention_heads=8,
        attention_layers=3,
        graph_function=None,
        device="cpu",
    ):
        super(LocalMapper, self).__init__()
        if node_in_feats is None or edge_in_feats is None or graph_function is None:
            from localmapper.utils import init_featurizer

            node_featurizer, edge_featurizer, default_graph_function = init_featurizer()
            if node_in_feats is None:
                node_in_feats = node_featurizer.feat_size()
            if edge_in_feats is None:
                edge_in_feats = edge_featurizer.feat_size()
            if graph_function is None:
                graph_function = default_graph_function
        self.graph_function = graph_function

        self.mpnn = MPNNGNN(
            node_in_feats=node_in_feats,
            node_out_feats=node_out_feats,
            edge_in_feats=edge_in_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )

        self.cross_att = CrossReactivityAttention(
            node_out_feats, attention_heads, attention_layers
        )
        self.amm_att = MultiHeadAttention(
            1, node_out_feats, dropout=0.2, return_att=True
        )
        self.to(device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu", **model_kwargs):
        model = cls(device=device, **model_kwargs)
        model.load_checkpoint(checkpoint_path, device=device)
        return model

    def save_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, device=None):
        map_location = device if device is not None else next(self.parameters()).device
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def batch_att(self, rbg, pbg, feats_r, feats_p):
        rbg.ndata["h"], pbg.ndata["h"] = feats_r, feats_p
        rgs, pgs = dgl.unbatch(rbg), dgl.unbatch(pbg)
        batch_p = pad_sequence(
            [g.ndata["h"] for g in pgs], batch_first=True, padding_value=0
        )  # batch, patom_n, hidden_n
        batch_r = pad_sequence(
            [g.ndata["h"] for g in rgs], batch_first=True, padding_value=0
        )  # batch, ratom_n, hidden_n

        batch_p = self.cross_att(batch_p, batch_r)
        aam_score = self.amm_att(batch_p, batch_r)
        aam_score = [
            score[: pg.num_nodes(), : rg.num_nodes()]
            for pg, rg, score in zip(pgs, rgs, aam_score)
        ]
        return aam_score

    def forward(self, rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats):
        rnode_feats = self.mpnn(rbg, rnode_feats, redge_feats)
        pnode_feats = self.mpnn(pbg, pnode_feats, pedge_feats)
        mapping_scores = self.batch_att(rbg, pbg, rnode_feats, pnode_feats)
        return mapping_scores

    def score_graphs(self, rgraphs, pgraphs):
        rbg, pbg = dgl.batch(rgraphs), dgl.batch(pgraphs)
        rbg.set_n_initializer(dgl.init.zero_initializer)
        pbg.set_n_initializer(dgl.init.zero_initializer)
        rbg.set_e_initializer(dgl.init.zero_initializer)
        pbg.set_e_initializer(dgl.init.zero_initializer)
        rbg, pbg = rbg.to(self.device), pbg.to(self.device)
        rnode_feats, pnode_feats = (
            rbg.ndata.pop("h").to(self.device),
            pbg.ndata.pop("h").to(self.device),
        )
        redge_feats, pedge_feats = (
            rbg.edata.pop("e").to(self.device),
            pbg.edata.pop("e").to(self.device),
        )
        return self(rbg, pbg, rnode_feats, pnode_feats, redge_feats, pedge_feats)

    def _rxns_to_graphs(self, rxns):
        if self.graph_function is None:
            raise ValueError("graph_function is required to score raw reaction strings")
        rgraphs, pgraphs = [], []
        for rxn in rxns:
            reactant, product = rxn.split(">>")
            rgraphs.append(self.graph_function(Chem.MolFromSmiles(reactant)))
            pgraphs.append(self.graph_function(Chem.MolFromSmiles(product)))
        return rgraphs, pgraphs

    def score_rxns(self, rxns, grad=False):
        single_input = isinstance(rxns, str)
        if single_input:
            rxns = [rxns]
        rgraphs, pgraphs = self._rxns_to_graphs(rxns)
        if grad:
            scores = self.score_graphs(rgraphs, pgraphs)
        else:
            with torch.no_grad():
                scores = self.score_graphs(rgraphs, pgraphs)
        return scores[0] if single_input else scores

    def map_rxns(
        self,
        rxns,
        accepted_templates=None,
        try_twice=True,
        return_dict=False,
    ):
        single_input = isinstance(rxns, str)
        if single_input:
            rxns = [rxns]
        was_training = self.training
        self.eval()
        logits_list = self.score_rxns(rxns, grad=False)
        results = self.map_scores(
            rxns,
            logits_list,
            accepted_templates=accepted_templates,
            try_twice=try_twice,
            return_dict=return_dict,
        )

        if was_training:
            self.train()
        return results[0] if single_input else results

    def map_scores(
        self,
        rxns,
        logits_list,
        accepted_templates=None,
        try_twice=True,
        return_dict=False,
    ):
        has_template_filter = accepted_templates is not None
        accepted_templates = set() if accepted_templates is None else accepted_templates

        results = []
        for rxn, logits in zip(rxns, logits_list):
            prediction = torch.softmax(logits, dim=1).cpu().numpy()
            mapped_result = prediction2map(rxn, prediction)
            result = {"rxn": rxn}
            result.update(mapped_result)
            confident = result["template"] in accepted_templates
            if has_template_filter and try_twice and not confident:
                mapped_result = prediction2map(rxn, prediction, neighbor_weight=90)
                result.update(mapped_result)
                confident = result["template"] in accepted_templates
            result["confident"] = confident if has_template_filter else None
            results.append(result if return_dict else mapped_result["mapped_rxn"])
        return results
