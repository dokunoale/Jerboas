"""Fitting the models that kge.py defines.

A model class here carries no maths and no shape of its own: both come from its
`spec`, a kge.Model. This module supplies what only training needs -- allocating
the tables as nn.Embedding, the margin objective, and writing the checkpoint.

So adding a model is adding a kge.Model plus three lines here. There is no torch
scoring function to keep in step with a numpy one, because there is only one
scoring function and it lives in kge.py.
"""

import torch
from torch import nn

from ..checkpoint import provenance, save as save_checkpoint
from ..kge import NODE


class Translational(nn.Module):
    """Fits any kge.Model whose score is a translation to be minimised."""

    spec = None          # a kge.Model

    def __init__(self, factors=64, margin=1.0, seed=42):
        super().__init__()
        self.factors = factors
        self.margin = margin
        self.seed = seed
        self.tables = nn.ModuleDict()
        self._graph = None
        self.meta = {}

    @property
    def name(self):
        return self.spec.name

    @property
    def built(self):
        return self._graph is not None

    def build(self, graph):
        """Size the tables to a graph. Separate from __init__ so a model can be
        described before the data it will be fitted to is loaded."""
        torch.manual_seed(self.seed)
        self._graph = graph
        for table in self.spec.tables:
            rows = graph.n_nodes if table.space == NODE else len(graph.relations)
            embedding = nn.Embedding(rows, self.factors)
            nn.init.xavier_normal_(embedding.weight)
            self.tables[table.name] = embedding
        return self

    def gather(self, table, index):
        return self.tables[table](index)

    def score(self, head, relation, tail):
        """The model's own arithmetic, from kge.py -- the same function the numpy
        serving path calls, differing only in how the vectors are fetched."""
        return self.spec.score(self.gather, head, relation, tail)

    def loss(self, head, relation, tail, corrupt_head, corrupt_tail):
        """Margin ranking: a real triple must outscore its corruption by `margin`.

        Corrupting the head and the tail are separate terms rather than one
        averaged example, so a relation is pushed to learn both of its
        directions -- which matters here because the graph stores each edge once
        and every relation is read in both."""
        positive = self.score(head, relation, tail)
        negative_tail = self.score(head, relation, corrupt_tail)
        negative_head = self.score(corrupt_head, relation, tail)
        return (torch.relu(self.margin - positive + negative_tail).mean()
                + torch.relu(self.margin - positive + negative_head).mean())

    def save(self, path, **details):
        """Write a checkpoint that can be rebound to any graph, by name."""
        if not self.built:
            raise ValueError("nothing to save: the model has not been built or fitted")
        weights = {name: table.weight.detach().cpu().numpy()
                   for name, table in self.tables.items()}
        meta = provenance(self._graph, model=self.name, factors=self.factors,
                          margin=self.margin, seed=self.seed, **self.meta, **details)
        return save_checkpoint(path, self.spec, self.factors, self._graph, weights, meta)
