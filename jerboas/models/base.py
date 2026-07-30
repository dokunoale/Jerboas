"""What every translational embedding shares.

TransE, TransD and their relatives differ in exactly one thing: how a triple is
scored. The tables they allocate, the margin-ranking objective, and the
checkpoint they write are the same, so they live here and a model is left as its
score function plus the list of tables it needs.

Table naming is a convention rather than a declaration: a table whose name starts
with `relation` has one row per relation, anything else has one row per node.
checkpoint.py splits them the same way, which is what lets a checkpoint rebind
node rows by identity while leaving relation rows to be matched by name.
"""

import numpy as np
import torch
from torch import nn

from ..checkpoint import save as save_checkpoint


class Translational(nn.Module):
    """A scoring model of the form -||f(h) + r - f(t)||."""

    name = None          # the key a checkpoint records, and SCORERS looks up
    tables = ()          # embedding tables to allocate

    def __init__(self, factors=64, margin=1.0, seed=42):
        super().__init__()
        self.factors = factors
        self.margin = margin
        self.seed = seed
        self._graph = None

    @property
    def built(self):
        return self._graph is not None

    def build(self, graph):
        """Size the tables to a graph. Separate from __init__ so a model can be
        described before the data it will be fitted to is loaded."""
        torch.manual_seed(self.seed)
        self._graph = graph
        for name in self.tables:
            rows = len(graph.relations) if name.startswith("relation") else graph.n_nodes
            table = nn.Embedding(rows, self.factors)
            nn.init.xavier_normal_(table.weight)
            setattr(self, name, table)
        return self

    def score(self, head, relation, tail):
        """-||...||, higher is more plausible. One line per model."""
        raise NotImplementedError

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

    def save(self, path):
        """Write a checkpoint that can be rebound to a graph by name."""
        tensors = {name: getattr(self, name).weight.detach().cpu().numpy().astype(np.float32)
                   for name in self.tables}
        save_checkpoint(path, self.name, self.factors, self._graph, tensors)
        return path
