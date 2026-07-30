"""TransD: knowledge-graph embedding with a dynamic projection.

    Ji et al., "Knowledge Graph Embedding via Dynamic Mapping Matrix", ACL 2015.

Every entity and relation carries two vectors: the embedding itself and a
projection vector. An entity is projected into a relation's space before the
translation is scored, so the same entity can sit differently under different
relations -- which is what lets a movie be close to its director under
`directed_by` and to its genre under `has_genre` at once.

The paper writes the projection as a matrix product; the identity

    p_r(e) = r_p (e_p . e) + e

computes the same thing without ever forming the matrix.

One departure from the usual formulation: there is no separate user embedding
table and no reserved "interaction" relation slot. In this graph a user is an
entity and `has_interact` is a relation like any other, so one entity table and
one relation table cover everything. That is a property of the data model, not a
simplification of the method -- and it means the same checkpoint scores a
recommendation and a KG completion with the same code path.
"""

import numpy as np
import torch
from torch import nn

from ..checkpoint import save as save_checkpoint

NAME = "transd"


def project(entity, entity_vec, relation_vec):
    """p_r(e) = r_p (e_p . e) + e -- the torch twin of strategies.embedding._project."""
    return relation_vec * (entity * entity_vec).sum(-1, keepdim=True) + entity


class TransD(nn.Module):
    """Entity and relation embeddings, plus their projection vectors."""

    name = NAME

    def __init__(self, factors=64, margin=1.0, seed=42):
        super().__init__()
        self.factors = factors
        self.margin = margin
        self.seed = seed
        self.entity = self.entity_vec = None
        self.relation = self.relation_vec = None
        self._graph = None

    def build(self, graph):
        """Size the tables to a graph. Separate from __init__ so a model can be
        described before the data it will be fitted to is loaded."""
        torch.manual_seed(self.seed)
        self._graph = graph
        self.entity = nn.Embedding(graph.n_nodes, self.factors)
        self.entity_vec = nn.Embedding(graph.n_nodes, self.factors)
        self.relation = nn.Embedding(len(graph.relations), self.factors)
        self.relation_vec = nn.Embedding(len(graph.relations), self.factors)
        for table in (self.entity, self.entity_vec, self.relation, self.relation_vec):
            nn.init.xavier_normal_(table.weight)
        return self

    def score(self, head, relation, tail):
        """-||p_r(h) + r - p_r(t)||, higher is more plausible."""
        relation_vec = self.relation_vec(relation)
        projected_head = project(self.entity(head), self.entity_vec(head), relation_vec)
        projected_tail = project(self.entity(tail), self.entity_vec(tail), relation_vec)
        translated = projected_head + self.relation(relation)
        return -torch.norm(translated - projected_tail, p=2, dim=-1)

    def loss(self, head, relation, tail, corrupt_head, corrupt_tail):
        """Margin ranking: a real triple must outscore its corruption by `margin`.

        Corrupting the head and the tail are separate examples rather than one
        averaged term, so a relation learns both of its directions."""
        positive = self.score(head, relation, tail)
        negative_tail = self.score(head, relation, corrupt_tail)
        negative_head = self.score(corrupt_head, relation, tail)
        return (torch.relu(self.margin - positive + negative_tail).mean()
                + torch.relu(self.margin - positive + negative_head).mean())

    def save(self, path):
        """Write a checkpoint that can be rebound to a graph by name."""
        tensors = {name: table.weight.detach().cpu().numpy().astype(np.float32)
                   for name, table in (("entity", self.entity),
                                       ("entity_vec", self.entity_vec),
                                       ("relation", self.relation),
                                       ("relation_vec", self.relation_vec))}
        save_checkpoint(path, self.name, self.factors, self._graph, tensors)
        return path
