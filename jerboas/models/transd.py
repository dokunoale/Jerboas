"""TransD: knowledge-graph embedding with a dynamic projection.

    Ji et al., "Knowledge Graph Embedding via Dynamic Mapping Matrix", ACL 2015.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

Every entity and relation carries two vectors: the embedding itself and a
projection vector. An entity is projected into a relation's space before the
translation is scored, so the same entity can sit differently under different
relations -- which is what lets a movie be close to its director under
`directed_by` and to its genre under `has_genre` at once.

The paper writes the projection as a matrix product; the identity

    p_r(e) = r_p (e_p . e) + e

computes the same thing without ever forming the matrix.

One departure from the reference implementation: there is no separate user
embedding table and no relation slot reserved for interactions. Here a user is an
entity and `has_interact` is a relation like any other, so one entity table and
one relation table cover both recommendation and KG completion -- with the same
code path, not two. That follows from the data model rather than simplifying the
method.
"""

import torch

from .base import Translational


def project(entity, entity_vec, relation_vec):
    """p_r(e) = r_p (e_p . e) + e -- the torch twin of strategies.embedding._project."""
    return relation_vec * (entity * entity_vec).sum(-1, keepdim=True) + entity


class TransD(Translational):
    name = "transd"
    tables = ("entity", "entity_vec", "relation", "relation_vec")

    def score(self, head, relation, tail):
        relation_vec = self.relation_vec(relation)
        projected_head = project(self.entity(head), self.entity_vec(head), relation_vec)
        projected_tail = project(self.entity(tail), self.entity_vec(tail), relation_vec)
        translated = projected_head + self.relation(relation)
        return -torch.norm(translated - projected_tail, p=2, dim=-1)
