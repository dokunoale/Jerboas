"""TransD: knowledge-graph embedding with a dynamic projection.

    Ji et al., "Knowledge Graph Embedding via Dynamic Mapping Matrix", ACL 2015.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

Every entity and relation carries two vectors: the embedding, and a projection
vector. An entity is projected into a relation's space before the translation is
scored, so the same entity can sit differently under different relations -- which
is what lets a movie be close to its director under `directed_by` and to its
genre under `has_genre` at once.

The paper writes the projection as a matrix product; the identity

    p_r(e) = r_p (e_p . e) + e

computes the same thing without ever forming the matrix.

One departure from the reference implementation: there is no separate user
embedding table and no relation slot reserved for interactions. Here a user is an
entity and `has_interact` is a relation like any other, so one entity table and
one relation table cover both recommendation and KG completion with the same code
path. That follows from the data model rather than simplifying the method.
"""

from .base import NODE, RELATION, Translational


class TransD(Translational):
    name = "transd"
    tables = (("entity", NODE), ("entity_vec", NODE),
              ("relation", RELATION), ("relation_vec", RELATION))

    def plausibility(self, head, relation, tail):
        entity_head, vec_head = self.get("entity", head), self.get("entity_vec", head)
        entity_tail, vec_tail = self.get("entity", tail), self.get("entity_vec", tail)
        translation = self.get("relation", relation)
        projector = self.get("relation_vec", relation)

        projected_head = projector * (entity_head * vec_head).sum(-1)[..., None] + entity_head
        projected_tail = projector * (entity_tail * vec_tail).sum(-1)[..., None] + entity_tail
        return self.norm(projected_head + translation - projected_tail)
