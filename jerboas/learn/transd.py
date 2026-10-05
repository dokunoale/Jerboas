"""TransD: knowledge-graph embedding with a dynamic projection.

    Ji et al., "Knowledge Graph Embedding via Dynamic Mapping Matrix", ACL 2015.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

Every entity and relation carries an embedding and a projection vector. An
entity is projected into a relation's space before the translation is scored, so
a movie can be close to its director under `directed_by` and to its genre under
`has_genre` at once. The paper writes the projection as a matrix product; the
identity

    p_r(e) = r_p (e_p . e) + e

computes it without ever forming the matrix.

One departure from the reference implementation: no separate user table and no
relation slot reserved for interactions. A user is an entity and `has_interact`
is a relation like any other, so one entity table and one relation table cover
recommendation and KG completion with the same code path.
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
