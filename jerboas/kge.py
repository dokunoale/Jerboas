"""What a knowledge-graph embedding *is* -- independent of how it is fitted or
stored.

A model here is three things that must never disagree: its name, the tables it
allocates, and the arithmetic that turns those tables into a plausibility. They
live in one frozen object, so a checkpoint saying `model: "transd"` determines
both the shape of its weights and how to read them. There is nothing to keep in
sync, and therefore nothing that can drift.

Two consumers, one definition:

    jerboas.models       fits it with torch (the tables are nn.Embedding)
    strategies.Embedding ranks with it using numpy (the tables are arrays)

The score functions are written using only operators numpy and torch spell the
same way -- `*`, `+`, `-`, `.sum(-1)`, `[..., None]`, `**` -- so one function
serves both. That is deliberately narrower than either library: `keepdim` vs
`keepdims` and `torch.norm` vs `np.linalg.norm` are the two places they diverge,
and both are avoidable. The cost is that `_norm` cannot rescale the way a
library norm does, so it would overflow for magnitudes beyond ~1e19 in float32;
embeddings are O(1), so this does not arise.

Gathering is *not* shared, because it genuinely differs: torch calls an
nn.Embedding, numpy takes a fancy index. Each side passes its own `get`, and the
maths above it stays one implementation.
"""

from dataclasses import dataclass
from typing import Callable

# the two index spaces a weight table can live in
NODE = "node"
RELATION = "relation"


@dataclass(frozen=True)
class Table:
    """One weight table: its name in the checkpoint, and what its rows count."""

    name: str
    space: str          # NODE | RELATION


@dataclass(frozen=True)
class Model:
    """A model's complete identity: name, shape, and arithmetic."""

    name: str
    tables: tuple
    score: Callable     # (get, head, relation, tail) -> plausibility

    def space(self, name):
        for table in self.tables:
            if table.name == name:
                return table.space
        raise KeyError(f"{self.name} has no table {name!r}")

    @property
    def names(self):
        return tuple(table.name for table in self.tables)


def _norm(difference):
    """-||d||, in the intersection of the two array libraries."""
    return -((difference * difference).sum(-1) ** 0.5)


def transd(get, head, relation, tail):
    """TransD: project each entity into the relation's space, then translate.

    p_r(e) = r_p (e_p . e) + e, which is the paper's mapping matrix without ever
    forming the matrix."""
    entity_head, vec_head = get("entity", head), get("entity_vec", head)
    entity_tail, vec_tail = get("entity", tail), get("entity_vec", tail)
    translation, projector = get("relation", relation), get("relation_vec", relation)

    projected_head = projector * (entity_head * vec_head).sum(-1)[..., None] + entity_head
    projected_tail = projector * (entity_tail * vec_tail).sum(-1)[..., None] + entity_tail
    return _norm(projected_head + translation - projected_tail)


def transe(get, head, relation, tail):
    """TransE: the relation is a translation, and a triple holds when head plus
    relation lands on tail."""
    return _norm(get("entity", head) + get("relation", relation) - get("entity", tail))


TRANSD = Model(
    name="transd",
    tables=(Table("entity", NODE), Table("entity_vec", NODE),
            Table("relation", RELATION), Table("relation_vec", RELATION)),
    score=transd,
)

TRANSE = Model(
    name="transe",
    tables=(Table("entity", NODE), Table("relation", RELATION)),
    score=transe,
)

REGISTRY = {model.name: model for model in (TRANSD, TRANSE)}
