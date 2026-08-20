"""Jerboas: query a knowledge graph like a dataframe, rank it like a recommender.

Two objects, and the second is a dataframe:

    g = jb.Graph(kg=..., edges=[...], attrs=[...])
    print(g)                # <Graph: 15369 nodes in 5 types, 127k edges ...>
    print(g.nodes("movie")) # shape: (1682, 1) -- a table, with a graph behind it

`Graph` is the data structure: a typed, positional node store plus an
edge-labeled CSR. `Frame` is what you ask it -- a polars DataFrame that knows
which of its columns hold nodes, and adds the four verbs a table cannot get from
being a table:

    hop      one traversal, one result row per edge
    like     graded membership over a text column -- the search box
    attrs    a stored attribute as a column
    labels   the column a person reads, per the graph's `readable` map

Everything else is polars: `filter`, `with_columns`, `group_by`, `sort`, `join`,
and `.pl` for whatever is not forwarded. A pattern variable is a column name --
`v.rec` is the column `rec`, `v.rec.year` the column `rec.year` -- so two of them
are the same variable when they are spelled the same, and the frame prints what
it is holding.

    seeds = g.nodes("artist").labels("artist").like(v.artist.label, "Golden", k=3)

    (g.nodes(seed=seeds).hop(to="song", type="song")
       .with_columns(score=PageRank(to=seeds).on("song"))
       .top(5).attrs(song="name"))

A `Strategy` is the one thing a column cannot be: a score computed from the
graph -- a walk, a factorization, an embedding. `on(...)` names the columns it
reads, and the frame turns it into a column, which is then ordinary arithmetic:

    .with_columns(pr=PageRank(to=seeds).on("rec").norm(),
                  kg=TransD.load(path, g, to=seeds).on("rec").norm())
    .with_columns(score=0.7 * v.pr + 0.3 * v.kg)

Nothing is combined behind your back, and every intermediate signal stays a
column you can print.

The embedding models (TransD, TransE) are strategies like the rest, but they are
imported on demand: fitting one needs torch, which is an optional extra, and the
base install must stay importable without it.
"""

_LAZY = {"TransD": "models", "TransE": "models", "Translational": "models",
         "train": "models"}


def __getattr__(name):
    """Reach jerboas.models only when something in it is actually asked for."""
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .core import Strategy, Signal
from .expr import SCORE, VIA, col, norm, reverse, shadow, v
from .frame import Frame, concat
from .graph import Graph
from .keys import Key
from .optimize import optimize
from .rules import Fuzzy, Rule, Search, Semantic, Words
from .strategies import (
    Connectivity,
    DiffusedMatrixFactorization,
    MatrixFactorization,
    PageRank,
    Weight,
)

__all__ = [
    # the data, and what you ask it
    "Graph", "Frame", "concat",
    # naming a column, and its confidence
    "v", "col", "norm", "reverse", "shadow", "SCORE", "VIA",
    # a node, outside the frame
    "Key",
    # deferring a walk so its cost has a ceiling
    "optimize",
    # how a search decides what is close
    "Rule", "Search", "Fuzzy", "Words", "Semantic",
    # strategies: the scores a column cannot hold on its own
    "Strategy", "Signal",
    "Connectivity", "MatrixFactorization", "DiffusedMatrixFactorization",
    "PageRank", "Weight",
    "TransD", "TransE",
]
