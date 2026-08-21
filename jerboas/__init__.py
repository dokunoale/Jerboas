"""Jerboas: query a knowledge graph like a dataframe, rank it like a recommender.

Two objects, and the second is a dataframe:

    g = jb.Graph(kg=..., edges=[...], attrs=[...])
    print(g)                # <Graph: 15369 nodes in 5 types, 127k edges ...>
    print(g.nodes("movie")) # shape: (1682, 1) -- a table, with a graph behind it

`Graph` is the data: a typed, positional node store plus an edge-labeled CSR.
`Frame` is what you ask it -- a polars DataFrame that knows which of its columns
hold nodes -- with three verbs of its own (`hop`, `attrs`, `labels`) and the
rest of polars forwarded. A pattern variable is a column name, so two of them
are the same variable when spelled the same.

    seeds = g.nodes(artist="artist").filter(v.artist.label.like("Golden",
                                                               rule=Words(k=3)))
    (g.nodes(seed=seeds).hop(song="~performed_by")
       .with_columns(score=PageRank(to=seeds).on("song"))
       .top(5).labels("song"))

Everything a graph can be asked besides walking it is a condition, and every
condition goes in `filter`. What a search measures with is a `Rule` (rules.py);
what a ranking is computed with is a `Strategy`, and a strategy is a column, so
combining several is arithmetic that is written down.

The embedding models (TransD, TransE) are strategies like the rest, imported on
demand: fitting one needs torch, and the base install stays importable without
it.
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
    Concentration,
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
    "Concentration", "Connectivity", "MatrixFactorization", "DiffusedMatrixFactorization",
    "PageRank", "Weight",
    "TransD", "TransE",
]
