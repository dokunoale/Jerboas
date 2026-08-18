"""Jerboas: a non-deterministic-first, ORM-like graph query library.

Public surface, grouped by the four object families (see core.py):

    references   Node, Edge, Path            -> select(...)
                 Sum, Count, Mean, Min, Max  -> rank(...), over what matched
    conditions   Like, In, Has, Match,       -> where(...)
                 And, Or, Not
    strategies   Score, Ascending, Descending, PageRank,        -> rank(...)
                 MatrixFactorization, DiffusedMatrixFactorization,
                 Weight, TransD, TransE
    engines      Default, Greedy             -> using(...)

Plus Graph (the data + `select`), the values a query returns (Key, Rel), and the
interfaces (Ref, Condition, Strategy, Engine) for extending any family.
Comparisons on refs (`node.year >= 1990`, `movie.directed_by == Node("person")`)
build conditions implicitly; everything else is an explicit object.

A relation has one name and two directions: `person.directed_by.inverse` walks
it backwards, and the wildcard `Edge()` walks both.

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

from .core import Ref, Expr, Condition, Strategy, Engine
from .graph import Graph
from .keys import Key, Rel
from .refs import (Node, Edge, Path, Attr, Degree, EdgeScore,
                   Sum, Count, Mean, Min, Max)
from .conditions import Like, In, Has, Compare, Match, And, Or, Not
from .strategies import (
    Score,
    ExprStrategy,
    Ascending,
    Descending,
    MatrixFactorization,
    DiffusedMatrixFactorization,
    Connectivity,
    PageRank,
    Weight,
)
from .engine import Default, Greedy

__all__ = [
    # interfaces
    "Ref", "Expr", "Condition", "Strategy", "Engine",
    # data + entry point
    "Graph",
    # result values
    "Key", "Rel",
    # references
    "Node", "Edge", "Path", "Attr", "Degree", "EdgeScore",
    "Sum", "Count", "Mean", "Min", "Max",
    # conditions
    "Like", "In", "Has", "Compare", "Match", "And", "Or", "Not",
    # strategies
    "Score", "ExprStrategy", "Ascending", "Descending",
    "MatrixFactorization", "DiffusedMatrixFactorization", "Connectivity", "PageRank",
    "Weight",
    "TransD", "TransE",
    # engines
    "Default", "Greedy",
]
