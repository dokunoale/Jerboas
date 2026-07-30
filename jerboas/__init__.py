"""Jerboas: a non-deterministic-first, ORM-like graph query library.

Public surface, grouped by the four object families (see core.py):

    references   Node, Edge, Path            -> select(...)
    conditions   Like, In, Has, Match,       -> where(...)
                 And, Or, Not
    strategies   Score, Alphabetical, PageRank, Connectivity,   -> rank(...)
                 MatrixFactorization, DiffusedMatrixFactorization
    engines      Default, Greedy             -> using(...)

Plus Graph (the data + `select`), the values a query returns (Key, Rel), and the
interfaces (Ref, Condition, Strategy, Engine) for extending any family.
Comparisons on refs (`node.year >= 1990`, `movie.directed_by == Node("person")`)
build conditions implicitly; everything else is an explicit object.

A relation has one name and two directions: `person.directed_by.inverse` walks
it backwards, and the wildcard `Edge()` walks both.
"""

from .core import Ref, Expr, Condition, Strategy, Engine
from .graph import Graph
from .keys import Key, Rel
from .refs import Node, Edge, Path, Attr, Degree
from .conditions import Like, In, Has, Compare, Match, And, Or, Not
from .strategies import (
    Score,
    ExprStrategy,
    Alphabetical,
    MatrixFactorization,
    DiffusedMatrixFactorization,
    Connectivity,
    PageRank,
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
    "Node", "Edge", "Path", "Attr", "Degree",
    # conditions
    "Like", "In", "Has", "Compare", "Match", "And", "Or", "Not",
    # strategies
    "Score", "ExprStrategy", "Alphabetical",
    "MatrixFactorization", "DiffusedMatrixFactorization", "Connectivity", "PageRank",
    # engines
    "Default", "Greedy",
]
