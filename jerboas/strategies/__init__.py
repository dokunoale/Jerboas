"""Ranking strategies: the soft, non-deterministic-first scoring layer.

Every ranker is a Strategy (core.py): fit() once, score() per run, optional
edge_weight for guided engines. Where a Condition hard-keeps or drops a row, a
Strategy orders rows by a learned or computed affinity, and several combine
(each min-max normalized, then averaged) so a soft signal rewards rather than
excludes.

One module per family, because this is the family users extend most:

    basic                  Score, ExprStrategy, Alphabetical
    matrix_factorization   MatrixFactorization, DiffusedMatrixFactorization
    connectivity           Connectivity
    pagerank               PageRank

Nodes are integers throughout, which is what makes an embedding table a single
(N, factors) array a strategy can index directly.

fit() means "prepare to score", and is expected to cost milliseconds. A model
whose training is orders of magnitude slower than that lives in jerboas.models,
where it is a Strategy too -- fitted by an explicit batch job, then loaded from a
checkpoint and passed to rank(...) like any other.
"""

from .basic import Score, ExprStrategy, Alphabetical
from .connectivity import Connectivity
from .matrix_factorization import MatrixFactorization, DiffusedMatrixFactorization
from .pagerank import PageRank

__all__ = [
    "Score", "ExprStrategy", "Alphabetical",
    "MatrixFactorization", "DiffusedMatrixFactorization",
    "Connectivity", "PageRank",
]
