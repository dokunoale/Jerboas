"""Ranking strategies: the scores a column cannot hold on its own.

A strategy is what is left when everything a dataframe already does is taken
away. Sorting by a stored value is `sort`, ranking by a matched edge's weight is
a column, counting the matches is `group_by(...).agg(...)` -- none of those is a
strategy any more, and none of them needs to be. What remains is the family of
scores that have to be computed *from the graph*: a random walk, a
factorization, a two-hop reachability count, an embedding.

    frame.with_columns(pr=PageRank(to=seeds).on("rec").norm(),
                       kg=TransD.load(path, g, to=seeds).on("rec").norm())
         .with_columns(score=0.7 * v.pr + 0.3 * v.kg)

Each one names the columns it reads (see core.Strategy.on) instead of guessing
them from the shape of the query, which is the difference between a strategy
that knows whose taste it is modelling and one that picks a user out of whatever
the search happened to walk through.

    connectivity           Connectivity     two-hop reachability from a seed set
    pagerank               PageRank         random-walk importance, global or personalized
    matrix_factorization   MatrixFactorization, DiffusedMatrixFactorization
    weight                 Weight           the weight the data already put on a node

Nodes are integers throughout, which is what makes an embedding table a single
(N, factors) array a strategy can index directly.

fit() means "prepare to score", and is expected to cost milliseconds. A model
whose training is orders of magnitude slower than that lives in jerboas.models,
where it is a Strategy too -- fitted by an explicit batch job, then loaded from a
checkpoint and used in a column like any other.
"""

from .connectivity import Connectivity
from .matrix_factorization import MatrixFactorization, DiffusedMatrixFactorization
from .pagerank import PageRank
from .weight import Weight

__all__ = ["Connectivity", "MatrixFactorization", "DiffusedMatrixFactorization",
           "PageRank", "Weight"]
