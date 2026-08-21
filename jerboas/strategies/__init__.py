"""Ranking strategies: the scores a column cannot hold on its own.

A strategy is what is left when everything a dataframe already does is taken
away: sorting by a stored value is `sort`, ranking by an edge's weight is a
column, counting matches is a `group_by`. What remains has to be computed *from
the graph* -- a random walk, a factorization, an embedding.

    frame.with_columns(pr=PageRank(to=seeds).on("rec").norm(),
                       kg=TransD.load(path, g, to=seeds).on("rec").norm())
         .with_columns(score=0.7 * v.pr + 0.3 * v.kg)

Each names the columns it reads (`on`) rather than guessing them from the shape
of the query -- the difference between a strategy that knows whose taste it is
modelling and one that picks a user out of whatever the search walked through.

    concentration          Concentration    whether a neighbourhood points one way
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

from .concentration import Concentration
from .connectivity import Connectivity
from .matrix_factorization import MatrixFactorization, DiffusedMatrixFactorization
from .pagerank import PageRank
from .weight import Weight

__all__ = ["Concentration", "Connectivity", "MatrixFactorization", "DiffusedMatrixFactorization",
           "PageRank", "Weight"]
