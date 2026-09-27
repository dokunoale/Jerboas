"""Ranking by what the data already said: the weight stored on the edge.

The weight of a single traversed edge is already a column: it comes back from
`hop` as `<edge>.score`, so a walk's average step is
`(v.a.score + v.b.score) / 2` and its weakest is a `min_horizontal` -- written
out, in the frame, where it can be read.

What is left is the one reading no single row holds: how much weight the graph
put on a node *in total*. That is popularity measured in weight rather than in
edges, which `count()` cannot say and no traversal reveals.
"""

import numpy as np

from .core import Strategy


class Weight(Strategy):
    """Score a node by the total weight of the edges incident to it.

        .with_columns(w=Weight().on("rec"))
        .with_columns(w=Weight("has_interact").on("rec"))

    Weights are normalized by default (min-max within their relation): raw
    scores are unbounded and per-relation, so summing them across relations is
    arithmetic on incomparable units. Pass normalized=False when the raw number
    is the quantity you mean.
    """

    def __init__(self, relation=None, normalized=True):
        self.relation = relation
        self.normalized = normalized

    def scores(self, graph, columns):
        incident = self.cached(graph, ("incident", self.relation, self.normalized),
                               lambda: self._incident(graph))
        return incident[columns[0]]

    def _incident(self, graph):
        weights = "norm" if self.normalized else "raw"
        if self.relation is None:
            return np.asarray(graph.adjacency(weights).sum(axis=1)).ravel()
        matrix = graph.relation_matrix(self.relation, weights)
        return (np.asarray(matrix.sum(axis=1)).ravel()
                + np.asarray(matrix.sum(axis=0)).ravel())
