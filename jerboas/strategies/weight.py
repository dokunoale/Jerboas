"""Ranking by what the data already said: the weight stored on the edge.

Every other strategy here infers an affinity -- a factorization, a walk, an
embedding. This one reads the number that was in the file. It is the right
ranker exactly when the judgement is already in the data (a rating, a
confidence, a duration) and the wrong one when it has to be discovered.

It is also the strategy that makes the load-time score filter unnecessary.
Dropping an edge below a threshold answers "is this good enough?" once, for
every query; ranking by its weight answers "which of these is better?" per
query, and keeps the graph honest about what it holds.
"""

import numpy as np

from ..core import Strategy

# how the weights along a walk become one number. `min` is the weakest link (a
# chain is as good as its worst step), `mean` the average step, `product` the
# compounding reading a normalized weight invites.
AGGREGATE = {
    "mean": lambda values: float(np.mean(values)),
    "min": lambda values: float(np.min(values)),
    "max": lambda values: float(np.max(values)),
    "sum": lambda values: float(np.sum(values)),
    "product": lambda values: float(np.prod(values)),
}


class Weight(Strategy):
    """Score a row by the weights of the edges it was reached through.

        rank(Weight())                     # the walk's average edge weight
        rank(Weight("has_interact"))       # only that relation's edges count
        rank(Weight(how="min"))            # as good as its weakest step

    With a Path in select(...), the score aggregates the weights actually
    traversed. Without one there is no walk to read, so it falls back to the
    node's total incident weight -- popularity measured in weight rather than in
    edges, which is what `count()` cannot say.

    Weights are normalized by default (min-max within their relation): raw
    scores are unbounded and per-relation, so summing or comparing them across
    relations is arithmetic on incomparable units. Pass normalized=False when
    the raw number is the quantity you mean.

    supports_guidance is on, and here it is exact rather than a proxy: what
    guides a Greedy beam is the very weight the ranking will use.
    """

    supports_guidance = True

    def __init__(self, relation=None, how="mean", normalized=True):
        self.relation = relation
        self.how = how
        self.normalized = normalized

    def fit(self, graph):
        self._graph = graph
        if self.how not in AGGREGATE:
            raise ValueError(f"unknown aggregate {self.how!r}; expected one of {sorted(AGGREGATE)}")

    def edge_weight(self, source, relation, target):
        return self._weight(source, target, relation)

    def score(self, query, rows):
        if not rows:
            return []
        path_col = query.path_column
        if path_col is None:
            incident = self._incident(query.graph)
            return [float(incident[row[query.primary_column]]) for row in rows]
        aggregate = AGGREGATE[self.how]
        return [aggregate(self._walk(row[path_col]) or [0.0]) for row in rows]

    def _walk(self, walk):
        """The weight of every step of one path row (node, code, node, ...)."""
        return [self._weight(walk[i], walk[i + 2], walk[i + 1])
                for i in range(0, len(walk) - 2, 2)]

    def _weight(self, source, target, code):
        """One traversed edge's weight. A negative code means the edge was walked
        against the direction it is stored in, so the lookup has to be too; None
        is a wildcard traversal, where either direction may have produced it."""
        wanted = None if self.relation is None else self._graph.relation_code(self.relation)
        if self.relation is not None and wanted is None:
            return 0.0                          # a relation the graph never saw
        if code is None:                        # unrecorded traversal: ask either way
            return self._graph.weight_of(source, target, wanted, None, self.normalized)
        relation, reverse = (~code, True) if code < 0 else (code, False)
        if wanted is not None and relation != wanted:
            return 0.0                          # a step of some other relation weighs nothing
        return self._graph.weight_of(source, target, relation, reverse, self.normalized)

    def _incident(self, graph):
        """Total weight on a node's edges, both directions -- the fallback when
        the query kept no walk to read."""
        return self.cached(graph, ("incident", self.relation, self.normalized),
                           lambda: self._compute_incident(graph))

    def _compute_incident(self, graph):
        weights = "norm" if self.normalized else "raw"
        if self.relation is None:
            return np.asarray(graph.adjacency(weights).sum(axis=1)).ravel()
        matrix = graph.relation_matrix(self.relation, weights)
        return (np.asarray(matrix.sum(axis=1)).ravel()
                + np.asarray(matrix.sum(axis=0)).ravel())
