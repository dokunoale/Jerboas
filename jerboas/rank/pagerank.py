"""Random-walk importance, global or personalized to a seed set."""

import numpy as np

from .core import Strategy


class PageRank(Strategy):
    """Random-walk importance over the graph, computed once by power iteration.

    Global by default; personalized (random walk with restart) when `to=` seeds
    are given -- then the walk teleports back to the seed set instead of the
    whole graph, so a node scores by its proximity to the things you like. That
    makes PageRank(to=seeds) a principled sibling of Connectivity: same intent
    (relevance to the seeds), but as the stationary distribution of a restarting
    walk rather than a two-hop count.

    The walk runs on the graph's undirected adjacency, so mass flows both ways
    along every stored edge.

    `weighted=True` makes a step's probability proportional to the edge's stored
    score rather than uniform among the neighbours -- a 5-star rating carries
    more of the walker than a 1-star one. It reads the normalized weights and
    not the raw ones: a transition probability cannot be negative, and two
    relations' scales have to be reconciled before mass can flow between them.
    """

    def __init__(self, to=None, damping=0.85, iterations=100, tol=1e-6, weighted=False):
        self.to = to                    # optional seed set -> personalized (RWR)
        self.damping = damping
        self.iterations = iterations
        self.tol = tol
        self.weighted = weighted

    def fit(self, graph):
        seeds = tuple(sorted(graph.ids_of(self.to or ()).tolist()))
        self._ranks = self.cached(
            graph, ("pagerank", seeds, self.damping, self.iterations, self.weighted),
            lambda: self._power_iteration(graph, seeds))
        return self._ranks

    def scores(self, graph, columns):
        return self._ranks[columns[0]]

    def _power_iteration(self, graph, seeds):
        """Power iteration as repeated sparse matrix-vector products.

        One iteration pushes every node's rank along every edge, which is one
        sparse matrix-vector product: the inner loop lives in compiled code and
        only `iterations` numpy calls remain at Python level."""
        n = graph.n_nodes
        if n == 0:
            return np.zeros(0)

        adjacency = graph.adjacency("norm" if self.weighted else None)
        outdeg = np.asarray(adjacency.sum(axis=1)).ravel()
        dangling = np.flatnonzero(outdeg == 0)

        # teleport distribution: uniform, or concentrated on the seeds (RWR)
        teleport = np.zeros(n)
        if seeds:
            teleport[list(seeds)] = 1.0 / len(seeds)
        else:
            teleport[:] = 1.0 / n

        # transition[j, i] = the share of i's rank that flows to j. Scaling the
        # transpose column-wise by 1/outdeg does that in one sparse operation.
        share = np.zeros(n)
        np.divide(1.0, outdeg, out=share, where=outdeg > 0)
        transition = adjacency.T.multiply(share).tocsr()

        d = self.damping
        rank = teleport.copy()
        for _ in range(self.iterations):
            # rank stranded on dangling nodes has nowhere to flow, so it is
            # redistributed along the teleport distribution
            leaked = d * rank[dangling].sum()
            new_rank = (1.0 - d + leaked) * teleport + d * (transition @ rank)
            delta = np.abs(new_rank - rank).sum()
            rank = new_rank
            if delta < self.tol:
                break
        return rank
