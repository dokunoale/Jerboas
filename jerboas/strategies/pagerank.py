"""Random-walk importance, global or personalized to a seed set."""

import numpy as np

from ..core import Strategy


class PageRank(Strategy):
    """Random-walk importance over the graph, computed once by power iteration.

    Global by default; personalized (random walk with restart) when `to=` seeds
    are given -- then the walk teleports back to the seed set instead of the
    whole graph, so a node scores by its proximity to the things you like. That
    makes PageRank(to=seeds) a principled sibling of Connectivity: same intent
    (relevance to the seeds), but as the stationary distribution of a restarting
    walk rather than a two-hop count.

    The walk runs on the graph's undirected adjacency, so mass flows both ways
    along every stored edge. supports_guidance is on: edge_weight returns the
    target's rank, so a Greedy engine prefers important nodes.
    """

    supports_guidance = True

    def __init__(self, to=None, damping=0.85, iterations=100, tol=1e-6):
        self.to = to                    # optional seed set -> personalized (RWR)
        self.damping = damping
        self.iterations = iterations
        self.tol = tol

    def fit(self, graph):
        seeds = tuple(sorted(i for i in (graph.lookup(s) for s in (self.to or ()))
                             if i is not None))
        self._ranks = self.cached(graph, ("pagerank", seeds, self.damping, self.iterations),
                                  lambda: self._power_iteration(graph, seeds))
        return self._ranks

    def edge_weight(self, source, relation, target):
        return float(self._ranks[target])

    def score(self, query, rows):
        ranks, col = self._ranks, query.primary_column
        return [float(ranks[row[col]]) for row in rows]

    def _power_iteration(self, graph, seeds):
        """Power iteration as repeated sparse matrix-vector products.

        One iteration pushes every node's rank along every out-edge, which written
        by hand is a Python loop over all E edges -- repeated `iterations` times.
        That is exactly what a sparse matrix-vector product does, so the whole
        inner loop lives in compiled code and only `iterations` numpy calls
        remain at Python level."""
        n = graph.n_nodes
        if n == 0:
            return np.zeros(0)

        adjacency = graph.adjacency()
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
