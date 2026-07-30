"""Two-hop reachability from a seed set, counted with a bitset."""

from ..core import Strategy


class Connectivity(Strategy):
    """Soft filter: score a candidate by how many distinct seed nodes reach it
    within two hops. Rewards well-connected results without hard-excluding the
    rest. fit() walks outward from the (few) seeds, not the (many) candidates."""

    def __init__(self, to):
        self.to = to

    def fit(self, graph):
        seeds = tuple(sorted(i for i in (graph.lookup(s) for s in self.to) if i is not None))
        self._counts = self.cached(graph, ("connectivity", seeds),
                                   lambda: self._compute_counts(graph, seeds))

    def _compute_counts(self, graph, seeds):
        # Walking out from each seed separately re-expands every intermediate node
        # once per seed that reaches it -- and popular intermediates are reached by
        # nearly all of them. So carry the seed set along instead, packed into an
        # int used as a bitset: hop 2 then visits each distinct intermediate once,
        # and the final count per node is just that int's population count.
        #
        # This stays a bitset rather than a boolean sparse product because the
        # product needs a seeds x nodes intermediate, which is fine at 200 seeds
        # and untenable at 15000; the bitset's per-node state is one integer.
        hop1 = {}
        for i, seed in enumerate(seeds):
            bit = 1 << i
            for target in graph.neighbours(seed):
                hop1[target] = hop1.get(target, 0) | bit

        reached = dict(hop1)
        for node, mask in hop1.items():
            for target in graph.neighbours(node):
                reached[target] = reached.get(target, 0) | mask

        return {node: mask.bit_count() for node, mask in reached.items()}

    def score(self, query, rows):
        col = query.primary_column
        counts = getattr(self, "_counts", None) or {}
        return [float(counts.get(row[col], 0)) for row in rows]
