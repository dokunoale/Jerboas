"""How gathered a node's neighbourhood is, rather than how large."""

import numpy as np

from .core import Strategy

# how many dimensions of the neighbours' sum exist at once
_DIMENSIONS = 8


class Concentration(Strategy):
    """Whether what a node touches points one way or every way, in `[0, 1]`.

    A degree says how many; this says how alike. Add the neighbours' vectors and
    compare the length of the sum with the sum of the lengths: pointing together
    the two are equal, scattered the sum cancels itself out. That resultant
    length is what tells a song in five hundred playlists *about the same thing*
    from a song in five hundred playlists about anything.

        .with_columns(gathered=Concentration(model).on("rec"))
        .filter(v.gathered >= 0.4)          # keep what belongs somewhere

    `space` is anything with `embeddings(graph)` -- a fitted factorization, a
    loaded model -- or an (n, d) array. `relation` picks which edges count;
    without one, every edge does, in either direction.

    Two sparse products over the whole graph, memoized: it costs the same for
    one node as for all of them, so it is worth asking of all of them. The memo
    lives on the space when the space is a strategy, because a model outlives
    the query that asks -- a service that writes `Concentration(model)` per
    request pays for the products once, not once a request."""

    def __init__(self, space, relation=None):
        self.space = space
        self.relation = relation

    def fit(self, graph):
        holder = self.space if isinstance(self.space, Strategy) else self
        self._gathered = holder.cached(graph, ("concentration", self.relation),
                                       lambda: self._compute(graph))
        return self._gathered

    def scores(self, graph, columns):
        return self._gathered[np.asarray(columns[0], dtype=np.int64)]

    def _compute(self, graph):
        vectors = self._vectors(graph)
        touching = self._touching(graph)
        mass = np.zeros(graph.n_nodes)
        for part in touching:
            mass += part @ np.linalg.norm(vectors, axis=1)
        # only the resultant's length is wanted, and a squared length is a sum
        # over dimensions -- so the (N, d) sum of neighbours is taken a few
        # dimensions at a time and never held whole
        squared = np.zeros(graph.n_nodes)
        for low in range(0, vectors.shape[1], _DIMENSIONS):
            block = np.ascontiguousarray(vectors[:, low:low + _DIMENSIONS])
            summed = touching[0] @ block
            for part in touching[1:]:
                summed += part @ block
            squared += np.einsum("ij,ij->i", summed, summed)
        resultant = np.sqrt(squared)
        return np.divide(resultant, mass, out=np.zeros_like(mass), where=mass > 0)

    def _vectors(self, graph):
        space = self.space
        block = space.embeddings(graph) if hasattr(space, "embeddings") else np.asarray(space)
        if len(block) != graph.n_nodes:
            raise ValueError(f"the space has {len(block)} rows and the graph has "
                             f"{graph.n_nodes} nodes")
        return block

    def _touching(self, graph):
        """Who counts as a neighbour, in either direction: a song's playlists
        are its neighbours whichever way the edge is stored.

        The two directions as two products rather than one symmetric matrix:
        `matrix.T` is a view, and `matrix + matrix.T` would be two more copies
        of the relation -- on the whole Spotify graph, the difference between a
        6 GB peak and one a container survives."""
        if self.relation is None:
            return (graph.adjacency(),)
        matrix = graph.relation_matrix(self.relation)
        return (matrix, matrix.T)
