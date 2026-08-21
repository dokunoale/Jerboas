"""How gathered a node's neighbourhood is, rather than how large."""

import numpy as np

from ..core import Strategy


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
    one node as for all of them, so it is worth asking of all of them."""

    def __init__(self, space, relation=None):
        self.space = space
        self.relation = relation

    def fit(self, graph):
        self._gathered = self.cached(graph, ("concentration", self.relation),
                                     lambda: self._compute(graph))
        return self._gathered

    def scores(self, graph, columns):
        return self._gathered[np.asarray(columns[0], dtype=np.int64)]

    def _compute(self, graph):
        vectors = self._vectors(graph)
        touching = self._touching(graph)
        resultant = np.linalg.norm(touching @ vectors, axis=1)
        mass = touching @ np.linalg.norm(vectors, axis=1)
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
        are its neighbours whichever way the edge is stored."""
        if self.relation is None:
            return graph.adjacency()
        matrix = graph.relation_matrix(self.relation)
        return matrix + matrix.T
