"""Ranking by a trained knowledge-graph embedding.

One strategy, every model: Embedding does not know whether the checkpoint holds
TransD or TransE. The checkpoint names its model, kge.REGISTRY resolves that name
to the arithmetic, and the same function the trainer optimised is the one that
ranks here -- not a reimplementation of it.

numpy only, by design. Fitting an embedding needs torch; using one needs a
projection and a norm, so a deployment installs the base package.
"""

import numpy as np

from ..checkpoint import load as load_checkpoint
from ..core import Strategy

INTERACT = "has_interact"


class Embedding(Strategy):
    """Rank candidates by how plausible a trained embedding finds the edge that
    would connect them to a seed.

    A row scores `-||proj(h) + r - proj(t)||`: the model's own measure of whether
    that triple belongs in the graph. Two readings, chosen the way
    DiffusedMatrixFactorization chooses its own:

        Embedding.load(path, g, to=seeds)   the best score against any seed
        Embedding.load(path, g)             each row against its own path seed

    `reverse=True` scores seed-as-tail instead of seed-as-head, which is what a
    relation stored the other way round needs: `directed_by` runs movie -> person,
    so ranking movies for a person seed reads it backwards.

    edge_weight is the same quantity on a single edge, which makes this the one
    strategy where guiding a Greedy engine is the model's actual objective rather
    than a proxy for it.
    """

    supports_guidance = True

    def __init__(self, checkpoint, to=None, relation=INTERACT, reverse=False):
        self.checkpoint = checkpoint
        self.to = to
        self.relation = relation
        self.reverse = reverse
        self._seeds = None
        self._code = None

    @classmethod
    def load(cls, path, graph, to=None, relation=INTERACT, reverse=False):
        return cls(load_checkpoint(path, graph), to=to, relation=relation, reverse=reverse)

    def fit(self, graph):
        code = graph.relation_code(self.relation)
        self._code = code if self.checkpoint.knows(code) else None
        self._seeds = np.fromiter(
            (i for i in (graph.lookup(s) for s in (self.to or ())) if i is not None),
            dtype=np.int64)

    def _plausibility(self, seed, candidate):
        """Score the triple, with the seed on whichever end the relation expects."""
        relation = np.full(len(candidate), self._code, dtype=np.int64)
        head, tail = (candidate, seed) if self.reverse else (seed, candidate)
        return self.checkpoint.score(head, relation, tail)

    def edge_weight(self, source, relation, target):
        if self._code is None:
            return 0.0
        return float(self._plausibility(np.array([source]), np.array([target]))[0])

    def score(self, query, rows):
        if not rows or self._code is None:
            return [0.0] * len(rows)
        candidates = np.fromiter((row[query.primary_column] for row in rows),
                                 dtype=np.int64, count=len(rows))

        if len(self._seeds):                     # best plausibility against any seed
            best = None
            for seed in self._seeds:
                scores = self._plausibility(np.full(len(candidates), seed), candidates)
                best = scores if best is None else np.maximum(best, scores)
            return best.tolist()

        path_col = query.path_column
        if path_col is None:
            return [0.0] * len(rows)
        seeds = np.fromiter((row[path_col][-1] for row in rows),
                            dtype=np.int64, count=len(rows))
        return self._plausibility(seeds, candidates).tolist()
