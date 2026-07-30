"""Ranking by a trained knowledge-graph embedding.

One strategy, many models: Embedding does not know whether the checkpoint holds
TransD, TransE or TransH. It reads the model name and applies the matching
scoring function, so adding a model means adding an entry here and a trainer in
jerboas.models -- never a new Strategy.

This module is numpy-only on purpose. Scoring a translational embedding is a
projection and a norm; torch is needed to *fit* one, not to use it. That is what
lets a serving deployment install the library without the `torch` extra.

The scoring functions below are deliberately duplicated from their torch twins in
jerboas.models: each is one line, and `test_torch_and_numpy_score_agree` pins them
together. An abstraction to share them across the two array libraries would cost
more than the line it saved.
"""

import numpy as np

from ..checkpoint import load as load_checkpoint
from ..core import Strategy


def _project(entity, entity_vec, relation_vec):
    """TransD's dynamic projection: p_r(e) = r_p (e_p . e) + e."""
    return relation_vec * (entity * entity_vec).sum(-1, keepdims=True) + entity


def _transd(tensors, relation_row, head, tail):
    projected_head = _project(tensors["entity"][head], tensors["entity_vec"][head],
                              tensors["relation_vec"][relation_row])
    projected_tail = _project(tensors["entity"][tail], tensors["entity_vec"][tail],
                              tensors["relation_vec"][relation_row])
    translated = projected_head + tensors["relation"][relation_row]
    return -np.linalg.norm(translated - projected_tail, axis=-1)


def _transe(tensors, relation_row, head, tail):
    translated = tensors["entity"][head] + tensors["relation"][relation_row]
    return -np.linalg.norm(translated - tensors["entity"][tail], axis=-1)


SCORERS = {"transd": _transd, "transe": _transe}


class Embedding(Strategy):
    """Rank candidates by how plausible a trained embedding finds the edge that
    would connect them to a seed.

    The score of a row is `-||proj(h) + r - proj(t)||`: the model's own measure of
    whether that triple belongs in the graph. Two readings, chosen the way
    DiffusedMatrixFactorization chooses its own:

        Embedding.load(path, to=seeds)   the best score against any seed
        Embedding.load(path)             each row scored against its own path seed

    edge_weight is the same quantity on a single edge, which makes this the one
    strategy where guiding a Greedy engine is the model's actual objective rather
    than a proxy for it.
    """

    supports_guidance = True

    def __init__(self, checkpoint, to=None, relation=None):
        self.checkpoint = checkpoint
        self.to = to
        # which relation the score is measured along; the interaction relation is
        # the natural default for a recommender
        self.relation = relation or "has_interact"
        self._score_fn = SCORERS[checkpoint.model]
        self._seeds = None
        self._relation_row = None

    @classmethod
    def load(cls, path, graph, to=None, relation=None):
        return cls(load_checkpoint(path, graph), to=to, relation=relation)

    def fit(self, graph):
        code = graph.relation_code(self.relation)
        self._relation_row = None if code is None else self.checkpoint.relation_row(code)
        self._seeds = np.fromiter(
            (i for i in (graph.lookup(s) for s in (self.to or ())) if i is not None),
            dtype=np.int64)

    def _triples(self, head, tail):
        return self._score_fn(self.checkpoint.tensors, self._relation_row, head, tail)

    def edge_weight(self, source, relation, target):
        if self._relation_row is None:
            return 0.0
        return float(self._triples(np.array([source]), np.array([target]))[0])

    def score(self, query, rows):
        if not rows or self._relation_row is None:
            return [0.0] * len(rows)
        candidates = np.fromiter((row[query.primary_column] for row in rows),
                                 dtype=np.int64, count=len(rows))

        if len(self._seeds):                    # best plausibility against any seed
            best = None
            for seed in self._seeds:
                scores = self._triples(np.full(len(candidates), seed), candidates)
                best = scores if best is None else np.maximum(best, scores)
            return best.tolist()

        path_col = query.path_column
        if path_col is None:
            return [0.0] * len(rows)
        seeds = np.fromiter((row[path_col][-1] for row in rows),
                            dtype=np.int64, count=len(rows))
        return self._triples(seeds, candidates).tolist()
