"""Knowledge-graph embeddings: models that are also ranking strategies.

A model here is one class holding everything about it -- the tables it
allocates, the arithmetic that scores a triple, how it is fitted, and how it
ranks. There is no separate serving object and no registry pairing a model with
its maths, because there is nothing to pair: the class is both.

That follows the library's own rule rather than working around it. `Strategy` is
the extension point for ranking (core.py); a trained embedding is a ranker, so it
subclasses `Strategy` exactly as `PageRank` does, and `rank(TransD.load(...))`
needs no wrapper.

Two words, deliberately kept apart:

    score(query, rows)              the Strategy contract -- one float per result row
    plausibility(head, rel, tail)   this family's own quantity -- is this triple real?

`score` stays the single scoring interface everything passed to `rank(...)`
implements. `plausibility` is what a subclass defines, in three or four lines.
"""

import numpy as np
import torch
from torch import nn

from ..checkpoint import load as load_checkpoint, provenance, save as save_checkpoint
from ..core import Strategy

NODE = "node"            # a table with one row per node
RELATION = "relation"    # a table with one row per relation

INTERACT = "has_interact"


class Translational(Strategy, nn.Module):
    """A model of the form -||f(head) + relation - f(tail)||.

    As a Strategy it ranks candidates by how plausible the edge joining them to a
    seed would be. Two readings, chosen the way DiffusedMatrixFactorization
    chooses its own:

        TransD.load(path, g, to=seeds)   the best plausibility against any seed
        TransD.load(path, g)             each row against its own path seed

    `reverse=True` puts the seed on the tail instead of the head, which is what a
    relation stored the other way round needs: `directed_by` runs movie ->
    person, so ranking movies for a person seed reads it backwards.

    edge_weight is the same quantity on a single edge, which makes this family the
    one place where guiding a Greedy engine is the model's own objective rather
    than a proxy for it.
    """

    supports_guidance = True

    name = None          # the key a checkpoint records
    tables = ()          # (name, space) pairs

    def __init__(self, factors=64, margin=1.0, seed=42,
                 to=None, relation=INTERACT, reverse=False):
        nn.Module.__init__(self)
        self.factors = factors
        self.margin = margin
        self.seed = seed
        # not `self.to`: nn.Module.to() is device movement, and shadowing it
        # would break train()
        self._to = to
        self.relation = relation
        self.reverse = reverse
        self.weights = nn.ModuleDict()      # populated by build()
        self.arrays = None                  # populated by load()
        self.meta = {}
        self.missing_nodes = ()
        self.missing_relations = ()
        self._graph = None
        self._seeds = None
        self._code = None

    # --- what a subclass provides --------------------------------------------

    def plausibility(self, head, relation, tail):
        """How real this triple looks, higher is better.

        Written with operators numpy and torch spell identically -- `*`, `+`,
        `-`, `.sum(-1)`, `[..., None]`, `**` -- so the same lines serve the
        gradient step and the query. `get` hides the one genuine difference: an
        nn.Embedding call while fitting, a fancy index once loaded."""
        raise NotImplementedError

    def get(self, table, index):
        """One table's rows, from whichever form the weights are in."""
        if self.arrays is not None:
            return self.arrays[table][index]
        return self.weights[table](index)

    @staticmethod
    def norm(difference):
        """-||d||, using only what both array libraries spell the same way. A
        library norm rescales to avoid overflow; embeddings are O(1), so squaring
        is safe here and keeps one implementation instead of two."""
        return -((difference * difference).sum(-1) ** 0.5)

    # --- fitting --------------------------------------------------------------

    @property
    def built(self):
        return len(self.weights) > 0

    def build(self, graph):
        """Size the tables to a graph. Separate from __init__ so a model can be
        described before the data it will be fitted to is loaded."""
        torch.manual_seed(self.seed)
        self._graph = graph
        for table, space in self.tables:
            rows = len(graph.relations) if space == RELATION else graph.n_nodes
            embedding = nn.Embedding(rows, self.factors)
            nn.init.xavier_normal_(embedding.weight)
            self.weights[table] = embedding
        return self

    def loss(self, head, relation, tail, corrupt_head, corrupt_tail):
        """Margin ranking: a real triple must outscore its corruption by `margin`.

        Corrupting the head and the tail are separate terms rather than one
        averaged example, so a relation is pushed to learn both of its
        directions -- which matters because the graph stores each edge once and
        every relation is read in both."""
        positive = self.plausibility(head, relation, tail)
        return (torch.relu(self.margin - positive
                           + self.plausibility(head, relation, corrupt_tail)).mean()
                + torch.relu(self.margin - positive
                             + self.plausibility(corrupt_head, relation, tail)).mean())

    # --- storage --------------------------------------------------------------

    def save(self, path, **details):
        """Write a checkpoint that can be rebound to any graph, by name."""
        if not self.built:
            raise ValueError("nothing to save: the model has not been built or fitted")
        arrays = {table: weights.weight.detach().cpu().numpy()
                  for table, weights in self.weights.items()}
        meta = provenance(self._graph, model=self.name, factors=self.factors,
                          margin=self.margin, seed=self.seed, **self.meta, **details)
        return save_checkpoint(path, self.name, self.tables, self.factors,
                               self._graph, arrays, meta)

    @classmethod
    def load(cls, path, graph, to=None, relation=INTERACT, reverse=False):
        """Read a checkpoint back as a strategy ready for rank(...)."""
        stored = load_checkpoint(path, graph, cls.name, cls.tables)
        model = cls(factors=stored.factors, to=to, relation=relation, reverse=reverse)
        model._graph = graph
        model.arrays = stored.tensors
        model.meta = stored.meta
        model.missing_nodes = stored.missing_nodes
        model.missing_relations = stored.missing_relations
        return model

    # --- the Strategy contract ------------------------------------------------

    def fit(self, graph):
        code = graph.relation_code(self.relation)
        self._code = None if code is None or code in self.missing_relations else code
        self._seeds = np.fromiter(
            (i for i in (graph.lookup(s) for s in (self._to or ())) if i is not None),
            dtype=np.int64)

    def _against(self, seed, candidate):
        relation = np.full(len(candidate), self._code, dtype=np.int64)
        head, tail = (candidate, seed) if self.reverse else (seed, candidate)
        return self.plausibility(head, relation, tail)

    def edge_weight(self, source, relation, target):
        if self._code is None:
            return 0.0
        return float(self._against(np.array([source]), np.array([target]))[0])

    def score(self, query, rows):
        if not rows or self._code is None:
            return [0.0] * len(rows)
        candidates = np.fromiter((row[query.primary_column] for row in rows),
                                 dtype=np.int64, count=len(rows))

        if len(self._seeds):                     # best plausibility against any seed
            best = None
            for seed in self._seeds:
                against = self._against(np.full(len(candidates), seed), candidates)
                best = against if best is None else np.maximum(best, against)
            return best.tolist()

        path_col = query.path_column
        if path_col is None:
            return [0.0] * len(rows)
        seeds = np.fromiter((row[path_col][-1] for row in rows),
                            dtype=np.int64, count=len(rows))
        return self._against(seeds, candidates).tolist()
