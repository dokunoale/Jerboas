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


class Translational(Strategy, nn.Module):
    """A model of the form -||f(head) + relation - f(tail)||.

    As a Strategy it ranks candidates by how plausible the edge joining them to a
    seed would be. Two readings, chosen the way DiffusedMatrixFactorization
    chooses its own:

        TransD.load(path, g, to=seeds)   the best plausibility against any seed
        TransD.load(path, g)             each row against its own path seed

    `relation` picks the edge being judged. Left as None -- the default -- the
    score is the best over *every* relation in both directions, which is link
    prediction without naming the relation, and the only thing that works when
    the seeds are of mixed types: a person is joined to a film by `directed_by`
    read backwards, a genre by `has_genre` backwards, a user by `has_interact`
    forwards. Naming one relation (with `reverse=` for its direction) asks the
    narrower question.

    edge_weight is the same quantity on a single edge, which makes this family the
    one place where guiding a Greedy engine is the model's own objective rather
    than a proxy for it.
    """

    supports_guidance = True

    name = None          # the key a checkpoint records
    tables = ()          # (name, space) pairs

    def __init__(self, factors=64, margin=1.0, seed=42,
                 to=None, relation=None, reverse=False):
        nn.Module.__init__(self)
        self.factors = factors
        self.margin = margin
        self.seed = seed
        # not `self.to`: nn.Module.to() is device movement, and shadowing it
        # would break train()
        self._to = to
        self.relation = relation        # None = any relation, either direction
        self.reverse = reverse
        self.weights = nn.ModuleDict()      # populated by build()
        self.arrays = None                  # populated by load()
        self.meta = {}
        self.missing_nodes = ()
        self.missing_relations = ()
        self._graph = None
        self._seeds = None
        self._edges = ()                # (relation code, reverse) pairs to score over

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
    def load(cls, path, graph, to=None, relation=None, reverse=False):
        """Read a checkpoint back as a strategy ready for rank(...)."""
        stored = load_checkpoint(path, graph, cls.name, cls.tables)
        model = cls(factors=stored.factors, to=to, relation=relation, reverse=reverse)
        model._graph = graph
        model.arrays = stored.tensors
        model.meta = stored.meta
        model.missing_nodes = stored.missing_nodes
        model.missing_relations = stored.missing_relations
        return model

    def seeded(self, to, relation=None, reverse=False):
        """The same trained weights, aimed at a different seed set.

        Loading rebinds every row against the graph, which is linear in its size;
        changing who you are asking about is not. A service loads once at startup
        and calls this per request -- and because the weights are shared rather
        than copied, concurrent requests do not tread on each other."""
        clone = type(self)(factors=self.factors, to=to, relation=relation, reverse=reverse)
        clone._graph = self._graph
        clone.arrays = self.arrays               # shared, read-only
        clone.meta = self.meta
        clone.missing_nodes = self.missing_nodes
        clone.missing_relations = self.missing_relations
        return clone

    # --- the Strategy contract ------------------------------------------------

    def fit(self, graph):
        known = [code for code in range(len(graph.relations))
                 if code not in self.missing_relations]
        if self.relation is None:               # any relation, either direction
            self._edges = tuple((code, reverse) for code in known for reverse in (False, True))
        else:
            code = graph.relation_code(self.relation)
            self._edges = () if code is None or code not in known else ((code, self.reverse),)
        self._seeds = np.fromiter(
            (i for i in (graph.lookup(s) for s in (self._to or ())) if i is not None),
            dtype=np.int64)

    def _one(self, code, reverse, seed, candidate):
        relation = np.full(len(candidate), code, dtype=np.int64)
        head, tail = (candidate, seed) if reverse else (seed, candidate)
        return self.plausibility(head, relation, tail)

    def _against(self, seed, candidate):
        """The most plausible of the edges under consideration -- one when a
        relation was named, every relation both ways when it was not."""
        best = None
        for code, reverse in self._edges:
            scored = self._one(code, reverse, seed, candidate)
            best = scored if best is None else np.maximum(best, scored)
        return best

    def edge_weight(self, source, relation, target):
        if not self._edges:
            return 0.0
        return float(self._against(np.array([source]), np.array([target]))[0])

    def score(self, query, rows):
        if not rows or not self._edges:
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
