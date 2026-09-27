"""Knowledge-graph embeddings: models that are also ranking strategies.

A model here is one class holding everything about it -- the tables it
allocates, the arithmetic that scores a triple, how it is fitted, and how it
ranks. There is no separate serving object and no registry pairing a model with
its maths, because there is nothing to pair: the class is both.

That follows the library's own rule rather than working around it. `Strategy` is
the extension point for ranking (core.py); a trained embedding is a ranker, so it
subclasses `Strategy` exactly as `PageRank` does, and `TransD.load(...).on(...)`
needs no wrapper.

Two words, deliberately kept apart:

    scores(graph, columns)          the Strategy contract -- one float per row
    plausibility(head, rel, tail)   this family's own quantity -- is this triple real?

`scores` stays the single scoring interface every strategy implements.
`plausibility` is what a subclass defines, in three or four lines.
"""

import numpy as np
import torch
from torch import nn

from ..store.checkpoint import (IDENTITY, load as load_checkpoint, provenance,
                          save as save_checkpoint)
from ..rank.core import Strategy

NODE = "node"            # a table with one row per node
RELATION = "relation"    # a table with one row per relation


class Translational(Strategy, nn.Module):
    """A model of the form -||f(head) + relation - f(tail)||.

    As a Strategy it ranks candidates by how plausible the edge joining them to a
    seed would be. Two readings, chosen the way DiffusedMatrixFactorization
    chooses its own:

        TransD.load(path, g, to=seeds).on("rec")     best against any seed
        TransD.load(path, g).on("rec", "seed")       each row against its own

    `relation` picks the edge being judged. Left as None -- the default -- the
    score is the best over *every* relation in both directions, which is link
    prediction without naming the relation, and the only thing that works when
    the seeds are of mixed types: a person is joined to a film by `directed_by`
    read backwards, a genre by `has_genre` backwards, a user by `has_interact`
    forwards. Naming one relation (with `reverse=` for its direction) asks the
    narrower question.
    """

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

    def loss(self, head, relation, tail, corrupt_head, corrupt_tail, weight=None):
        """Margin ranking: a real triple must outscore its corruption by `margin`.

        Corrupting the head and the tail are separate terms rather than one
        averaged example, so a relation is pushed to learn both of its
        directions -- which matters because the graph stores each edge once and
        every relation is read in both.

        `weight` scales each example by how much the edge is worth (its
        normalized score), so a 5-star rating pushes harder than a 2-star one
        and an edge weighing nothing contributes no gradient: rather than
        deciding once, for every query, that a poor edge is not an edge, the
        model is told how much to believe it."""
        positive = self.plausibility(head, relation, tail)
        corrupted_tail = torch.relu(self.margin - positive
                                    + self.plausibility(head, relation, corrupt_tail))
        corrupted_head = torch.relu(self.margin - positive
                                    + self.plausibility(corrupt_head, relation, tail))
        if weight is None:
            return corrupted_tail.mean() + corrupted_head.mean()
        return (weight * corrupted_tail).mean() + (weight * corrupted_head).mean()

    # --- storage --------------------------------------------------------------

    def save(self, path, alias=IDENTITY, **details):
        """Write a checkpoint that can be rebound to any graph, by name.

        `alias` is the attribute that identifies a node durably, for a graph
        whose ids are its own numbering rather than the world's -- `alias="uri"`
        on Spotify. It defaults to the id and is stored in the file."""
        if not self.built:
            raise ValueError("nothing to save: the model has not been built or fitted")
        arrays = {table: weights.weight.detach().cpu().numpy()
                  for table, weights in self.weights.items()}
        meta = provenance(self._graph, model=self.name, factors=self.factors,
                          margin=self.margin, seed=self.seed, **self.meta, **details)
        return save_checkpoint(path, self.name, self.tables, self.factors,
                               self._graph, arrays, meta, alias=alias)

    @classmethod
    def load(cls, path, graph, to=None, relation=None, reverse=False, alias=None):
        """Read a checkpoint back as a strategy ready to score.

        The alias comes off the file; pass one only to override it."""
        stored = load_checkpoint(path, graph, cls.name, cls.tables, alias=alias)
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
        clone.weights = self.weights             # so a just-fitted model aims too
        clone.meta = self.meta
        clone.missing_nodes = self.missing_nodes
        clone.missing_relations = self.missing_relations
        return clone

    # --- the Strategy contract ------------------------------------------------

    def fit(self, graph):
        if self.arrays is None and self.built:
            # a model that has just been fitted still holds its weights as torch
            # tables. Scoring reads them with an integer array, so they are read
            # out once here -- the same arrays a checkpoint would have stored,
            # which is what makes "train it, then rank with it" one step
            self.arrays = {table: weights.weight.detach().cpu().numpy()
                           for table, weights in self.weights.items()}
        known = [code for code in range(len(graph.relations))
                 if code not in self.missing_relations]
        if self.relation is None:               # any relation, either direction
            self._edges = tuple((code, reverse) for code in known for reverse in (False, True))
        else:
            code = graph.relation_code(self.relation)
            self._edges = () if code is None or code not in known else ((code, self.reverse),)
        self._seeds = graph.ids_of(self._to or ()).astype(np.int64)

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

    def scores(self, graph, columns):
        """on("rec") against the `to=` seeds, or on("rec", "seed") per row."""
        candidates = np.asarray(columns[0], dtype=np.int64)
        if not len(candidates) or not self._edges:
            return np.zeros(len(candidates))

        if len(columns) > 1:                     # each row against its own seed
            return self._against(np.asarray(columns[1], dtype=np.int64), candidates)
        if not len(self._seeds):
            return np.zeros(len(candidates))
        best = None                              # the best plausibility of any seed
        for seed in self._seeds:
            against = self._against(np.full(len(candidates), seed), candidates)
            best = against if best is None else np.maximum(best, against)
        return best
