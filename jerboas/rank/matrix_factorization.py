"""Implicit-feedback matrix factorization, and its diffused variant.

Both read the user-item matrix as a block slice of the relation matrix the
Graph already holds -- or, with `where`, from a frame of the interactions they
are allowed to learn from."""

import copy

import numpy as np
import scipy.sparse as sp

from ..store.checkpoint import (IDENTITY, NODE, load as load_checkpoint, provenance,
                                save as save_checkpoint)
from .core import Strategy

# what a checkpoint holds: one row of factors per node of the two blocks
TABLES = (("factors", NODE),)

# how many dimensions of a diffused sum exist at once
_DIMENSIONS = 8


class MatrixFactorization(Strategy):
    """Implicit-feedback matrix factorization, scored per row.

    The full user-item matrix is factorized once in fit() (cached per graph);
    each row is scored by its user's affinity for its item.

        .with_columns(mf=MatrixFactorization().on("rec", "user"))
        .with_columns(mf=MatrixFactorization(user=who).on("rec"))

    Whose taste is being applied is *named*, either as a second column holding
    the user of each row or as one user for the whole frame. It is not inferred:
    a guessed user is an arbitrary person's ranking wearing the shape of an
    answer.

    `weighted=True` reads the interaction's stored score instead of its mere
    existence, which turns the same solver into explicit feedback: the target is
    the rating rather than a 1. Implicit stays the default because "she watched
    it" and "she rated it 2" are different claims, and only the caller knows
    which one their edges carry.

    Fitting takes minutes on a large graph, so a fitted factorization is stored
    the way an embedding is -- rebound by name on load:

        model.save("checkpoints/spotify.dmf.npz", graph, alias="uri")
        DiffusedMatrixFactorization.load("checkpoints/spotify.dmf.npz", graph)
    """

    name = "mf"          # the key a checkpoint records

    def __init__(self, factors=8, iterations=20, regularization=0.05, seed=42,
                 item_type="movie", user_type="user", relation="has_interact",
                 weighted=False, user=None, where=None):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.seed = seed
        self.item_type = item_type
        self.user_type = user_type
        self.relation = relation
        self.weighted = weighted
        self.user = user                 # one user for the whole frame, if named
        self.where = where               # the interactions it may learn from

    def config(self):
        """The hyperparameters, as a checkpoint records them: what has to match
        for a stored factorization to be this one."""
        return {"factors": self.factors, "iterations": self.iterations,
                "regularization": self.regularization, "seed": self.seed,
                "item_type": self.item_type, "user_type": self.user_type,
                "relation": self.relation, "weighted": self.weighted}

    def save(self, path, graph, alias=IDENTITY, **details):
        """Write the fitted factors, rebindable to any graph by name.

        Only the two blocks that have factors are stored. `details` go into the
        provenance beside the hyperparameters -- whatever narrowed `where` is
        the caller's to record, since a frame of edges has no name."""
        users, items, user_factors, item_factors = self.fit(graph)
        nodes = np.concatenate([np.arange(*users), np.arange(*items)])
        table = np.concatenate([user_factors, item_factors]).astype(np.float32)
        meta = provenance(graph, model=self.name, **self.config(), **details)
        return save_checkpoint(path, self.name, TABLES, self.factors, graph,
                               {"factors": table}, meta, alias=alias, nodes=nodes)

    @classmethod
    def load(cls, path, graph, alias=None, **kwargs):
        """A stored factorization, rebound to `graph` and ready to score.

        The hyperparameters come off the file; `kwargs` are the per-query ones
        (`to=`, `user=`). What the file recorded is `model.meta`, so a caller can
        check it is the factorization it meant."""
        stored = load_checkpoint(path, graph, cls.name, TABLES, alias=alias)
        meta = stored.meta
        model = cls(factors=stored.factors, iterations=meta["iterations"],
                    regularization=meta["regularization"], seed=meta["seed"],
                    item_type=meta["item_type"], user_type=meta["user_type"],
                    relation=meta["relation"], weighted=meta["weighted"], **kwargs)
        model.meta = meta
        model.missing_nodes = stored.missing_nodes
        users, items = graph.block(model.user_type), graph.block(model.item_type)
        table = stored.tensors["factors"]
        # copies of the two blocks, so the (N, factors) table can go
        fitted = (users, items, table[users[0]:users[1]].copy(),
                  table[items[0]:items[1]].copy())
        del table, stored
        model.cached(graph, ("factors", None), lambda: fitted)
        return model

    def fit(self, graph):
        self._graph = graph
        key = ("factors", None if self.where is None else id(self.where))
        self._last_fit = self.cached(graph, key, lambda: self._compute_factors(graph))
        return self._last_fit

    def _compute_factors(self, graph):
        """The user-item matrix is the interaction relation restricted to the two
        type blocks -- a slice of a matrix the Graph already holds.

        `where` narrows it to a frame of interactions, the same way `train`
        narrows what a model may learn from. It is not a nicety: a factorization
        of everything is a factorization of mostly nothing when the long tail is
        long enough, and which part of the tail to keep is a claim about the
        data that only the caller can make."""
        users = graph.block(self.user_type)
        items = graph.block(self.item_type)
        weights = "raw" if self.weighted else None
        if self.where is not None:
            matrix = self._from_frame(graph, users, items)
        else:
            matrix = graph.relation_matrix(self.relation, weights)[users[0]:users[1],
                                                                   items[0]:items[1]]
        matrix = matrix.tocsr()
        # the graph keeps a multi-edge as repeated cells; one cell per pair, so
        # that binary feedback is binary and a rating is not counted twice
        matrix.sum_duplicates()
        if not self.weighted:
            matrix.data.fill(1.0)
        matrix.sort_indices()
        user_factors, item_factors = self._factorize(matrix)
        # solved in float64, kept in float32: a score needs seven digits, and on
        # a large graph the factors are gigabytes
        return (users, items, user_factors.astype(np.float32),
                item_factors.astype(np.float32))

    def _from_frame(self, graph, users, items):
        """The interactions a frame of edges holds, as the same sparse matrix.

        Either endpoint may be on either side -- `g.edges("contains")` runs
        playlist to song and its reverse runs the other way -- so the pair is
        sorted into (user, item) rather than assumed."""
        frame = self.where.pl if hasattr(self.where, "pl") else self.where
        source = frame["source"].to_numpy()
        target = frame["target"].to_numpy()
        forwards = (source >= users[0]) & (source < users[1])
        user_ids = np.where(forwards, source, target) - users[0]
        item_ids = np.where(forwards, target, source) - items[0]
        keep = ((user_ids >= 0) & (user_ids < users[1] - users[0])
                & (item_ids >= 0) & (item_ids < items[1] - items[0]))
        data = (frame["score"].to_numpy() if self.weighted and "score" in frame.columns
                else np.ones(len(source)))
        return sp.csr_matrix((data[keep], (user_ids[keep], item_ids[keep])),
                             shape=(users[1] - users[0], items[1] - items[0]))

    def scores(self, graph, columns):
        """on("item") with `user=` given, or on("item", "user") per row."""
        users, items, user_factors, item_factors = self.fit(graph)
        item = np.asarray(columns[0], dtype=np.int64)
        rows = _local(item, items)

        if len(columns) > 1:
            people = _local(np.asarray(columns[1], dtype=np.int64), users)
        elif self.user is not None:
            named = graph.ids_of([self.user] if _one(self.user) else self.user)
            people = np.full(len(item), _local(named[:1], users)[0] if len(named) else -1)
        else:
            raise ValueError(
                f"{type(self).__name__} needs to know whose taste to apply: name "
                f"the user column, on(\"rec\", \"user\"), or one user, "
                f"{type(self).__name__}(user=key).")

        known = (rows >= 0) & (people >= 0)
        out = np.zeros(len(item))
        if known.any():
            out[known] = np.einsum("ij,ij->i", user_factors[people[known]],
                                   item_factors[rows[known]])
        return out

    def _factorize(self, matrix):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = matrix.shape
        user_factors = rng.normal(scale=0.1, size=(n_users, self.factors))
        item_factors = rng.normal(scale=0.1, size=(n_items, self.factors))
        reg = self.regularization * np.eye(self.factors)

        # Who interacted with what never changes across iterations, so read it
        # straight off the sparse layout once: CSR stores row i's column indices
        # contiguously in indices[indptr[i]:indptr[i+1]], and CSC does the same
        # per column. The values travel with them, so implicit and explicit
        # feedback run the same loop.
        csr, csc = matrix.tocsr(), matrix.tocsc()
        by_user = [_slice(csr, i) for i in range(n_users)]
        by_item = [_slice(csc, j) for j in range(n_items)]

        for _ in range(self.iterations):
            for i, (observed, target) in enumerate(by_user):
                if observed.size:
                    A = item_factors[observed]
                    # the right-hand side is A.T @ target; with binary feedback
                    # every target is 1 and it degenerates to the column sum
                    user_factors[i] = np.linalg.solve(A.T @ A + reg, target @ A)
            for j, (observed, target) in enumerate(by_item):
                if observed.size:
                    A = user_factors[observed]
                    item_factors[j] = np.linalg.solve(A.T @ A + reg, target @ A)

        # The matrix spans the whole type block, so it includes items nobody
        # interacted with. Their solve is skipped, which would leave them holding
        # the random initialization -- an affinity invented out of nothing. No
        # evidence means no affinity, so they are zeroed.
        user_factors[[i for i, (o, _) in enumerate(by_user) if not o.size]] = 0.0
        item_factors[[j for j, (o, _) in enumerate(by_item) if not o.size]] = 0.0
        return user_factors, item_factors


class DiffusedMatrixFactorization(MatrixFactorization):
    """MatrixFactorization whose latent space bleeds from interaction edges onto
    the rest of the graph: each attribute node gets an embedding = mean of its
    neighbour items' factors, so every edge has a weight. Scores against an
    explicit seed set (to=), a per-row seed column, or the base user model.

    The embedding table is one (n_nodes, factors) array. Nodes being integers is
    what allows that -- and with it, scoring a whole result set is one matrix
    product instead of a Python loop over rows."""

    name = "dmf"

    def __init__(self, *args, to=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to = to

    def seeded(self, to):
        """The same fitted factors, aimed at a different seed set.

        Fitting costs seconds and choosing what to compare against costs
        nothing, so a service fits once at startup and calls this per request.
        The cache is shared rather than copied, which is what makes the second
        call free -- and shared read-only, so concurrent requests do not tread
        on each other."""
        clone = copy.copy(self)
        clone.to = to
        clone.__dict__["_strategy_cache"] = self.__dict__.setdefault(
            "_strategy_cache", {})
        return clone

    def embeddings(self, graph):
        # both halves come out of one cache entry: recomputing on a cache hit is
        # what would otherwise leave `known` stale from an earlier graph
        self._embeddings, self._known = self.cached(
            graph, "embeddings", lambda: self._compute_embeddings(graph))
        return self._embeddings

    def _compute_embeddings(self, graph):
        users, items, user_factors, item_factors = self.fit(graph)

        # "mean of the neighbouring items' factors" is a grouped sum over a group
        # size -- i.e. one sparse matmul against the item block of the adjacency.
        # The group counts only items that *have* factors: an item nobody
        # interacted with has no representation, so letting it into the
        # denominator would shrink its neighbours' embeddings toward zero for no
        # reason. That count is itself a matrix-vector product.
        #
        # Weighted, the mean becomes a weighted mean -- normalized, because a
        # negative weight would pull an embedding to the far side of the space
        # rather than count for less.
        #
        # Neighbours in either direction are two products, one per direction of
        # the store, over a vector that is zero outside the item block -- rather
        # than a symmetric adjacency sliced to the items, which would be held
        # whole to be multiplied once. The sum is taken a few dimensions at a
        # time for the same reason.
        directions = graph.matrices("norm" if self.weighted else None)
        represented = np.zeros(graph.n_nodes, dtype=np.float32)
        represented[items[0]:items[1]] = (item_factors != 0).any(axis=1)
        counts = sum(direction @ represented for direction in directions)
        known = counts > 0
        embeddings = np.zeros((graph.n_nodes, self.factors), dtype=np.float32)
        spread = np.zeros((graph.n_nodes, _DIMENSIONS), dtype=np.float32)
        for low in range(0, self.factors, _DIMENSIONS):
            high = min(low + _DIMENSIONS, self.factors)
            spread[:, :high - low] = 0.0
            spread[items[0]:items[1], :high - low] = item_factors[:, low:high]
            summed = sum(direction @ spread[:, :high - low] for direction in directions)
            embeddings[known, low:high] = summed[known] / counts[known, None]

        # the factorized blocks are authoritative for themselves
        embeddings[items[0]:items[1]] = item_factors
        embeddings[users[0]:users[1]] = user_factors
        known[items[0]:items[1]] = True
        known[users[0]:users[1]] = True
        return embeddings, known

    def scores(self, graph, columns):
        """on("rec") against the `to=` seeds, or on("rec", "seed") per row."""
        embeddings = self.embeddings(graph)
        recommended = np.asarray(columns[0], dtype=np.int64)
        if not len(recommended):
            return np.zeros(0)

        if len(columns) > 1:                    # each row against its own seed
            seeds = np.asarray(columns[1], dtype=np.int64)
            return np.einsum("ij,ij->i", embeddings[recommended], embeddings[seeds])
        if self.to:                             # the best against one seed set
            seeds = graph.ids_of(self.to)
            if not len(seeds):
                return np.zeros(len(recommended))
            return (embeddings[recommended] @ embeddings[seeds].T).max(axis=1)
        return super().scores(graph, columns)   # fall back to the user model

def _local(nodes, block):
    """Node ids as positions inside a type block, and -1 for the ones outside it
    -- a node the factorization has no row for scores zero rather than someone
    else's affinity."""
    low, high = block
    return np.where((nodes >= low) & (nodes < high), nodes - low, -1)


def _one(value):
    """Is this one node, or a collection of them?"""
    return isinstance(value, (str, tuple)) or hasattr(value, "__index__")


def _slice(matrix, i):
    """One row (CSR) or column (CSC): its observed indices and their values."""
    start, stop = matrix.indptr[i], matrix.indptr[i + 1]
    return matrix.indices[start:stop], matrix.data[start:stop]
