"""Implicit-feedback matrix factorization, and its diffused variant.

Both read the user-item matrix as a block slice of the relation matrix the
Graph already holds, rather than assembling their own CSR from the adjacency."""

import numpy as np

from ..core import Strategy


class MatrixFactorization(Strategy):
    """Implicit-feedback matrix factorization scored per row. The full user-item
    matrix is factorized once in fit() (cached per graph); each row is scored by
    its user's affinity for its item. edge_weight guides a Greedy engine."""

    supports_guidance = True

    def __init__(self, factors=8, iterations=20, regularization=0.05, seed=42,
                 item_type="movie", user_type="user", relation="has_interact"):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.seed = seed
        self.item_type = item_type
        self.user_type = user_type
        self.relation = relation

    def fit(self, graph):
        self._graph = graph
        self._last_fit = self.cached(graph, "factors", lambda: self._compute_factors(graph))
        return self._last_fit

    def _compute_factors(self, graph):
        """The user-item matrix is the interaction relation restricted to the two
        type blocks -- a slice of a matrix the Graph already holds, where this
        used to be a hand-built CSR with its own index arrays."""
        users = graph.block(self.user_type)
        items = graph.block(self.item_type)
        matrix = graph.relation_matrix(self.relation)[users[0]:users[1], items[0]:items[1]]
        matrix = matrix.tocsr()
        matrix.data.fill(1.0)      # CSR sums duplicate entries; feedback stays binary
        matrix.sort_indices()
        user_factors, item_factors = self._factorize(matrix)
        return users, items, user_factors, item_factors

    def edge_weight(self, source, relation, target):
        users, items, user_factors, item_factors = self._last_fit
        # node 0 is a real node, so these are None-checks, not truth tests
        user = _within(source, users)
        user = _within(target, users) if user is None else user
        item = _within(source, items)
        item = _within(target, items) if item is None else item
        if user is None or item is None:
            return 0.0
        return float(user_factors[user - users[0]] @ item_factors[item - items[0]])

    def score(self, query, rows):
        users, items, user_factors, item_factors = self.fit(query.graph)
        user_col = query.node_columns.get(self.user_type)
        item_col = query.node_columns.get(self.item_type, query.primary_column)

        anchor = None
        if user_col is None:
            index = self._anchor_user(query, users)
            anchor = user_factors[index - users[0]] if index is not None else None

        scores = []
        for row in rows:
            item = _within(row[item_col], items)
            if user_col is None:
                vector = anchor
            else:
                user = _within(row[user_col], users)
                vector = user_factors[user - users[0]] if user is not None else None
            scores.append(float(vector @ item_factors[item - items[0]])
                          if vector is not None and item is not None else 0.0)
        return scores

    def _anchor_user(self, query, users):
        # fallback when the query has no user column: the first user node the
        # search walked through
        return next((n for n in query.visited_nodes if users[0] <= n < users[1]), None)

    def _factorize(self, matrix):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = matrix.shape
        user_factors = rng.normal(scale=0.1, size=(n_users, self.factors))
        item_factors = rng.normal(scale=0.1, size=(n_items, self.factors))
        reg = self.regularization * np.eye(self.factors)

        # Who interacted with what never changes across iterations, so read it
        # straight off the sparse layout once: CSR stores row i's column indices
        # contiguously in indices[indptr[i]:indptr[i+1]], and CSC does the same
        # per column. That is the alternative to re-scanning a boolean mask (and
        # a strided column of it) on every pass.
        csr, csc = matrix.tocsr(), matrix.tocsc()
        by_user = [csr.indices[csr.indptr[i]:csr.indptr[i + 1]] for i in range(n_users)]
        by_item = [csc.indices[csc.indptr[j]:csc.indptr[j + 1]] for j in range(n_items)]

        for _ in range(self.iterations):
            for i, observed in enumerate(by_user):
                if observed.size:
                    A = item_factors[observed]
                    # feedback is binary, so the right-hand side A.T @ ones is
                    # just the column sum -- one matmul less per row
                    user_factors[i] = np.linalg.solve(A.T @ A + reg, A.sum(axis=0))
            for j, observed in enumerate(by_item):
                if observed.size:
                    A = user_factors[observed]
                    item_factors[j] = np.linalg.solve(A.T @ A + reg, A.sum(axis=0))

        # The matrix spans the whole type block, so it includes items nobody
        # interacted with. Their solve is skipped, which would leave them holding
        # the random initialization -- an affinity invented out of nothing. No
        # evidence means no affinity, so they are zeroed.
        user_factors[[i for i, o in enumerate(by_user) if not o.size]] = 0.0
        item_factors[[j for j, o in enumerate(by_item) if not o.size]] = 0.0
        return user_factors, item_factors


class DiffusedMatrixFactorization(MatrixFactorization):
    """MatrixFactorization whose latent space bleeds from interaction edges onto
    the rest of the graph: each attribute node gets an embedding = mean of its
    neighbour items' factors, so every edge has a weight. Scores against an
    explicit seed set (to=), a per-row path seed, or the base user model.

    The embedding table is one (n_nodes, factors) array. Nodes being integers is
    what allows that -- and with it, scoring a whole result set is one matrix
    product instead of a Python loop over rows."""

    def __init__(self, *args, to=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to = to

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
        incidence = graph.adjacency()[:, items[0]:items[1]]
        represented = (item_factors != 0).any(axis=1)
        counts = incidence @ represented.astype(np.float64)
        embeddings = np.zeros((graph.n_nodes, self.factors))
        known = counts > 0
        embeddings[known] = (incidence @ item_factors)[known] / counts[known, None]

        # the factorized blocks are authoritative for themselves
        embeddings[items[0]:items[1]] = item_factors
        embeddings[users[0]:users[1]] = user_factors
        known[items[0]:items[1]] = True
        known[users[0]:users[1]] = True
        return embeddings, known

    def edge_weight(self, source, relation, target):
        embeddings = getattr(self, "_embeddings", None)
        if embeddings is None or not (self._known[source] and self._known[target]):
            return 0.0
        return float(embeddings[source] @ embeddings[target])

    def score(self, query, rows):
        if not rows:
            return []
        embeddings = self.embeddings(query.graph)
        recommended = np.fromiter((row[query.primary_column] for row in rows),
                                  dtype=np.int64, count=len(rows))

        if self.to:                             # mode 1: explicit seed set, no path needed
            seeds = self._seed_indices(query.graph)
            if not len(seeds):
                return [0.0] * len(rows)
            return (embeddings[recommended] @ embeddings[seeds].T).max(axis=1).tolist()

        path_col = query.path_column
        if path_col is None:                    # mode 3: fall back to user-based MF
            return super().score(query, rows)
        seeds = np.fromiter((row[path_col][-1] for row in rows),   # mode 2: each row's own seed
                            dtype=np.int64, count=len(rows))
        return np.einsum("ij,ij->i", embeddings[recommended], embeddings[seeds]).tolist()

    def _seed_indices(self, graph):
        return np.fromiter((i for i in (graph.lookup(s) for s in self.to) if i is not None),
                           dtype=np.int64)


def _within(index, block):
    """The index itself when it falls inside a type's block, else None."""
    return index if block[0] <= index < block[1] else None
