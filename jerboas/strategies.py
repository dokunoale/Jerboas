"""Ranking strategies: the soft, non-deterministic-first scoring layer.

Every ranker is a Strategy (core.py): fit() once, score() per run, optional
edge_weight for guided engines.

Each of these used to open the adjacency and assemble its own sparse matrix --
one built the user-item matrix, one the transition matrix, one an incidence
matrix, all by scanning a dict of dicts of lists. The Graph now stores the
adjacency as CSR, so those three constructions are block slices of a shared,
memoized matrix and the strategies are left with only their own arithmetic.

Nodes are integers, which is what makes an embedding table a single (N, factors)
array indexed directly, instead of a dict keyed by node name.

Score is the projection marker that pulls the row's final combined score out in
select(...). ExprStrategy adapts a bare scalar Expr (a Degree) into a Strategy so
rank(expr) works with no wrapping.
"""

import numpy as np

from .core import Strategy


class Score:
    """A projection marker: select(rec, Score()) yields the row's final ranking
    score. Not a Strategy -- it reads back what rank(...) computed."""


class ExprStrategy(Strategy):
    """Adapts a scalar Expr (currently a Degree) into a Strategy: scores each row
    by evaluating the expression against its binding."""

    def __init__(self, expr):
        self.expr = expr        # a Degree: has .node, .relation, .reverse

    def score(self, query, rows):
        col = query._columns.get(id(self.expr.node), query.primary_column)
        degrees = query.graph.degree(self.expr.relation, self.expr.reverse)
        return [float(degrees[row[col]]) for row in rows]


class Alphabetical(Strategy):
    """Rank rows by the label of their primary node, A to Z."""

    def score(self, query, rows):
        graph, col = query.graph, query.primary_column
        labels = [str(graph.label_of(row[col])) for row in rows]
        order = sorted(range(len(rows)), key=lambda i: labels[i])
        scores = [0.0] * len(rows)
        for position, i in enumerate(order):
            scores[i] = float(-position)
        return scores


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


class PageRank(Strategy):
    """Random-walk importance over the graph, computed once by power iteration.

    Global by default; personalized (random walk with restart) when `to=` seeds
    are given -- then the walk teleports back to the seed set instead of the
    whole graph, so a node scores by its proximity to the things you like. That
    makes PageRank(to=seeds) a principled sibling of Connectivity: same intent
    (relevance to the seeds), but as the stationary distribution of a restarting
    walk rather than a two-hop count.

    The walk runs on the graph's undirected adjacency, so mass flows both ways
    along every stored edge. supports_guidance is on: edge_weight returns the
    target's rank, so a Greedy engine prefers important nodes.
    """

    supports_guidance = True

    def __init__(self, to=None, damping=0.85, iterations=100, tol=1e-6):
        self.to = to                    # optional seed set -> personalized (RWR)
        self.damping = damping
        self.iterations = iterations
        self.tol = tol

    def fit(self, graph):
        seeds = tuple(sorted(i for i in (graph.lookup(s) for s in (self.to or ()))
                             if i is not None))
        self._ranks = self.cached(graph, ("pagerank", seeds, self.damping, self.iterations),
                                  lambda: self._power_iteration(graph, seeds))
        return self._ranks

    def edge_weight(self, source, relation, target):
        return float(self._ranks[target])

    def score(self, query, rows):
        ranks, col = self._ranks, query.primary_column
        return [float(ranks[row[col]]) for row in rows]

    def _power_iteration(self, graph, seeds):
        """Power iteration as repeated sparse matrix-vector products.

        One iteration pushes every node's rank along every out-edge, which written
        by hand is a Python loop over all E edges -- repeated `iterations` times.
        That is exactly what a sparse matrix-vector product does, so the whole
        inner loop lives in compiled code and only `iterations` numpy calls
        remain at Python level."""
        n = graph.n_nodes
        if n == 0:
            return np.zeros(0)

        adjacency = graph.adjacency()
        outdeg = np.asarray(adjacency.sum(axis=1)).ravel()
        dangling = np.flatnonzero(outdeg == 0)

        # teleport distribution: uniform, or concentrated on the seeds (RWR)
        teleport = np.zeros(n)
        if seeds:
            teleport[list(seeds)] = 1.0 / len(seeds)
        else:
            teleport[:] = 1.0 / n

        # transition[j, i] = the share of i's rank that flows to j. Scaling the
        # transpose column-wise by 1/outdeg does that in one sparse operation.
        share = np.zeros(n)
        np.divide(1.0, outdeg, out=share, where=outdeg > 0)
        transition = adjacency.T.multiply(share).tocsr()

        d = self.damping
        rank = teleport.copy()
        for _ in range(self.iterations):
            # rank stranded on dangling nodes has nowhere to flow, so it is
            # redistributed along the teleport distribution
            leaked = d * rank[dangling].sum()
            new_rank = (1.0 - d + leaked) * teleport + d * (transition @ rank)
            delta = np.abs(new_rank - rank).sum()
            rank = new_rank
            if delta < self.tol:
                break

        return rank


def _within(index, block):
    """The index itself when it falls inside a type's block, else None."""
    return index if block[0] <= index < block[1] else None
