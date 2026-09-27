"""One hop, as a gather over the CSR store.

Expanding a set of nodes is not a join. The CSR already indexes exactly what a
join would have to build: node i's edges live in `indices[indptr[i]:indptr[i+1]]`,
contiguously and sorted by relation. So a hop is a concatenation of slices --
which numpy does without a Python loop, given the slice bounds as arrays.

    starts, counts   ->  rows (which source row each result came from)
                         targets, codes, weights (the edge that produced it)

`rows` is what stitches the result back onto the frame it came from: the new
frame is the old one taken at `rows`, plus the columns above. No key, no join,
no hash table.

A named relation narrows each node's slice to a sub-range. Because the slice is
sorted by relation, that sub-range is contiguous too, so it is again a pair of
bounds -- computed once per relation and memoized on the graph.
"""

import numpy as np


def bounds(graph, code, reverse):
    """(lo, hi) per node: the CSR range holding its edges of one relation.

    `code=None` is the wildcard, where the range is the node's whole slice."""
    indptr = graph.in_indptr if reverse else graph.out_indptr
    if code is None:
        return indptr[:-1], indptr[1:]
    return graph.cached(("bounds", code, reverse), lambda: _bounds(graph, code, reverse))


def _bounds(graph, code, reverse):
    # within a node's slice the edges are sorted by relation, so the ones
    # carrying `code` sit after every edge of a lower-numbered relation and
    # before every edge of a higher one. Counting both per node is two bincounts
    # over the whole store, once per relation asked for.
    indptr = graph.in_indptr if reverse else graph.out_indptr
    rels = graph.in_rels if reverse else graph.out_rels
    sources = graph.sources(reverse)
    size = graph.n_nodes
    before = np.bincount(sources[rels < code], minlength=size)
    length = np.bincount(sources[rels == code], minlength=size)
    lo = indptr[:-1] + before
    return lo, lo + length


def ranges(starts, counts):
    """The concatenation of `arange(starts[i], starts[i] + counts[i])`.

    Written as one cumulative sum: the result increases by 1 inside a range and
    jumps to the next start between two of them, so the whole thing is a vector
    of increments summed in C."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    keep = counts > 0
    starts, counts = starts[keep], counts[keep]
    steps = np.ones(total, dtype=np.int64)
    steps[0] = starts[0]
    if len(starts) > 1:
        ends = np.cumsum(counts)[:-1]
        steps[ends] = starts[1:] - starts[:-1] - counts[:-1] + 1
    return np.cumsum(steps)


def _side(graph, nodes, code, reverse):
    """One direction of one hop."""
    lo, hi = bounds(graph, code, reverse)
    starts, counts = lo[nodes], (hi - lo)[nodes]
    positions = ranges(starts, counts)
    rows = np.repeat(np.arange(len(nodes), dtype=np.int64), counts)
    indices = graph.in_indices if reverse else graph.out_indices
    rels = graph.in_rels if reverse else graph.out_rels
    weights = graph.weights()[1 if reverse else 0]
    codes = rels[positions].astype(np.int64)
    return rows, indices[positions], (~codes if reverse else codes), weights[positions]


def expand(graph, nodes, relation=None, reverse=None, normalized=False):
    """Every edge leaving a set of nodes, as four parallel arrays.

    `reverse` is False forwards, True backwards, None both -- the wildcard's
    natural reading. A code comes back negated (`~code`) when the edge was walked
    against the direction it is stored in, so one integer carries both.

    The order is the store's: by the row walked from, then forwards before
    backwards, then by relation, then by the node reached -- so it does not
    depend on how many rows were walked at once."""
    nodes = np.asarray(nodes, dtype=np.int64)
    code = None if relation is None else graph.relation_code(relation)
    if relation is not None and code is None:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.astype(np.int32), empty, empty.astype(np.float64)

    sides = []
    if reverse is not True:
        sides.append(_side(graph, nodes, code, False))
    if reverse is not False:
        sides.append(_side(graph, nodes, code, True))
    if len(sides) == 1:
        rows, targets, codes, weights = sides[0]
    else:
        rows, targets, codes, weights = interleave(sides)

    if normalized:
        weights = _renormalize(graph, codes, weights)
    return rows, targets, codes, weights


class Reach:
    """Every edge of one step that lands in a set, found from the set's end.

    `expand` walks out of a frame's nodes and keeps whatever it lands on. When
    the landing is constrained to a set that costs less to walk than the frame
    does, the same edges are found by walking *into* the set instead: its
    in-edges are the step's edges read the other way. Found once, then gathered
    onto any slice of the frame -- same rows, same order as `expand` followed by
    the filter, because the store is canonical: within a node and a relation,
    both directions order their edges by the node at the other end.

        reach = Reach(graph, wanted, "has_interact", False)
        rows, targets, codes, weights = reach.gather(nodes)
    """

    __slots__ = ("graph", "side", "sources", "targets", "codes", "weights")

    def __init__(self, graph, targets, relation=None, reverse=None):
        self.graph = graph
        targets = np.unique(np.asarray(targets, dtype=np.int64))
        code = None if relation is None else graph.relation_code(relation)
        parts = []
        if relation is None or code is not None:
            for side, backwards in enumerate((False, True)):
                if reverse is not None and reverse is not backwards:
                    continue
                # the step read forwards leaves a source along out-edges and lands
                # on a target; read from the target, that is its in-edges
                rows, sources, signed, weights = _side(graph, targets, code, not backwards)
                positive = np.where(signed < 0, ~signed, signed)
                parts.append((np.full(len(rows), side, dtype=np.int8), sources,
                              targets[rows], ~positive if backwards else positive,
                              weights))
        if parts:
            self.side, self.sources, self.targets, self.codes, self.weights = (
                np.concatenate(column) for column in zip(*parts))
        else:
            empty = np.zeros(0, dtype=np.int64)
            self.side, self.sources, self.targets, self.codes, self.weights = (
                empty.astype(np.int8), empty, empty, empty, empty.astype(np.float64))

    @property
    def size(self):
        """How many edges land in the set: what walking from this end cost."""
        return len(self.sources)

    def degree(self):
        """Per node of the graph, how many of these edges leave it -- what one
        step out of a frame would produce, walked this way."""
        return np.bincount(self.sources, minlength=self.graph.n_nodes)

    def gather(self, nodes, normalized=False):
        """The four arrays `expand(graph, nodes, ...)` would return, kept to the
        edges into the set, in the order `expand` would return them."""
        nodes = np.asarray(nodes, dtype=np.int64)
        order = np.argsort(nodes, kind="stable")
        ordered = nodes[order]
        lo = np.searchsorted(ordered, self.sources, side="left")
        counts = np.searchsorted(ordered, self.sources, side="right") - lo
        rows = order[ranges(lo, counts)]
        edge = np.repeat(np.arange(self.size, dtype=np.int64), counts)
        codes = self.codes[edge]
        targets = self.targets[edge]
        # expand's order: the frame's row, then the forward side before the
        # backward one, then the relation, then the node at the far end
        positive = np.where(codes < 0, ~codes, codes)
        key = np.lexsort((targets, positive, self.side[edge], rows))
        rows, targets, codes = rows[key], targets[key].astype(np.int32), codes[key]
        weights = self.weights[edge][key]
        if normalized:
            weights = _renormalize(self.graph, codes, weights)
        return rows, targets, codes, weights


class Ranked:
    """Segments of an array of edges, each ordered and summed by a map.

    The edges lie segment after segment -- a node's edges in the store, a row's
    candidates in a walk -- and the map says what each is worth. Ordered once,
    the n best of a segment are the first n of its order; summed once, a draw in
    proportion to the map is a binary search. Both cost what they keep.

    A map that says the same of every edge orders nothing: the first n are as
    good as any, and a draw is a uniform offset into the segment."""

    __slots__ = ("count", "start", "ranking", "order", "cumulative", "descending")

    def __init__(self, count, ranking=None):
        self.count = np.asarray(count, dtype=np.int64)
        self.start = np.cumsum(self.count) - self.count
        self.ranking = self.order = self.cumulative = self.descending = None
        if ranking is None or not len(ranking) or ranking.min() == ranking.max():
            return
        self.ranking = ranking
        index = np.int32 if len(ranking) < 2**31 else np.int64
        segment = np.repeat(np.arange(len(self.count), dtype=index), self.count)
        # within a segment the order is the map's; between them, the array's
        self.order = np.lexsort((-ranking, segment)).astype(index)

    def best(self, segments, width):
        """(rows, index): the `width` best edges of each segment asked for,
        best first -- `rows` says which of `segments` each came from."""
        counts = np.minimum(self.count[segments], width)
        rows = np.repeat(np.arange(len(segments), dtype=np.int64), counts)
        chosen = ranges(self.start[segments], counts)
        return rows, (chosen if self.order is None else self.order[chosen])

    def nucleus(self, segments, share):
        """(rows, index): the fewest best edges of each segment holding
        `share` of its mass -- few where one edge dominates, many where none
        does. A running sum in the map's order makes it a binary search."""
        lo = self.start[segments]
        count = self.count[segments]
        if self.ranking is None:
            counts = np.ceil(share * count).astype(np.int64)
        else:
            if self.descending is None:
                self.descending = np.cumsum(self.ranking[self.order], dtype=np.float64)
            running = self.descending
            below = np.where(lo > 0, running[np.maximum(lo - 1, 0)], 0.0)
            mass = np.where(count > 0, running[np.maximum(lo + count - 1, 0)], 0.0) - below
            # the first edge whose running total reaches the share, inclusive
            reach = np.searchsorted(running, below + share * mass * (1 - 1e-12), side="left")
            counts = np.where(mass > 0, np.clip(reach - lo + 1, 1, count), 0)
        rows = np.repeat(np.arange(len(segments), dtype=np.int64), counts)
        chosen = ranges(lo, counts)
        return rows, (chosen if self.order is None else self.order[chosen])

    def draw(self, segments, width, seed, keys):
        """(rows, index): `width` draws per segment in proportion to the map,
        with replacement, folded to the distinct edges.

        A draw is a function of the seed, the segment's key and which draw it
        is (`_uniform`), never of where the segment sits in what was asked: a
        node reached twice draws the same edges both times, and a walk taken in
        slices draws what it would have whole.

        A segment whose edges all weigh nothing has nothing to draw from and
        contributes no rows, as one with no edges does."""
        lo = self.start[segments]
        hi = lo + self.count[segments]
        if self.ranking is None:
            live = np.flatnonzero(hi > lo)
            rows = np.repeat(live, width)
            uniform = _uniform(seed, np.asarray(keys)[rows], width)
            index = lo[rows] + (uniform * (hi - lo)[rows]).astype(np.int64)
        else:
            if self.cumulative is None:
                self.cumulative = np.cumsum(self.ranking, dtype=np.float64)
            cumulative = self.cumulative
            below = np.where(lo > 0, cumulative[np.maximum(lo - 1, 0)], 0.0)
            mass = np.where(hi > lo, cumulative[np.maximum(hi - 1, 0)], 0.0) - below
            live = np.flatnonzero(mass > 0)
            rows = np.repeat(live, width)
            drawn = below[rows] + _uniform(seed, np.asarray(keys)[rows], width) * mass[rows]
            index = np.clip(np.searchsorted(cumulative, drawn, side="right"),
                            lo[rows], hi[rows] - 1)
        # one row per distinct edge: a draw landing twice is one arrival
        keep = np.unique(rows * (int(self.count.sum()) + 1) + index, return_index=True)[1]
        keep.sort()
        return rows[keep], index[keep]


class Fused:
    """Some relations' edges, one direction, ranked by a map once for all.

    A map that reads only the edge and where it lands is a fact about the graph,
    so it is folded into the store: a `Ranked` whose segments are the nodes.
    A budgeted step out of any node then costs what it keeps rather than what
    the node's degree is.

        fused = Fused(graph, codes, reverse, ranking)   # ranking over `members`
        rows, positions = fused.best(nodes, 20)
        rows, positions = fused.draw(nodes, 20, seed)
        rows, positions = fused.nucleus(nodes, 0.9)
    """

    __slots__ = ("ranked", "positions", "shift")

    def __init__(self, graph, codes, reverse, ranking):
        if len(codes) == 1:
            # one relation is one contiguous run per node: its position is its
            # place in the run plus where the run starts, and needs no array
            lo, hi = bounds(graph, codes[0], reverse)
            self.ranked = Ranked(hi - lo, ranking)
            self.positions = None
            self.shift = lo - self.ranked.start
        else:
            self.positions = Fused.members(graph, codes, reverse)
            sources = graph.sources(reverse)[self.positions]
            self.ranked = Ranked(np.bincount(sources, minlength=graph.n_nodes), ranking)
            self.shift = None

    @staticmethod
    def members(graph, codes, reverse):
        """Every CSR position of these relations, in store order: the edges a
        map over them is evaluated on, and in that order."""
        rels = graph.in_rels if reverse else graph.out_rels
        return np.flatnonzero(np.isin(rels, codes))

    def best(self, nodes, width):
        rows, index = self.ranked.best(nodes, width)
        return rows, self._position(index, nodes[rows])

    def draw(self, nodes, width, seed):
        rows, index = self.ranked.draw(nodes, width, seed, nodes)
        return rows, self._position(index, nodes[rows])

    def nucleus(self, nodes, share):
        rows, index = self.ranked.nucleus(nodes, share)
        return rows, self._position(index, nodes[rows])

    def _position(self, index, nodes):
        if self.positions is not None:
            return self.positions[index]
        return index + self.shift[nodes]


def _uniform(seed, nodes, width):
    """One number in [0, 1) per (node, draw), from a counter-based hash
    (splitmix64): the same seed, node and draw give the same number anywhere.
    `nodes` holds each node `width` times in a row, one per draw."""
    draw = np.arange(len(nodes), dtype=np.uint64) % np.uint64(width)
    with np.errstate(over="ignore"):
        state = (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
                 + nodes.astype(np.uint64) * np.uint64(width) + draw)
        state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        state = state ^ (state >> np.uint64(31))
    return (state >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def gathered(graph, rows, positions, reverse, normalized=False):
    """The four arrays `expand` returns, for edges already chosen by position."""
    indices = graph.in_indices if reverse else graph.out_indices
    rels = graph.in_rels if reverse else graph.out_rels
    weights = graph.weights(normalized)[1 if reverse else 0]
    codes = rels[positions].astype(np.int64)
    return rows, indices[positions], (~codes if reverse else codes), weights[positions]


def interleave(parts):
    """Several walks out of the same rows as one, each row's edges together.

    Every part is already in row order, so a stable sort by row keeps each
    part's order within a row and the parts' order between them -- which is
    what makes a walk's order the same whether it was taken whole or a slice
    at a time."""
    rows, targets, codes, weights = (np.concatenate(column) for column in zip(*parts))
    order = np.argsort(rows, kind="stable")
    return rows[order], targets[order], codes[order], weights[order]


def _renormalize(graph, codes, weights):
    """The traversed weights on their own relation's [0, 1] scale.

    Per relation because scales do not compare across them: a 1-5 rating and a
    cosine similarity are both floats and mean nothing to each other."""
    low, high = graph.weight_bounds()
    positive = np.where(codes < 0, ~codes, codes)
    span = high[positive] - low[positive]
    scaled = np.ones(len(weights))
    varying = span > 0
    scaled[varying] = (weights[varying] - low[positive][varying]) / span[varying]
    return scaled
