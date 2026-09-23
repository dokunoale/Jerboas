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
    natural reading, and what closes a bridge pattern without the store holding
    each edge twice. A code comes back negated (`~code`) when the edge was walked
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
