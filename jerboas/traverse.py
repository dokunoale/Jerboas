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
    against the direction it is stored in, so one integer carries both."""
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
        rows, targets, codes, weights = (np.concatenate(part) for part in zip(*sides))

    if normalized:
        weights = _renormalize(graph, codes, weights)
    return rows, targets, codes, weights


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
