"""Key: one node, outside the frame.

Internally a node is an integer and a relation is a small code -- that is what
makes CSR slices and per-node arrays possible (see graph.py). A frame carries
those integers, because that is what indexes an array; `frame.keys(...)` is
where one becomes a thing that knows what it is.

There is no `Rel` here any more. A traversed relation is a column of names, with
a leading `~` for a step taken against the stored direction -- readable when the
frame prints, and comparable with `==` like any other string column.

Key is not an int subclass: variable-length builtins cannot carry __slots__, and
`str(key)` would then have to fight int's own formatting. It defines __index__
instead, so a Key still works directly as a numpy index or a dict key -- a
strategy can write `embeddings[key]` with no conversion -- while printing and
comparing on its own terms.
"""

from functools import total_ordering


@total_ordering
class Key:
    """One node, as returned by a query: `movie.123`.

    Carries the graph it came from so the things a caller actually wants -- the
    type, the original id, a human label, the attributes -- are reachable
    without reaching back into the Graph by hand.
    """

    __slots__ = ("_graph", "_index")

    def __init__(self, graph, index):
        self._graph = graph
        self._index = index

    @property
    def type(self):
        return self._graph.type_of(self._index)

    @property
    def id(self):
        """The id as it appears in the source files (an int for numeric ids)."""
        return self._graph.raw_id(self._index)

    @property
    def label(self):
        """What identifies this node outside the graph: the id its source used
        under `renumber`, and its position otherwise."""
        return self._graph.label_of(self._index)

    @property
    def attrs(self):
        return self._graph.attrs_of(self._index)

    def __index__(self):
        return self._index

    __int__ = __index__

    def __hash__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, Key):
            return self._index == other._index and self._graph is other._graph
        return NotImplemented

    def __lt__(self, other):
        # rows are sorted for deterministic output; ordering by index is the
        # graph's own load order, which is stable across runs
        if isinstance(other, Key):
            return self._index < other._index
        return NotImplemented

    def __str__(self):
        return f"{self.type}.{self.id}"

    __repr__ = __str__

    def __format__(self, spec):
        # int.__format__ would win for an int subclass; here the only risk is an
        # empty spec silently falling back to object.__format__, so be explicit
        return format(str(self), spec)

    def __reduce__(self):
        """A Key holds its graph so a caller can read a label off it, which the
        default pickling would then serialise -- 4.6 MB for one MovieLens node,
        gigabytes for one Spotify song. It is a *reference into* a graph, and a
        reference is meaningless without the thing it points at, so it refuses
        rather than quietly copying one. `str(key)` is what travels."""
        raise TypeError(
            f"{self} cannot be pickled: a Key points into a Graph, and pickling "
            f"it would carry the whole graph along. Send str(key) and resolve it "
            f"with graph[...] on the other side.")
