"""The values a query hands back: Key and Rel.

Internally a node is an integer and a relation is a small code -- that is what
makes masks, CSR slices and per-node arrays possible (see graph.py). Those
integers are meaningless on their own, so the pipeline re-attaches their meaning
at the very last step, in Query._render.

Key is not an int subclass: variable-length builtins cannot carry __slots__, and
`str(key)` would then have to fight int's own formatting. It defines __index__
instead, so a Key still works directly as a numpy index or a dict key -- a
strategy can write `embeddings[key]` with no conversion -- while printing and
comparing on its own terms.
"""


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
        """The type's label column (name/title/...), falling back to str(self)."""
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


class Rel:
    """One relation traversal inside a Path: a name plus the direction it was
    walked.

    There is no `_r` twin relation any more -- `directed_by` walked backwards is
    still `directed_by`, with reverse=True. Equality is (name, direction) and
    never compares equal to a bare string: a Rel that equalled "directed_by" in
    both directions would make the forward and reverse ones indistinguishable
    through a string, which is exactly the confusion the `_r` suffix caused.
    Compare `rel.name` when the direction does not matter.
    """

    __slots__ = ("name", "reverse")

    def __init__(self, name, reverse=False):
        self.name = name
        self.reverse = reverse

    def __hash__(self):
        return hash((self.name, self.reverse))

    def __eq__(self, other):
        if isinstance(other, Rel):
            return self.name == other.name and self.reverse == other.reverse
        return NotImplemented

    def __str__(self):
        return f"~{self.name}" if self.reverse else self.name

    __repr__ = __str__

    def __format__(self, spec):
        return format(str(self), spec)
