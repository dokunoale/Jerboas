"""References: the passive nouns of the DSL.

None of these classes know anything about a Graph. `Node("movie").name` is just
`Attr(node, "name")`; whether "name" is a column or a relation, and what its
label column is, is decided by the Query at compile time. This is the whole
point of moving from `g.node(type)` to a bare `Node(type)`: the reference is
pure syntax, the Query connects it to the graph.

Operators build Conditions (see conditions.py) and never touch the graph, so the
model stays fully lazy: nothing runs until the Query does.
"""

from .core import Ref, Expr, And
from .conditions import Compare, Has, In, Match


class Node(Ref):
    """A pattern variable, optionally typed and pre-constrained by kwargs.

        Node("movie")
        Node("movie", year=1994)          # kwargs are sugar for `.attr == value`
        Node()                            # untyped wildcard (e.g. a path midpoint)

    Attribute access returns an Attr; the Query decides at compile time whether
    that name is a column or a relation -- driven by how the Attr is used, e.g.
    `movie.directed_by == Node("person")` is a relation, `movie.year >= 1990`
    is a column. Two Nodes are distinct variables even if they look identical
    (identity semantics), so reuse the *same* Node object to mean the same
    variable across select/where.
    """

    def __init__(self, type=None, **attrs):
        self.type = type
        self.attrs = attrs        # {name: value} sugar, expanded to Compare at compile

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return Attr(self, name)

    # membership that operators can't express: fluent sugar for the In object.
    def is_in(self, keys):
        """The In condition on this node's identity: bind only to these full keys.
        Fluent equivalent of `In(node, keys)` -- one mechanism, two spellings."""
        return In(self, keys)


class Attr(Expr):
    """A reference to a Node's attribute *or* relation: `node.<name>`.

    Which one it is is never decided here -- it is decided by what you do with
    it:
        node.year >= 1990                 -> Compare  (a column predicate)
        node.title == "Akira"             -> Compare  (a column predicate)
        node.directed_by == Node("person")-> Has      (a relation edge)
        Like(node.title, "akira")         -> Like     (a fuzzy column match)
        select(node.title)                -> projection

    __eq__ is the pivot: a Ref on the right means "relation to that node", any
    other value means "column equals value".

    `.inverse` walks a relation backwards -- `person.directed_by.inverse` is the
    movies a person directed. There is no `directed_by_r`: the reverse of a
    relation is the same relation, and the graph stores it once.

    Two names are resolved rather than looked up literally. `.id` is the node's
    own identifier, and `.label` is whichever column carries the type's readable
    name -- `title` for a movie, `name` for a person. That is what lets one
    query serve every type:

        Like(Node(type_).label.is_in(names))    # works for movie, person, genre

    On an untyped Node, `.label` resolves per type as the search crosses them.
    """

    def __init__(self, node, name, reverse=False):
        self.node = node
        self.name = name
        self.reverse = reverse

    @property
    def inverse(self):
        """The same relation, traversed from target to source."""
        return Attr(self.node, self.name, not self.reverse)

    def __eq__(self, other):
        if isinstance(other, Ref):                # relation: node -name-> other
            return Has(self.node, self.name, other, self.reverse)
        return Compare(self, "eq", other)

    def __ne__(self, other):
        return Compare(self, "ne", other)

    def __lt__(self, other):
        return Compare(self, "lt", other)

    def __le__(self, other):
        return Compare(self, "le", other)

    def __gt__(self, other):
        return Compare(self, "gt", other)

    def __ge__(self, other):
        return Compare(self, "ge", other)

    def is_in(self, values):
        """node.<attr> is one of these values (Polars `.is_in`). Softens under
        Like(...) to set similarity."""
        return In(self, values)

    def is_between(self, lower, upper):
        """lower <= node.<attr> <= upper (Polars `.is_between`)."""
        return And(self >= lower, self <= upper)

    def contains(self, text):
        """Crisp substring match on node.<attr> (Polars `.str.contains`, flattened).
        Softens under Like(...) to fuzzy similarity."""
        return Compare(self, "contains", text)

    def count(self):
        """The relation's arity as a scalar expression: `node.directed_by.count()`.
        Usable as a predicate (`>= 2`), a projection, or a rank signal."""
        return Degree(self.node, self.name, self.reverse)

    # Attr is used as a dict-ish key inside the compiler; keep hashing by identity
    # even though __eq__ is overloaded to build a Condition.
    __hash__ = object.__hash__


class Edge:
    """A relation marker for path patterns: `Edge("directed_by")`, or `Edge()`
    for a wildcard. Edges are how traversal is written when you prefer an
    explicit path over the `attr == Node(...)` relation form; both compile to
    the same edge in the IR.

    Direction: a named Edge walks source -> target, and `.inverse` walks it the
    other way. The wildcard `Edge()` walks *both* directions -- which is what a
    two-hop pattern like [movie, Edge(), attribute, Edge(), movie] needs, and
    what the old `rv=True` edge duplication was there to fake."""

    def __init__(self, *relations, reverse=False):
        self.relations = relations   # empty tuple = wildcard
        self.reverse = reverse

    @property
    def inverse(self):
        return Edge(*self.relations, reverse=not self.reverse)


class Path(Ref):
    """A variable bound to a whole walk (nodes + the concrete relations between
    them). Constrained by equating it to a pattern list:

        path == [rec, Edge(), seed]                       # rec -*-> seed
        path == [rec, Edge(), Node(), Edge(), seed]       # a 2-hop bridge

    `path == [...]` returns a Match condition; two Matches OR together for
    alternative shapes. path[0] / path[-1] (or .first / .last) reference a node
    inside the walk, itself projectable to an attribute.
    """

    def __eq__(self, pattern):
        return Match(self, pattern)

    def __getitem__(self, index):
        return PathStep(self, index)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "first":
            return PathStep(self, 0)
        if name == "last":
            return PathStep(self, -1)
        raise AttributeError(name)

    __hash__ = object.__hash__


class PathStep(Ref):
    """A node position inside a Path (path[-1], path.first). Projectable as-is,
    or narrowed to an attribute: `path.last.title`."""

    def __init__(self, path, index):
        self.path = path
        self.index = index

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return PathStepAttr(self.path, self.index, name)


class PathStepAttr(Ref):
    """An attribute of a node inside a Path: `path.last.title`."""

    def __init__(self, path, index, name):
        self.path = path
        self.index = index
        self.name = name


class Degree(Expr):
    """The arity of a node's relation as a scalar expression: `node.rel.count()`.

    One expression, three uses -- like a Polars expression across select/filter/
    agg -- but all three are the *expression* role, not a blur of families:
        select(rec.has_interact.inverse.count())       # projection
        where(rec.has_interact.inverse.count() >= 5)   # Compare condition
        rank(rec.has_interact.inverse.count())         # rank signal (Strategy adapter)

    Comparisons build a Compare like any other Expr; the Query knows a Compare on
    a Degree is structural (counted against the graph) rather than a stored attr.
    Using a Degree in rank(...) wraps it in a trivial Strategy that scores by the
    count (see strategies.ExprStrategy).
    """

    def __init__(self, node, relation, reverse=False):
        self.node = node
        self.relation = relation
        self.reverse = reverse

    def __eq__(self, other):
        return Compare(self, "eq", other)

    def __ne__(self, other):
        return Compare(self, "ne", other)

    def __lt__(self, other):
        return Compare(self, "lt", other)

    def __le__(self, other):
        return Compare(self, "le", other)

    def __gt__(self, other):
        return Compare(self, "gt", other)

    def __ge__(self, other):
        return Compare(self, "ge", other)

    __hash__ = object.__hash__
