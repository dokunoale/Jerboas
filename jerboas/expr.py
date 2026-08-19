"""Names that stay names until a frame resolves them.

A pattern variable is a column, so `v.rec` is the column `rec` and `v.rec.year`
the column `rec.year`. That much was always true. What is new here is *when* the
name is decided: nothing is turned into a polars expression at the point it is
written. `v.movie.has_genre >= 2` is a little tree of names, and the Frame --
which is the only thing that knows the graph -- resolves it.

Late resolution buys three things that eager construction cannot:

    .filter(v.movie.year >= 1990)              # the attribute is read on demand
    .filter(v.movie.has_genre.count() >= 2)    # a relation's arity, not a column
    .filter(v.movie.directed_by.is_in(people)) # the edge exists, unexpanded

and, because the predicate reaches the Frame before it reaches polars, the Frame
can decide *where* to apply it -- to the rows a hop has not built yet rather than
to the rows it has. That is the old compiler's admission mask, obtained by
writing an ordinary filter.

A name resolves in this order, and the order is the whole rule:

    1. a column the frame already has
    2. an attribute of that variable's type   -> read from the graph, on demand
    3. a relation of the graph                -> arity with .count(), existence
                                                 with .is_in(...)

`v.movie.attr.x` and `v.movie.rel.x` say which namespace to use when a type has
an attribute named like a relation. `.expr` drops out to raw polars, unresolved.
"""

import numpy as np
import polars as pl

from .keys import Key

# the two escapes, for a type whose attribute is named like a relation
ATTR, REL = "attr", "rel"


class Expr:
    """A predicate or a value, not yet bound to a frame.

    Subclasses implement `resolve(ctx)`, which returns a polars expression (or,
    for a bare relation, the marker the method around it needs)."""

    def resolve(self, ctx):
        raise NotImplementedError

    def reads(self):
        """The name paths this expression reads, for deciding what it depends
        on before deciding where to apply it."""
        return ()

    # -- comparison --

    def __eq__(self, other):
        return _Binary("__eq__", self, other)

    def __ne__(self, other):
        return _Binary("__ne__", self, other)

    def __lt__(self, other):
        return _Binary("__lt__", self, other)

    def __le__(self, other):
        return _Binary("__le__", self, other)

    def __gt__(self, other):
        return _Binary("__gt__", self, other)

    def __ge__(self, other):
        return _Binary("__ge__", self, other)

    # -- arithmetic, so several signals combine by weights that are written down --

    def __add__(self, other):
        return _Binary("__add__", self, other)

    def __radd__(self, other):
        return _Binary("__radd__", self, other)

    def __sub__(self, other):
        return _Binary("__sub__", self, other)

    def __rsub__(self, other):
        return _Binary("__rsub__", self, other)

    def __mul__(self, other):
        return _Binary("__mul__", self, other)

    def __rmul__(self, other):
        return _Binary("__rmul__", self, other)

    def __truediv__(self, other):
        return _Binary("__truediv__", self, other)

    def __rtruediv__(self, other):
        return _Binary("__rtruediv__", self, other)

    def __neg__(self):
        return _Unary("__neg__", self)

    # -- boolean --

    def __and__(self, other):
        return _Binary("__and__", self, other)

    def __or__(self, other):
        return _Binary("__or__", self, other)

    def __invert__(self):
        return _Unary("__invert__", self)

    def __bool__(self):
        raise TypeError(
            "a jerboas expression has no truth value, so `and`, `or`, `not` and "
            "chained comparisons would silently drop half of what you wrote. Use "
            "`&`, `|`, `~`, and `.is_between(a, b)` for a range.")

    def __hash__(self):
        return id(self)


class _Methods:
    """The named questions an expression answers.

    Deliberately not on `Col`: attribute access there is the column path, and a
    method defined on the class would win the lookup and silently shadow every
    column of that name -- `v.person.name` meaning something other than the
    person's name. A `Col` reaches these by being called (see `_Field`), so the
    two readings never compete."""

    def _call(self, method, *args, **kwargs):
        return _Method(self, method, args, kwargs)

    def sum(self):
        return self._call("sum")

    def count(self):
        return self._call("count")

    def n_unique(self):
        return self._call("n_unique")

    def mean(self):
        return self._call("mean")

    def min(self):
        return self._call("min")

    def max(self):
        return self._call("max")

    def std(self):
        return self._call("std")

    def first(self):
        return self._call("first")

    def last(self):
        return self._call("last")

    def abs(self):
        return self._call("abs")

    def alias(self, name):
        return self._call("alias", name)

    def is_null(self):
        return self._call("is_null")

    def is_not_null(self):
        return self._call("is_not_null")

    def is_between(self, low, high):
        return self._call("is_between", low, high)

    def contains(self, text):
        return _Contains(self, text)

    def is_in(self, values):
        return _In(self, values)

    def norm(self):
        return _Norm(self)


class Col(Expr):
    """A name path: `v.rec` is ("rec",), `v.rec.year` is ("rec", "year").

    Attribute access here means one thing only -- the path grows -- because a
    column may be called anything, `name` and `count` and `min` included. What
    would be a method elsewhere is reached by *calling* the name instead:
    `v.x.sum` is the column `x.sum`, and `v.x.sum()` is the sum of `x`."""

    __slots__ = ("_path",)

    def __init__(self, path):
        self._path = tuple(path) if not isinstance(path, str) else tuple(path.split("."))

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return _Field(self._path + (attr,), self)

    def __str__(self):
        return ".".join(self._path)

    def __repr__(self):
        return f"v.{self}"

    def __hash__(self):
        return hash(self._path)

    @property
    def expr(self):
        """Raw polars, unresolved: the column by this exact name, whatever the
        graph might have said about it. (A column actually named `expr` is
        reached with `col("x.expr")`.)"""
        return pl.col(str(self))

    def reads(self):
        return (self._path,)

    def resolve(self, ctx):
        return ctx.lookup(self._path)


class _Field(Col):
    """A name that is also, when called, the method of that name on its parent.

    `v.person.name` is the column; `v.person.count()` is a count of `v.person`.
    Which one it is, is decided by whether it is called -- so no method name is
    ever unreachable as a column, and no column name ever hides a method."""

    __slots__ = ("_parent",)

    def __init__(self, path, parent):
        super().__init__(path)
        self._parent = parent

    def __call__(self, *args, **kwargs):
        method = self._path[-1]
        if method == "is_in":
            return _In(self._parent, *args, **kwargs)
        if method == "contains":
            return _Contains(self._parent, *args, **kwargs)
        if method == "norm":
            return _Norm(self._parent, *args, **kwargs)
        if method not in _NAMED:
            raise AttributeError(
                f"{self._parent!r} has no method {method!r}; as a column it would "
                f"be {str(self)!r}, which this frame does not have.")
        return _Method(self._parent, method, args, kwargs)


class _Vars:
    """`v.rec` is the column named "rec". There is nothing to construct, and no
    two of them are ever different variables."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return Col((name,))

    def __getitem__(self, name):
        return Col(name)


v = _Vars()


def col(name):
    """`col("rec.title")` -- the same thing as `v.rec.title`, for a name that
    collides with a method or is built at runtime."""
    return Col(name)


class Relation:
    """What a name resolves to when the graph calls it a relation.

    Not a column: a relation has no per-row value, so it is only meaningful
    under the two questions that turn one into a number or a fact -- how many
    (`.count()`) and whether any (`.is_in(...)`, or bare)."""

    __slots__ = ("var", "name")

    def __init__(self, var, name):
        self.var = var
        self.name = name


# --------------------------------------------------------------------------- #
# The tree                                                                     #
# --------------------------------------------------------------------------- #

class _Binary(Expr, _Methods):
    __slots__ = ("op", "left", "right")

    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right

    def reads(self):
        return _reads(self.left) + _reads(self.right)

    def resolve(self, ctx):
        # a reflected operator resolves the same way: `0.7 * v.pr` is
        # `col.__rmul__(0.7)`, which polars reads as `0.7 * col`
        left, right = _side(self.left, ctx), _side(self.right, ctx)
        return getattr(_expr(left, self.op), self.op)(right)

    __hash__ = Expr.__hash__


class _Unary(Expr, _Methods):
    __slots__ = ("op", "operand")

    def __init__(self, op, operand):
        self.op, self.operand = op, operand

    def reads(self):
        return _reads(self.operand)

    def resolve(self, ctx):
        operand = _side(self.operand, ctx)
        if isinstance(operand, Relation) and self.op == "__invert__":
            return ~ctx.exists(operand, None)
        return getattr(operand, self.op)()

    __hash__ = Expr.__hash__


class _Method(Expr, _Methods):
    """A method on a name. `count` is the one that means something different on
    a relation: the graph's arity rather than the column's length."""

    __slots__ = ("target", "method", "args", "kwargs")

    def __init__(self, target, method, args=(), kwargs=None):
        self.target, self.method = target, method
        self.args, self.kwargs = args, kwargs or {}

    def reads(self):
        return _reads(self.target)

    def resolve(self, ctx):
        target = _side(self.target, ctx)
        if isinstance(target, Relation):
            if self.method != "count":
                raise TypeError(
                    f"{target.name!r} is a relation of the graph, and a relation has "
                    f"no {self.method}(): ask how many with .count(), or whether any "
                    f"with .is_in(...).")
            return ctx.degree(target)
        return getattr(target, self.method)(*self.args, **self.kwargs)

    __hash__ = Expr.__hash__


class _Contains(Expr, _Methods):
    __slots__ = ("target", "text")

    def __init__(self, target, text):
        self.target, self.text = target, text

    def reads(self):
        return _reads(self.target)

    def resolve(self, ctx):
        target = _expr(_side(self.target, ctx), "contains")
        return target.cast(pl.String).str.contains(str(self.text), literal=True)

    __hash__ = Expr.__hash__


class _In(Expr, _Methods):
    """Set membership, and the one place a name means two different questions.

    On a column it is the ordinary one. On a relation it asks whether an edge to
    that set exists -- which is a filter the graph answers without the frame
    walking anything."""

    __slots__ = ("target", "values")

    def __init__(self, target, values):
        self.target, self.values = target, values

    def reads(self):
        return _reads(self.target)

    def resolve(self, ctx):
        target = _side(self.target, ctx)
        if isinstance(target, Relation):
            return ctx.exists(target, self.values)
        return _expr(target, "is_in").is_in(ctx.ids(self.values))

    __hash__ = Expr.__hash__


class _Norm(Expr, _Methods):
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target

    def reads(self):
        return _reads(self.target)

    def resolve(self, ctx):
        return norm(_expr(_side(self.target, ctx), "norm"))

    __hash__ = Expr.__hash__


# the methods a name may be called as, beyond the three with their own nodes
_NAMED = ("sum", "count", "n_unique", "mean", "min", "max", "std", "first", "last",
          "abs", "alias", "is_null", "is_not_null", "is_between")


def _reads(thing):
    return thing.reads() if isinstance(thing, Expr) else ()


def _side(thing, ctx):
    """One operand, resolved: an Expr becomes a polars expression (or a
    Relation), and anything else is a literal polars already understands."""
    return thing.resolve(ctx) if isinstance(thing, Expr) else thing


def _expr(thing, what):
    if isinstance(thing, Relation):
        raise TypeError(
            f"{thing.name!r} is a relation of the graph, and {what} needs a value: "
            f"ask how many with .count(), or whether any with .is_in(...).")
    return thing


def norm(expr):
    """Min-max into [0, 1]. A column with no spread is all 1.0 rather than all
    0.0, so a signal that says the same about everything does not veto the ones
    that say something."""
    expr = expression(expr)
    span = expr.max() - expr.min()
    return pl.when(span > 0).then((expr - expr.min()) / span).otherwise(pl.lit(1.0))


def name_of(thing):
    """A column name from whatever names a column."""
    if isinstance(thing, Col):
        return str(thing)
    if isinstance(thing, str):
        return thing
    if isinstance(thing, pl.Expr):
        return thing.meta.output_name()
    raise TypeError(f"expected a column name, got {thing!r}")


def expression(thing):
    """A polars expression from a name, a Col, or an expression -- with no graph
    to consult, so a Col means exactly the column it spells."""
    if isinstance(thing, Col):
        return thing.expr
    if isinstance(thing, str):
        return pl.col(thing)
    return thing


def ids(values):
    """A set of nodes as an int32 array, however it was named.

    Accepts a Frame (its node column), Keys, plain integers and numpy arrays.
    A source string like "movie.12" is not accepted here: resolving one needs the
    graph, which this function does not have -- Frame does it instead."""
    if hasattr(values, "ids"):                      # a Frame
        return values.ids()
    if isinstance(values, np.ndarray):
        return values.astype(np.int32, copy=False)
    if isinstance(values, pl.Series):
        return values.to_numpy().astype(np.int32, copy=False)
    out = []
    for value in values:
        if isinstance(value, (Key, int, np.integer)):
            out.append(int(value))
        else:
            raise TypeError(
                f"cannot resolve {value!r} to a node without a graph: pass Keys "
                f"(g[\"movie.12\"]), a Frame, or integer ids.")
    return np.asarray(out, dtype=np.int32)
