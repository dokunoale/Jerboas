"""Column names, and the sugar that turns one into an expression.

A pattern variable is a column of the frame, and nothing else: `v.rec` names the
column `rec`, `v.rec.year` names the column `rec.year`. That is the whole
identity system. Two `v.rec` are the same variable because they are the same
string -- which is what the old Ref family, with its identity semantics and its
`{id(ref): column}` map, existed to work around.

`Col` is a name that also behaves like a polars expression, so a filter reads the
way it always did:

    frame.filter(v.movie.year >= 1990)          # pl.col("movie.year") >= 1990
    frame.top(10, by=v.score)                   # a name here, an expression there

Anything polars can do to a column, `.expr` hands back: `v.rec.expr.shift(1)`.
Attribute access is the column path, so the few method names defined here
(`sum`, `is_in`, ...) are not reachable as column names -- `col("rec.sum")`
spells that one out.
"""

import numpy as np
import polars as pl

from .keys import Key

_METHODS = ("sum", "count", "mean", "min", "max", "n_unique", "first", "last",
            "std", "norm", "is_in", "is_null", "is_not_null", "is_between",
            "contains", "alias", "abs", "expr", "name")


class Col:
    """A column name with expression behaviour."""

    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = str(name)

    # -- the name, and the path below it --

    @property
    def name(self):
        return self._name

    def __getattr__(self, attr):
        # only reached for names that are not methods above: those are the
        # attribute columns, `v.rec` -> `v.rec.title` -> "rec.title"
        if attr.startswith("_"):
            raise AttributeError(attr)
        return Col(f"{self._name}.{attr}")

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"v.{self._name}"

    def __hash__(self):
        return hash(self._name)

    # -- the expression --

    @property
    def expr(self):
        return pl.col(self._name)

    def __eq__(self, other):
        return self.expr == _operand(other)

    def __ne__(self, other):
        return self.expr != _operand(other)

    def __lt__(self, other):
        return self.expr < _operand(other)

    def __le__(self, other):
        return self.expr <= _operand(other)

    def __gt__(self, other):
        return self.expr > _operand(other)

    def __ge__(self, other):
        return self.expr >= _operand(other)

    def __add__(self, other):
        return self.expr + _operand(other)

    def __radd__(self, other):
        return _operand(other) + self.expr

    def __sub__(self, other):
        return self.expr - _operand(other)

    def __rsub__(self, other):
        return _operand(other) - self.expr

    def __mul__(self, other):
        return self.expr * _operand(other)

    def __rmul__(self, other):
        return _operand(other) * self.expr

    def __truediv__(self, other):
        return self.expr / _operand(other)

    def __rtruediv__(self, other):
        return _operand(other) / self.expr

    def __neg__(self):
        return -self.expr

    def __invert__(self):
        return ~self.expr

    def __and__(self, other):
        return self.expr & _operand(other)

    def __or__(self, other):
        return self.expr | _operand(other)

    # -- the handful of methods worth having on a name --

    def sum(self):
        return self.expr.sum()

    def count(self):
        return self.expr.count()

    def n_unique(self):
        return self.expr.n_unique()

    def mean(self):
        return self.expr.mean()

    def min(self):
        return self.expr.min()

    def max(self):
        return self.expr.max()

    def std(self):
        return self.expr.std()

    def first(self):
        return self.expr.first()

    def last(self):
        return self.expr.last()

    def abs(self):
        return self.expr.abs()

    def alias(self, name):
        return self.expr.alias(name)

    def is_null(self):
        return self.expr.is_null()

    def is_not_null(self):
        return self.expr.is_not_null()

    def is_between(self, low, high):
        return self.expr.is_between(low, high)

    def contains(self, text):
        return self.expr.cast(pl.String).str.contains(str(text), literal=True)

    def is_in(self, values):
        return self.expr.is_in(ids(values))

    def norm(self):
        return norm(self.expr)


class _Vars:
    """`v.rec` is the column named "rec". There is nothing to construct, and no
    two of them are ever different variables."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return Col(name)

    def __getitem__(self, name):
        return Col(name)


v = _Vars()


def col(name):
    """`col("rec.title")` -- the same thing as `v.rec.title`, for a name that
    collides with a method or is built at runtime."""
    return Col(name)


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
        return thing.name
    if isinstance(thing, str):
        return thing
    if isinstance(thing, pl.Expr):
        return thing.meta.output_name()
    raise TypeError(f"expected a column name, got {thing!r}")


def expression(thing):
    """A polars expression from a name, a Col, or an expression."""
    if isinstance(thing, Col):
        return thing.expr
    if isinstance(thing, str):
        return pl.col(thing)
    return thing


def _operand(thing):
    return thing.expr if isinstance(thing, Col) else thing


def ids(values):
    """A set of nodes as an int32 array, however it was named.

    Accepts a Frame (its node column), Keys, plain integers and numpy arrays.
    A source string like "movie.12" is not accepted here: resolving one needs the
    graph, which an expression does not have -- `g["movie.12"]` does it."""
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
                f"cannot resolve {value!r} to a node here: an expression has no "
                f"graph to look a name up in. Pass Keys (g[\"movie.12\"]), a Frame, "
                f"or integer ids.")
    return np.asarray(out, dtype=np.int32)
