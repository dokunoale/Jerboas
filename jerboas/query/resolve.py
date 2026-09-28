"""What a hop defers, and what turns a name into a column.

`Pending` is a hop that has walked but not built its rows: held one step, a
filter about the node just reached is applied to the arrays rather than to rows
that would then be dropped.

`Resolver` is where a name meets the graph. Four answers, in order: a column's
provenance (its confidence, the relation that revealed it, its type), a column
the frame has, an attribute of that variable's type, or a relation. What was
read only to decide is taken off again; what was *measured* stays.

"""

import numpy as np
import polars as pl

from .expr import (ATTR, NEEDLE, PROVENANCE, REL, SCORE, TYPE, Relation,
                   path_of, shadow)
from ..search.rules import Search, default_rule


class Pending:
    """A hop that has walked the graph but not yet built its rows.

    The walk produced four parallel arrays; turning them into a frame means
    taking every column the old frame had at `rows`, which is the expensive half
    of a hop. Holding them one step lets a filter about the node just reached be
    applied here -- to arrays -- instead of to rows that would then be dropped.

    Nothing else is deferred: any other verb reads `_df` and pays for it."""

    __slots__ = ("base", "rows", "added", "var")

    def __init__(self, base, rows, added, var):
        self.base = base            # the frame before the hop
        self.rows = rows            # which base row each candidate came from
        self.added = added          # {column: array} the step produced, shadows included
        self.var = var              # the new node column

    def narrow(self):
        """Just what the step produced -- the frame a pushable predicate is
        evaluated against."""
        return pl.DataFrame({name: _series(name, values)
                             for name, values in self.added.items()})

    def with_added(self, columns):
        """The same hop, carrying more columns beside what the step produced."""
        if not columns:
            return self
        return Pending(self.base, self.rows, {**self.added, **columns}, self.var)

    def masked(self, keep):
        return Pending(self.base, self.rows[keep],
                        {name: take(values, keep) for name, values in self.added.items()},
                        self.var)

    def build(self):
        data = self.base[self.rows] if len(self.rows) else self.base.clear()
        return data.with_columns([_series(name, values)
                                  for name, values in self.added.items()])


class Resolver:
    """What turns a name into a column, with the graph in hand.

    Four answers, in this order: the provenance of a column (its confidence, the
    relation that revealed it, the type its ids fall in), a column the frame
    already has, an attribute of that variable's type, or a relation of the
    graph. Only the first is free; the rest read one array out of the graph.

    Two kinds of thing get attached to the frame. What was read only to decide
    something is taken off again, because a filter filters rows and does not
    quietly widen the frame. What was *measured* stays -- a `like`'s closeness
    is the column's confidence from then on, and throwing it away would mean
    computing it twice or, worse, differently."""

    def __init__(self, frame):
        self.frame = frame
        self.extra = {}                    # name -> Series, for this verb only
        self.keep = {}                     # name -> Series, measured and kept
        self.constants = {}                # (kind, column) -> one value for every row

    # -- the surface an Expr resolves against --

    def lookup(self, path):
        name = ".".join(path)
        if self._present(name):
            return pl.col(name)
        if len(path) >= 2 and path[-1] in PROVENANCE and path[-2] != ATTR:
            return self._provenance(".".join(path[:-1]), path[-1])
        if len(path) >= 2 and path[0] in self.frame.vars:
            return self._below(path[0], path[1:])
        raise ValueError(
            f"no column {name!r}, and {path[0]!r} is not a node column of this "
            f"frame -- it has: {', '.join(self.frame.columns)}")

    def degree(self, relation):
        name = f"{relation.var}.{relation.name}_count"
        if name not in self.extra:
            counts = self.frame._degrees(relation.name, None)
            ids = self.frame._df[relation.var].to_numpy()
            self.extra[name] = pl.Series(name, counts[ids])
        return pl.col(name)

    def exists(self, relation, values):
        name = f"{relation.var}.{relation.name}_exists"
        if name not in self.extra:
            admissible = self.frame._reachable(relation.name, values, None)
            ids = self.frame._df[relation.var].to_numpy()
            self.extra[name] = pl.Series(name, admissible[ids])
        return pl.col(name)

    def like(self, column, needles, rule, exclusive):
        """What the rule admitted, and what it found kept.

        Two things, and they cost one computation: the closeness becomes the
        column's confidence, so admission and weight cannot disagree; and which
        needle each row answers becomes the column's `needle`, because that is
        the question a set of names asks and one name does not. With one needle
        the answer is the same for every row, so it is a value rather than a
        column."""
        name = str(column)
        asked = _needles(needles)
        search = Search(self.frame, name, lambda: self._values(name))
        found = (rule or default_rule()).matches(search, asked, exclusive)

        height = self.frame._df.height
        closeness = np.zeros(height)
        answered = [None] * height
        for row, (score, needle) in found.items():
            closeness[row] = score
            answered[row] = needle
        self.keep[shadow(SCORE, name)] = pl.Series(shadow(SCORE, name), closeness)
        if len(asked) == 1:
            self.constants[(NEEDLE, name)] = asked[0]
        elif _nameable(asked):
            self.keep[shadow(NEEDLE, name)] = pl.Series(
                shadow(NEEDLE, name), answered,
                dtype=pl.Enum([str(one) for one in dict.fromkeys(asked)]))
        else:
            self.keep[shadow(NEEDLE, name)] = pl.Series(shadow(NEEDLE, name), answered)

        admitted = f"{name}.__like__"
        self.extra[admitted] = pl.Series(admitted, closeness > 0)
        return pl.col(admitted)

    def signal(self, signal):
        """A strategy's score, as a literal column the rest of an expression can
        be arithmetic on."""
        name = f"_signal_{len(self.extra)}"
        arrays = tuple(self.frame._df[column].to_numpy() for column in signal.columns)
        self.extra[name] = pl.Series(name, signal.values(self.frame.graph, arrays),
                                     dtype=pl.Float64)
        return pl.col(name)

    def membership(self, target, resolved, values):
        """`is_in` over a column of nodes and over a column of values are two
        questions, and only the frame can tell them apart.

        A node column takes source keys, Keys and frames, all resolved against
        the graph. Anything else takes its values as they are."""
        path = path_of(target)
        nodes = (path is not None and len(path) == 1 and path[0] in self.frame.vars)
        if nodes or hasattr(values, "ids"):
            return resolved.is_in(self.frame.graph.ids_of(values))
        return resolved.is_in(list(values))

    # -- attaching and detaching --

    def wrap(self, data):
        """The frame a verb hands back, carrying whatever the resolution learned
        that holds for every row."""
        return self.frame._wrap(data, constants={**self.frame.constants,
                                                 **self.constants})

    def attach(self, data):
        added = list(self.keep.values()) + list(self.extra.values())
        return data.with_columns(added) if added else data

    def detach(self, data, keep=()):
        stale = [name for name in self.extra
                 if name in data.columns and name not in keep]
        return data.drop(stale) if stale else data

    # -- internals --

    def _present(self, name):
        return (name in self.frame._df.columns or name in self.extra
                or name in self.keep)

    def _values(self, name):
        """One column's values, materializing it first if it was only virtual."""
        if name in self.frame._df.columns:
            return self.frame._df[name].to_list()
        if name not in self.extra:
            self.lookup(tuple(name.split(".")))
        return self.extra[name].to_list()

    def _provenance(self, column, kind):
        if kind == TYPE:
            return self._types(column)
        name = shadow(kind, column)
        if self._present(name):
            return pl.col(name)
        root = column.partition(".")[0]
        if not self._present(column) and not self._present(root):
            # the provenance of nothing: answering 1.0 would be an answer
            raise ValueError(
                f"no column {column!r} to read the {kind} of -- this frame has: "
                f"{', '.join(self.frame.columns)}")
        # one value for every row -- as a column of them rather than a literal,
        # because a literal is one value in an aggregate too: `v.x.score.sum()`
        # over a group of three unweighted rows is 3.0, and `lit(1.0).sum()` is 1.0
        constant = self.frame.constants.get((kind, column))
        if constant is not None:
            return pl.repeat(constant, pl.len())
        # nothing measured this column, so nothing is in doubt about it
        if kind == SCORE:
            return pl.repeat(1.0, pl.len())
        return pl.repeat(None, pl.len(), dtype=pl.String)

    def _types(self, column):
        name = shadow(TYPE, column)
        if name not in self.extra:
            if column not in self.frame.vars:
                raise ValueError(f"{column!r} does not hold nodes, so it has no type")
            graph = self.frame.graph
            ids = self.frame._df[column].to_numpy()
            names = [graph.types[tag] for tag in graph._type_tag_of[ids].tolist()]
            self.extra[name] = pl.Series(name, names, dtype=pl.Enum(graph.types))
        return pl.col(name)

    def _below(self, var, rest):
        forced = None
        if rest[0] in (ATTR, REL) and len(rest) > 1:
            forced, rest = rest[0], rest[1:]
        name = ".".join(rest)
        if name == "label" and forced != REL:
            # the column a person reads, per the graph's `readable` map -- the
            # same one `labels()` names, so the two spellings cannot diverge
            return self._gathered(var, None)
        attribute = forced != REL and self.frame._has_attribute(var, name)
        relation = forced != ATTR and name in self.frame.graph.relations
        if attribute and relation and forced is None:
            raise ValueError(
                f"{var}.{name} is both an attribute of {self.frame.vars[var]} and a "
                f"relation of the graph. Say which: v.{var}.{ATTR}.{name} or "
                f"v.{var}.{REL}.{name}.")
        if attribute:
            return self._gathered(var, name)
        if relation:
            return Relation(var, name)
        raise ValueError(
            f"{var!r} has no attribute {name!r} and the graph has no relation by "
            f"that name. Attributes: {', '.join(self.frame._attribute_names(var))}. "
            f"Relations: {', '.join(self.frame.graph.relations)}.")

    def _gathered(self, var, attribute):
        name = f"{var}.{attribute if attribute is not None else 'label'}"
        if name not in self.extra:
            ids = self.frame._df[var].to_numpy()
            values, present = self.frame._gather(var, attribute, ids)
            series = pl.Series(name, values)
            if present is not None:
                series = pl.select(pl.when(pl.Series(present)).then(series)
                                   .otherwise(None).alias(name)).to_series()
            self.extra[name] = series
        return pl.col(name)


# what a group's confidence becomes: the rules `group_by(confidence=...)`
# takes, here because the planner folds a walk's slices by the same ones
FOLD = {
    "mean": lambda expr: expr.mean(),
    "min": lambda expr: expr.min(),
    "max": lambda expr: expr.max(),
    "product": lambda expr: expr.product(),
    # the mass of a set of walks: their probabilities, added
    "sum": lambda expr: expr.sum(),
    "first": lambda expr: expr.first(),
}




def _unit_queries(query):
    """One query vector or several, as unit rows."""
    values = np.asarray(query, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(lengths > 0, lengths, 1.0)


def _series(name, values):
    return values if isinstance(values, pl.Series) else pl.Series(
        name, values, dtype=pl.String if isinstance(values, list) else None)


def take(values, keep):
    """Index an array, a polars series or a python column, by mask or by
    position."""
    if isinstance(values, pl.Series):
        return (values.filter(pl.Series(keep)) if keep.dtype == bool
                else values.gather(keep))
    if not isinstance(values, list):
        return values[keep]
    if keep.dtype == bool:
        return [value for value, take in zip(values, keep.tolist()) if take]
    return [values[position] for position in keep.tolist()]


def _needles(needles):
    """One needle or several. A string is one thing and a vector is one thing;
    neither is a sequence of needles, however sequence-shaped it looks."""
    if isinstance(needles, (str, bytes)):
        return [needles]
    if isinstance(needles, np.ndarray):
        return [needles] if needles.ndim == 1 else list(needles)
    items = list(needles)
    if items and isinstance(items[0], (int, float, np.integer, np.floating)):
        return [items]                     # one vector, written out
    return items


def _nameable(needles):
    """Can these be the categories of an Enum? A query vector cannot."""
    return all(isinstance(one, str) for one in needles)
