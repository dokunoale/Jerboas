"""What a hop defers, and what turns a name into a column.

Two pieces the Frame leans on, neither of which is about being a dataframe.

`Pending` is a hop that has walked the graph but not built its rows. The walk
produced four parallel arrays, and turning them into a frame means taking every
column the old frame had at those positions -- the expensive half of a hop. Held
one step, a filter about the node just reached can be applied to the arrays
instead of to rows that would then be dropped.

`Resolver` is where a name meets the graph. Four answers, in order: the
provenance of a column (its confidence, the relation that revealed it, the type
its ids fall in), a column the frame already has, an attribute of that
variable's type, or a relation of the graph. Whatever is read only to decide
something is taken off again; whatever was *measured* stays, as the column's
confidence.
"""

import numpy as np
import polars as pl

from . import fuzzy
from .expr import (ATTR, PROVENANCE, REL, SCORE, TYPE, VIA, Relation, path_of,
                   shadow)


class Pending:
    """A hop that has walked the graph but not yet built its rows.

    The walk produced four parallel arrays; turning them into a frame means
    taking every column the old frame had at `rows`, which is the expensive half
    of a hop. Holding them one step lets a filter about the node just reached be
    applied here -- to arrays -- instead of to rows that would then be dropped.

    Nothing else is deferred: any other verb reads `_df` and pays for it. This
    is a peephole, not a query planner."""

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

    def like(self, column, needles, k, cutoff):
        """The k closest rows to each needle, and the closeness kept as the
        column's confidence -- one computation, so admission and weight cannot
        disagree."""
        name = str(column)
        values = self._values(name)
        texts = [(row, str(value).lower()) for row, value in enumerate(values)
                 if value is not None]
        found = fuzzy.best(needles, texts, k, cutoff)
        closeness = np.zeros(len(values))
        for row, score in found.items():
            closeness[row] = score
        self.keep[shadow(SCORE, name)] = pl.Series(shadow(SCORE, name), closeness)
        admitted = f"{name}.__like__"
        self.extra[admitted] = pl.Series(admitted, closeness > 0)
        return pl.col(admitted)

    def near(self, column, query, k, cutoff):
        """The rows nearest a query vector, and the cosine kept as the column's
        confidence.

        Scored as one matmul against the type's unit block rather than a
        distance per row, which is what makes an exact search worth having
        before an approximate one: a few hundred thousand rows are
        milliseconds, and there is nothing to be wrong about."""
        var, _, name = str(column).partition(".")
        if var not in self.frame.vars:
            raise ValueError(f"{var!r} is not a column of nodes, so it has no vectors")
        type_ = self.frame.vars[var]
        if type_ is None:
            raise ValueError(
                f"{var!r} holds nodes of no single type, so which type's {name!r} "
                f"vectors to read is a question: narrow it with v.{var}.type first.")
        block = self.frame.graph.unit(type_, name)
        if block is None:
            raise ValueError(
                f"{type_} has no vector column {name!r}; it has: "
                f"{', '.join(sorted(self.frame.graph.vectors.get(type_, {}))) or 'none'}")

        low = self.frame.graph.block(type_)[0]
        rows = self.frame._df[var].to_numpy() - low
        queries = _unit_queries(query)
        # the best any query vector says of each row: several needles are
        # several questions, and a row answers whichever it answers best
        similarity = (block[rows] @ queries.T).max(axis=1)
        similarity = np.clip(similarity, 0.0, 1.0)

        admitted = similarity >= cutoff
        if k is not None and admitted.sum() > k:
            best = np.argpartition(-np.where(admitted, similarity, -1.0), k)[:k]
            keep = np.zeros(len(similarity), dtype=bool)
            keep[best] = True
            admitted &= keep
        name_of_shadow = shadow(SCORE, str(column))
        self.keep[name_of_shadow] = pl.Series(name_of_shadow,
                                              np.where(admitted, similarity, 0.0))
        flag = f"{column}.__near__"
        self.extra[flag] = pl.Series(flag, admitted)
        return pl.col(flag)

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
        the graph. Anything else takes its values as they are -- resolving
        "Alpha" as a node key on a column of titles admitted nothing, which is
        the worst way to be wrong."""
        path = path_of(target)
        nodes = (path is not None and len(path) == 1 and path[0] in self.frame.vars)
        if nodes or hasattr(values, "ids"):
            return resolved.is_in(self.frame.graph.ids_of(values))
        return resolved.is_in(list(values))

    # -- attaching and detaching --

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
        if kind == VIA and column in self.frame.via:
            return pl.lit(self.frame.via[column])     # one relation, every row
        # nothing measured this column, so nothing is in doubt about it
        return pl.lit(1.0) if kind == SCORE else pl.lit(None, dtype=pl.String)

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


FOLD = {
    "mean": lambda expr: expr.mean(),
    "min": lambda expr: expr.min(),
    "max": lambda expr: expr.max(),
    "product": lambda expr: expr.product(),
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
    """Index an array or a python column, by mask or by position."""
    if not isinstance(values, list):
        return values[keep]
    if keep.dtype == bool:
        return [value for value, take in zip(values, keep.tolist()) if take]
    return [values[position] for position in keep.tolist()]


class Plan:
    """A walk described but not taken, and the conditions about where it lands.

    Held only inside `optimize`. Executing it runs the same eager hop and the
    same filters, a batch of source rows at a time, and stacks the results --
    so the answer is what it would have been and the peak is one batch's
    expansion rather than the whole walk's.
    """

    __slots__ = ("base", "variables", "via", "steps", "predicates", "batch")

    def __init__(self, base, variables, via, steps, batch, predicates=()):
        self.base = base                # the frame before the walk
        self.variables = variables      # what the base's columns hold
        self.via = via
        self.steps = steps              # [(relation spec, column or None)]
        self.predicates = list(predicates)   # conditions about where it lands
        self.batch = batch

    def narrowed(self, predicates):
        """The same plan, with more said about where the walk may land."""
        return Plan(self.base, self.variables, self.via, self.steps, self.batch,
                    self.predicates + list(predicates))

    def build(self, frame_of):
        """Run it. `frame_of` makes a Frame of one slice of the base, which is
        what keeps every rule about hopping and filtering in one place rather
        than in two."""
        through = [spec for spec, name in self.steps if name is None]
        named = {name: spec for spec, name in self.steps if name is not None}
        parts = []
        for start in range(0, max(self.base.height, 1), self.batch):
            slice_ = self.base.slice(start, self.batch)
            if not slice_.height:
                continue
            part = frame_of(slice_).hop(*through, **named)
            if self.predicates:
                part = part.filter(*self.predicates)
            parts.append(part.raw)
        if not parts:                      # an empty base still has a schema
            return frame_of(self.base.clear()).hop(*through, **named).raw
        return parts[0] if len(parts) == 1 else pl.concat(parts, how="vertical")
