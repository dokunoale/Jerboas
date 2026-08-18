"""The query: a dataframe linked to the graph.

`Graph` is the data structure and `Frame` is what you ask it. A Frame is a
polars DataFrame that remembers which of its columns are nodes of which graph,
and adds the four verbs a table cannot get from being a table:

    hop      one traversal: every edge leaving a column of nodes
    like     graded membership over a text column -- the search box
    attrs    a stored attribute as a column
    labels   the column a human reads, per the graph's `readable` map

Everything else is what it looks like: `filter` is polars' filter, `group_by`
polars' group_by, `sort` polars' sort. The forwarded verbs are listed one by one
rather than caught by `__getattr__`, so this class has an API instead of an
accident, and `.pl` hands back the DataFrame for anything not listed.

The frame is eager. Every verb here returns a materialized frame, which is what
makes `print(g.nodes("movie"))` a table rather than a plan -- and a hop has to
read its source ids anyway, so laziness would only defer the cheap half.
"""

import numpy as np
import polars as pl

from . import fuzzy, traverse
from .core import Signal
from .expr import Col, name_of
from .keys import Key

RELATION = "relation"


class Frame:
    """A table of node ids, joined to the graph that gave them meaning."""

    def __init__(self, graph, data, variables=None):
        self.graph = graph
        self._df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        # {column: type or None} for the columns that hold node ids, in the
        # order they were introduced. A hop with no `from_` reads the last one.
        self.vars = dict(variables or {})

    # --- construction --------------------------------------------------------

    def _wrap(self, data, variables=None):
        variables = self.vars if variables is None else variables
        kept = {name: type_ for name, type_ in variables.items() if name in data.columns}
        return Frame(self.graph, data, kept)

    # --- what it is ----------------------------------------------------------

    @property
    def pl(self):
        """The polars DataFrame, for anything this class does not forward."""
        return self._df

    def to_polars(self):
        return self._df

    def lazy(self):
        return self._df.lazy()

    def to_pandas(self):
        return self._df.to_pandas()

    def to_numpy(self):
        return self._df.to_numpy()

    def __dataframe__(self, *args, **kwargs):
        return self._df.__dataframe__(*args, **kwargs)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._df.to_numpy(), dtype=dtype)

    @property
    def columns(self):
        return self._df.columns

    @property
    def height(self):
        return self._df.height

    @property
    def shape(self):
        return self._df.shape

    @property
    def schema(self):
        return self._df.schema

    def is_empty(self):
        return self._df.is_empty()

    def __len__(self):
        return self._df.height

    def __iter__(self):
        return iter(self._df.iter_rows())

    def __getitem__(self, item):
        got = self._df[item]
        return self._wrap(got) if isinstance(got, pl.DataFrame) else got

    def __repr__(self):
        return repr(self._df)

    def __str__(self):
        return str(self._df)

    def _repr_html_(self):
        return self._df._repr_html_()

    def rows(self, named=False):
        return self._df.rows(named=named)

    # --- the graph verbs -----------------------------------------------------

    def hop(self, relation=None, *, to=None, type=None, reverse=None, as_=None,
            from_=None, norm=False):
        """Every edge leaving a column of nodes, one result row per edge.

            .hop("has_genre", to="genre")        # forwards along one relation
            .hop("directed_by", to="person", reverse=True)
            .hop(to="mid")                       # wildcard: any relation, both ways

        A named relation reads forwards unless `reverse=True`; the wildcard has
        no natural direction and walks both, which is what closes a bridge
        pattern without the store holding every edge twice.

        The step's own columns come back beside the target: `<as_>.score` always,
        and `<as_>.rel` when the relation was a wildcard and therefore worth
        naming. `as_` defaults to the relation's name, or to the target's.
        """
        source = self._source_var(from_)
        target = to or type
        if target is None:
            raise TypeError("hop(...) needs `to=` (or `type=`): the new column's name")
        if reverse is None and relation is not None:
            reverse = False                      # a named relation reads forwards

        nodes = self._df[source].to_numpy()
        rows, targets, codes, weights = traverse.expand(
            self.graph, nodes, relation, reverse, normalized=norm)

        if type is not None:
            low, high = self.graph.block(type)
            keep = (targets >= low) & (targets < high)
            rows, targets, codes, weights = rows[keep], targets[keep], codes[keep], weights[keep]

        edge = as_ or relation or target
        data = self._df[rows] if len(rows) else self._df.clear()
        added = [pl.Series(self._free(target), targets.astype(np.int32)),
                 pl.Series(self._free(f"{edge}.score"), weights)]
        if relation is None:
            added.append(pl.Series(self._free(f"{edge}.rel"), self._names(codes)))
        variables = dict(self.vars)
        variables[target] = type
        return self._wrap(data.with_columns(added), variables)

    def like(self, column, needles, k=1, cutoff=0.6):
        """Keep the k rows closest to each needle, and say how close they were.

            g.nodes("person").labels("person").like(v.person.label, ["tarantino"])

        A `similarity` column comes back with them, in [0, 1]: the same measure
        that decided admission, so the rows kept are exactly the ones a ranking
        would have put on top. It is an ordinary column -- add it to a score, or
        ignore it."""
        name = name_of(column)
        frame = self if name in self._df.columns else self._materialize(name)
        values = frame._df[name].to_list()
        texts = [(row, str(value).lower()) for row, value in enumerate(values)
                 if value is not None]
        if isinstance(needles, (str, bytes)):
            needles = [needles]
        found = fuzzy.best(list(needles), texts, k, cutoff)
        # best first, and among equals the frame's own order: a tie between two
        # exact matches is not a ranking, so it should not look like one
        order = sorted(found, key=lambda row: (-found[row], row))
        data = frame._df[order].with_columns(
            pl.Series(frame._free("similarity"), [found[row] for row in order], dtype=pl.Float64))
        return frame._wrap(data)

    def attrs(self, *paths, **named):
        """Stored attributes as columns, named `<var>.<attribute>`.

            .attrs("rec")                  every attribute of rec's type
            .attrs("rec.title")            one of them
            .attrs(rec="title")            the same
            .attrs(rec=["title", "year"])  several
        """
        wanted = []
        for path in paths:
            name = name_of(path)
            var, _, attribute = name.partition(".")
            wanted.extend((var, one) for one in
                          ([attribute] if attribute else self._attributes(var)))
        for var, value in named.items():
            names = [value] if isinstance(value, str) else list(value)
            wanted.extend((var, name_of(one)) for one in names)

        frame = self
        for var, attribute in wanted:
            frame = frame._attach(var, attribute, f"{var}.{attribute}")
        return frame

    def labels(self, *variables):
        """The column a person reads, as `<var>.label`.

        Which column that is comes from the graph's `readable` map, because it is
        a fact about the dataset and not about one query. A type that declares
        none falls back to its `label` column -- its identity, never a guess at
        which of its attributes happens to be a name."""
        frame = self
        for one in variables:
            var = name_of(one)
            frame = frame._attach(var, None, f"{var}.label")
        return frame

    def keys(self, variable=None):
        """One column as Key values -- what a caller outside the frame names a
        node with."""
        graph = self.graph
        return [Key(graph, int(index)) for index in self.ids(variable)]

    def ids(self, variable=None):
        """One column as an int array. Defaults to the first node column, which
        is what makes a frame usable as a seed set anywhere ids are."""
        return self._df[self._var(variable)].to_numpy()

    # --- the forwarded verbs -------------------------------------------------

    def filter(self, *predicates, **named):
        return self._wrap(self._df.filter(*_exprs(predicates), **_exprs_dict(named)))

    def with_columns(self, *exprs, **named):
        resolved = {name: self._resolve(name, value) for name, value in named.items()}
        return self._wrap(self._df.with_columns(*_exprs(exprs), **resolved))

    def select(self, *exprs, **named):
        return self._wrap(self._df.select(*_exprs(exprs), **_exprs_dict(named)))

    def sort(self, *by, descending=False, nulls_last=True):
        return self._wrap(self._df.sort(*_exprs(by), descending=descending,
                                        nulls_last=nulls_last))

    def unique(self, subset=None, keep="first", maintain_order=True):
        columns = None if subset is None else [name_of(one) for one in _listed(subset)]
        return self._wrap(self._df.unique(subset=columns, keep=keep,
                                          maintain_order=maintain_order))

    def head(self, n):
        return self._wrap(self._df.head(n))

    limit = head

    def tail(self, n):
        return self._wrap(self._df.tail(n))

    def drop(self, *names):
        return self._wrap(self._df.drop([name_of(one) for one in _flat(names)]))

    def rename(self, mapping):
        variables = {mapping.get(name, name): type_ for name, type_ in self.vars.items()}
        return self._wrap(self._df.rename(mapping), variables)

    def join(self, other, on=None, how="inner", **kwargs):
        right = other._df if isinstance(other, Frame) else other
        columns = None if on is None else [name_of(one) for one in _listed(on)]
        variables = dict(self.vars)
        if isinstance(other, Frame):
            variables.update(other.vars)
        return self._wrap(self._df.join(right, on=columns, how=how, **kwargs), variables)

    def group_by(self, *by, maintain_order=True):
        return _GroupBy(self, [name_of(one) for one in _flat(by)], maintain_order)

    def top(self, n, by=None, descending=True):
        """The n best rows. `by` defaults to a `score` column when there is one,
        because that is what the frame was building up to."""
        if by is None:
            by = "score" if "score" in self._df.columns else None
        if by is None:
            return self.head(n)
        return self.sort(by, descending=descending).head(n)

    # --- internals -----------------------------------------------------------

    def _resolve(self, name, value):
        """A Signal is a strategy that has not met a graph yet; here it does."""
        if isinstance(value, Signal):
            arrays = tuple(self._df[column].to_numpy() for column in value.columns)
            return pl.Series(name, value.values(self.graph, arrays), dtype=pl.Float64)
        return _expr(value)

    def _var(self, variable=None):
        if variable is not None:
            return name_of(variable)
        if not self.vars:
            raise ValueError("this frame has no node column")
        return next(iter(self.vars))

    def _source_var(self, from_):
        if from_ is not None:
            return name_of(from_)
        if not self.vars:
            raise ValueError("hop(...) needs a column of nodes to leave from")
        return list(self.vars)[-1]          # the last one introduced

    def _free(self, name):
        """`name`, suffixed if the frame already has a column called that."""
        if name not in self._df.columns:
            return name
        suffix = 2
        while f"{name}_{suffix}" in self._df.columns:
            suffix += 1
        return f"{name}_{suffix}"

    def _names(self, codes):
        """Relation codes as names, with `~` for an edge walked backwards."""
        relations = self.graph.relations
        return [f"~{relations[~code]}" if code < 0 else relations[code]
                for code in codes.tolist()]

    def _attributes(self, var):
        type_ = self.vars.get(self._var(var))
        if type_ is None:
            raise ValueError(
                f"{var!r} holds nodes of no single type, so it has no set of "
                f"attributes to expand: name the one you want, `{var}.title`")
        return [name for name in self.graph.columns[type_]]

    def _materialize(self, name):
        """The frame with column `name` present: an attribute path resolves
        itself, so `like(v.person.label, ...)` needs no separate step."""
        var, _, attribute = name.partition(".")
        if var not in self.vars:
            raise ValueError(f"no column {name!r}, and {var!r} is not a node column")
        return self.labels(var) if attribute == "label" else self.attrs(**{var: attribute})

    def _attach(self, var, attribute, out):
        if out in self._df.columns:
            return self
        var = self._var(var)
        ids = self._df[var].to_numpy()
        values, present = self._gather(var, attribute, ids)
        data = self._df.with_columns(pl.Series(out, values))
        if present is not None:
            data = data.with_columns(
                pl.when(pl.Series(present)).then(pl.col(out)).otherwise(None).alias(out))
        return self._wrap(data)

    def _gather(self, var, attribute, ids):
        """One attribute for a column of nodes. A gather, not a join: an id is a
        position, so the value is `column.values[id - block_start]`."""
        graph = self.graph
        type_ = self.vars.get(var)
        if type_ is not None:
            return _read(graph, type_, self._named(type_, attribute), ids)
        # an untyped column: read each type's block against its own table, and
        # let the types that have no such attribute stay absent
        values, present = [None] * len(ids), np.zeros(len(ids), dtype=bool)
        tags = graph._type_tag_of[ids]
        for tag in np.unique(tags):
            rows = np.flatnonzero(tags == tag)
            block = graph.types[int(tag)]
            column = graph.column(block, self._named(block, attribute))
            if column is None:
                continue
            part, mask = _read(graph, block, self._named(block, attribute), ids[rows])
            for position, row in enumerate(rows.tolist()):
                if mask is None or mask[position]:
                    values[row] = part[position]
                    present[row] = True
        return values, present

    def _named(self, type_, attribute):
        """Which column an attribute means for one type: itself, or -- for the
        readable label -- whatever that type declared."""
        if attribute is not None:
            return attribute
        return self.graph.readable.get(type_, "label")


def _read(graph, type_, name, ids):
    column = graph.column(type_, name)
    if column is None:
        available = ", ".join(sorted(graph.columns.get(type_, {})))
        raise ValueError(f"{type_} has no attribute {name!r}; it has: {available}")
    local = ids - graph.block(type_)[0]
    values = column.values[local]
    if values.dtype == object:
        values = values.tolist()
    present = None if column.present is None else column.present[local]
    return values, present


class _GroupBy:
    """What `group_by` returns: only `agg` follows it, so this is all of it."""

    def __init__(self, frame, by, maintain_order):
        self.frame = frame
        self.by = by
        self.maintain_order = maintain_order

    def agg(self, *exprs, **named):
        grouped = self.frame._df.group_by(self.by, maintain_order=self.maintain_order)
        return self.frame._wrap(grouped.agg(*_exprs(exprs), **_exprs_dict(named)))

    def len(self, name="len"):
        grouped = self.frame._df.group_by(self.by, maintain_order=self.maintain_order)
        return self.frame._wrap(grouped.len(name=name))


def concat(*frames, how="diagonal"):
    """Several frames as one -- the disjunction a single pattern cannot express.

    `diagonal` by default: two patterns reaching the same nodes by different
    routes do not have the same columns, and a null in the column one of them
    lacks is exactly what "reached the other way" means."""
    frames = list(_flat(frames))
    if not frames:
        raise ValueError("concat() needs at least one frame")
    variables = {}
    for frame in frames:
        variables.update(frame.vars)
    data = pl.concat([frame.pl for frame in frames], how=how)
    return Frame(frames[0].graph, data, variables)


def _expr(item):
    """A Col as the expression it stands for, and everything else untouched.

    Only Col is translated on purpose. polars already reads a bare string as a
    column name wherever these verbs take one, and reading it that way *here*
    too would turn `filter(title="Alpha")` -- where the string is a value -- into
    a comparison against a column called Alpha."""
    return item.expr if isinstance(item, Col) else item


def _exprs(items):
    return [_expr(item) for item in _flat(items)]


def _exprs_dict(named):
    return {name: _expr(value) for name, value in named.items()}


def _listed(thing):
    return [thing] if isinstance(thing, (str, Col)) else list(thing)


def _flat(items):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flat(item)
        else:
            yield item
