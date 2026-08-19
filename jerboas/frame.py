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
from .expr import (ATTR, PROVENANCE, REL, SCORE, TYPE, VIA, Col, Expr, Relation,
                   direction, name_of, reverse)
from .keys import Key

RELATION = "relation"

# Every column carries a confidence, and where something measured one it is kept
# here: an ordinary polars column under a reserved prefix, so filter, sort, join
# and group_by keep it aligned with its values without a line of code from us.
# Hidden from `columns` and from `print`, because it is an attribute of a column
# rather than a column -- `v.rec.score` is how it is read.
#
# Absent means 1.0. A graph with no weights and a query with no fuzzy matching
# therefore allocate nothing at all.
SHADOW = "__jb_"


def shadow(kind, column):
    return f"{SHADOW}{kind}__{column}"


def is_shadow(name):
    return name.startswith(SHADOW)


def shadowed(name):
    """(kind, column) of a shadow column, or None."""
    if not is_shadow(name):
        return None
    kind, _, column = name[len(SHADOW):].partition("__")
    return kind, column


class Frame:
    """A table of node ids, joined to the graph that gave them meaning."""

    def __init__(self, graph, data, variables=None, pending=None, via=None):
        self.graph = graph
        self._data = (None if data is None else
                      data if isinstance(data, pl.DataFrame) else pl.DataFrame(data))
        # {column: type or None} for the columns that hold node ids, in the
        # order they were introduced. A hop with no `from_` reads the last one.
        self.vars = dict(variables or {})
        # {column: relation} for the steps that walked one relation for every
        # row. A constant needs no array, but `v.x.via` should still answer
        self.via = dict(via or {})
        # a hop that has walked but not built its rows yet (see _Pending). The
        # only thing that reads it is `filter`, which may be able to apply
        # itself to the arrays instead of to the rows they would become
        self._pending = pending

    @property
    def _df(self):
        """The rows. Building them is what a pending hop was deferring, so
        anything that touches this pays for it -- and `filter` gets first
        refusal."""
        if self._pending is not None:
            self._data = self._pending.build()
            self._pending = None
        return self._data

    # --- construction --------------------------------------------------------

    def _wrap(self, data, variables=None):
        variables = self.vars if variables is None else variables
        kept = {name: type_ for name, type_ in variables.items() if name in data.columns}
        via = {name: relation for name, relation in self.via.items()
               if name in data.columns}
        return Frame(self.graph, data, kept, via=via)

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
        """What the frame holds, as a person reads it. A column's confidence and
        provenance are attributes of it (`v.rec.score`), not entries here."""
        return [name for name in self._df.columns if not is_shadow(name)]

    @property
    def hidden(self):
        """The shadow columns, for a caller who wants to see the bookkeeping."""
        return [name for name in self._df.columns if is_shadow(name)]

    @property
    def visible(self):
        """The polars frame without the shadows -- what `print` shows."""
        return self._df.select(self.columns)

    @property
    def height(self):
        return self._df.height

    @property
    def shape(self):
        return self._df.height, len(self.columns)

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
        return repr(self.visible)

    def __str__(self):
        return str(self.visible)

    def _repr_html_(self):
        return self.visible._repr_html_()

    def rows(self, named=False):
        return self._df.rows(named=named)

    # --- the graph verbs -----------------------------------------------------

    def hop(self, *through, **named):
        """Walk the graph. Every argument is one step, in order.

            .hop(genre="has_genre")            # one step, kept as `genre`
            .hop(person="~directed_by")        # the same relation, backwards
            .hop((), rec="has_genre")          # any relation, then has_genre
            .hop(step=("has_genre", "~directed_by"))   # either, at this step

        A **keyword** names the column the step's arrivals are kept in. A
        **positional** step is walked and not kept -- which is what lets it be
        folded: two routes that meet at an unnamed intermediate carry identical
        rows onward, and only one of them is worth continuing. That fold is the
        old engine's memoized sub-path search, and here it is a consequence of
        not having given something a name rather than a parameter.

        The last step must be named, because where the walk ends is what the
        frame holds. Python already requires the positional ones to come first,
        so the rule costs nothing to obey.

        A step is a relation name, `~name` for the same relation read backwards
        -- the spelling `v.x.via` prints -- or a collection of them for "any of
        these". The empty collection is "any relation at all", in either
        direction, which is what closes a bridge without the store holding every
        edge twice.

        Walking leaves from the rightmost column of nodes. To leave from another,
        `select` it and `join` the result back: that is what working on a
        dataframe is for.

        Only the named columns are added. What the step measured is each one's
        confidence (`v.genre.score`) and which relation it walked is
        `v.genre.via` -- attributes of a column rather than columns, costing
        nothing when they say the same thing about every row.
        """
        steps = [(spec, None) for spec in through]
        steps += [(spec, name) for name, spec in named.items()]
        if not named:
            raise ValueError(
                "the last step of a hop must be named: where the walk ends is "
                "what the frame holds. hop(..., rec=\"has_genre\")")
        graph = self.graph
        base = self._df
        source = self._rightmost()
        for _spec, name in steps:
            if name is not None:
                self._claim(name)

        rows = np.arange(base.height, dtype=np.int64)
        nodes = base[source].to_numpy()
        added, variables, via = {}, dict(self.vars), dict(self.via)

        for spec, name in steps:
            walked, targets, codes, weights, single = _walk(graph, nodes, spec)
            rows = rows[walked]
            added = {column: _take(values, walked) for column, values in added.items()}
            nodes = targets
            if name is None:
                # nothing names these, so nothing tells two routes through them
                # apart: keep one of each and carry that forward
                keep = _distinct(graph, rows, nodes)
                rows, nodes = rows[keep], nodes[keep]
                added = {column: _take(values, keep) for column, values in added.items()}
                continue
            added[name] = nodes.astype(np.int32)
            if len(weights) and not (weights == 1.0).all():
                added[shadow(SCORE, name)] = weights
            if single is None and len(codes):
                added[shadow(VIA, name)] = self._names(codes)
            elif single is not None:
                via[name] = single           # every row walked the same relation
            variables[name] = self._one_type(nodes)

        if any(name is None for _spec, name in steps):
            # a row that differs only where nothing was named is not a different
            # row: two routes through an unnamed step are one answer
            keep = _folded(rows, [added[name] for _spec, name in steps if name])
            rows = rows[keep]
            added = {column: _take(values, keep) for column, values in added.items()}

        last = steps[-1][1]
        return Frame(graph, None, variables, pending=_Pending(base, rows, added, last),
                     via=via)

    def _rightmost(self):
        """The column of nodes furthest to the right -- where a walk leaves from.

        By the frame's own column order rather than by when a variable was
        introduced, because that is the one a person reads off the print."""
        for name in reversed(self._df.columns):
            if name in self.vars:
                return name
        raise ValueError("hop(...) needs a column of nodes to leave from")

    def _one_type(self, targets):
        """The type these nodes are, when they are all of one -- read off the
        data rather than declared, so it cannot be declared wrongly."""
        if not len(targets):
            return None
        tags = np.unique(self.graph._type_tag_of[targets])
        return self.graph.types[int(tags[0])] if len(tags) == 1 else None

    def _reachable(self, relation, where, reverse=None):
        """Which nodes of the graph have such an edge, as one boolean array.

        With no `where` that is the node's arity. With one it is read from the
        other end -- what has an edge into this set is what this set reaches
        when walked the other way -- so the cost is the degree of `where` and
        the frame is never expanded.

        Both directions unless told otherwise: `v.movie.directed_by.is_in(who)`
        asks whether the two are joined by that relation, and a condition has
        nowhere to say which way it is stored."""
        if where is None:
            return self._degrees(relation, reverse) > 0
        back = None if reverse is None else not reverse
        _rows, targets, _codes, _weights = traverse.expand(
            self.graph, self.graph.ids_of(where), relation, back)
        admissible = np.zeros(self.graph.n_nodes, dtype=bool)
        admissible[targets] = True
        return admissible

    def _has_attribute(self, var, name):
        """Does this variable's type carry such a column? An untyped variable is
        asked of every type, the way reading one is."""
        graph, type_ = self.graph, self.vars.get(var)
        types = [type_] if type_ is not None else graph.types
        return any(graph.column(one, name) is not None for one in types)

    def _attribute_names(self, var):
        type_ = self.vars.get(var)
        if type_ is not None:
            return sorted(self.graph.columns.get(type_, {}))
        names = set()
        for one in self.graph.types:
            names |= set(self.graph.columns.get(one, {}))
        return sorted(names)

    def _degrees(self, relation, reverse=None):
        """Per-node arity. Both directions unless told otherwise: how many edges
        of that relation a node is part of, which is what `.count()` on a
        relation asks and the only reading a condition can mean."""
        graph = self.graph
        if reverse is None:
            return graph.degree(relation, False) + graph.degree(relation, True)
        return graph.degree(relation, reverse)

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
        """Keep the rows a predicate admits.

        Two things happen here that a dataframe's filter does not do. The
        predicate arrives unresolved, so a name in it can still turn out to be
        an attribute the frame has not read yet or a relation of the graph. And
        if the frame is a hop that has walked but not built its rows, a
        predicate about the node it just reached is applied to the arrays
        instead -- so the rows it would have dropped are never built."""
        frame, predicates = self, list(predicates)
        if frame._pending is not None and predicates:
            frame, predicates = frame._pushdown(predicates)
        if not predicates and not named:
            return frame
        resolver = _Resolver(frame)
        exprs = [_resolved(one, resolver) for one in _flat(predicates)]
        data = resolver.attach(frame._df).filter(*exprs, **_exprs_dict(named))
        return frame._wrap(resolver.detach(data))

    def with_columns(self, *exprs, **named):
        resolver = _Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        data = resolver.attach(self._df).with_columns(*columns, **keyed)
        # what was asked for stays; what was only read to compute it does not
        asked = set(named) | {one.meta.output_name() for one in columns
                              if isinstance(one, pl.Expr)}
        return self._wrap(resolver.detach(data, keep=asked))

    def select(self, *exprs, **named):
        """Choose columns. A column's confidence and provenance travel with it,
        because they are attributes of it and not columns of their own."""
        resolver = _Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        data = resolver.attach(self._df).select(*columns, **keyed)
        carried = [name for name in self._df.columns
                   if is_shadow(name) and shadowed(name)[1] in data.columns
                   and name not in data.columns]
        if carried:
            data = data.hstack(resolver.attach(self._df).select(carried))
        return self._wrap(data)

    def sort(self, *by, descending=False, nulls_last=True):
        resolver = _Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(by)]
        data = resolver.attach(self._df).sort(*columns, descending=descending,
                                              nulls_last=nulls_last)
        return self._wrap(resolver.detach(data))

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
        """Drop columns, and with each one the shadows that belong to it."""
        going = [name_of(one) for one in _flat(names)]
        going += [name for name in self._df.columns
                  if is_shadow(name) and shadowed(name)[1] in going]
        return self._wrap(self._df.drop(going))

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

    def top(self, n, by=None, descending=True, over=None):
        """The n best rows. `by` defaults to a `score` column when there is one,
        because that is what the frame was building up to.

        `over` makes it the n best *per group* -- one user's ten films, one
        walk's five most promising steps -- which is a window function rather
        than a sort, so it costs one pass instead of one query per group:

            .top(10, by="score", over="user")
            .hop(to="mid").top(5, by=v.pr, over="seed").hop(to="rec")

        That second line is a beam search: keep the k most promising partial
        walks at each step and expand only those. It was an engine once."""
        if by is None:
            by = "score" if "score" in self._df.columns else None
        if by is None:
            return self.head(n) if over is None else self
        resolver = _Resolver(self)
        ordering = _resolved(by, resolver)
        if isinstance(ordering, str):          # polars reads a bare name as a column
            ordering = pl.col(ordering)
        data = resolver.attach(self._df)
        if over is None:
            data = data.sort(ordering, descending=descending).head(n)
            return self._wrap(resolver.detach(data))
        groups = [name_of(one) for one in _flat([over])]
        ranked = ordering.rank("ordinal", descending=descending).over(groups)
        data = (data.filter(ranked <= n)
                .sort(groups + [ordering], descending=[False] * len(groups) + [descending]))
        return self._wrap(resolver.detach(data))

    # --- internals -----------------------------------------------------------

    # --- pushing a predicate into a hop that has not built its rows ----------

    def _pushdown(self, predicates):
        """Apply what can be applied to the arrays, and hand back the rest.

        A predicate is pushable when everything it reads is about the node the
        hop just reached, or about the step itself -- which is exactly the class
        of constraint the old compiler folded into an admission mask."""
        pending = self._pending
        stays, pushed = [], []
        for one in predicates:
            paths = one.reads() if isinstance(one, Expr) else _root_paths(one)
            if paths and all(path[0] == pending.var for path in paths):
                pushed.append(one)
            else:
                stays.append(one)
        if not pushed:
            return self, stays

        narrow = Frame(self.graph, pending.narrow(),
                       {pending.var: self.vars.get(pending.var)})
        resolver = _Resolver(narrow)
        exprs = [_resolved(one, resolver) for one in pushed]
        keep = (resolver.attach(narrow._df)
                .select(_all(exprs).fill_null(False).alias("keep"))["keep"].to_numpy())
        return Frame(self.graph, None, self.vars, pending=pending.masked(keep)), stays

    def _var(self, variable=None):
        if variable is not None:
            return name_of(variable)
        if not self.vars:
            raise ValueError("this frame has no node column")
        return next(iter(self.vars))

    def _free(self, name):
        """`name`, suffixed if the frame already has a column called that. Used
        only where a repeat is harmless -- a second `similarity` is a second
        search, not a second reading of the same thing."""
        if name not in self._df.columns:
            return name
        suffix = 2
        while f"{name}_{suffix}" in self._df.columns:
            suffix += 1
        return f"{name}_{suffix}"

    def _claim(self, target):
        """Refuse a hop that would land on a column already there.

        Suffixing it to `tag_2` is what a dataframe would do, and it is wrong
        here: the two are different steps, and a filter written against the
        obvious name would silently read the other one."""
        if target in self._df.columns:
            raise ValueError(
                f"this hop would land on {target!r}, which the frame already has. "
                f"Give the new column its own name: hop(<name>=<relation>).")

    def _names(self, codes):
        """Relation codes as names, with `~` for an edge walked backwards."""
        relations = self.graph.relations
        return [reverse(relations[~code]) if code < 0 else relations[code]
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


class _Pending:
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
        return _Pending(self.base, self.rows[keep],
                        {name: _take(values, keep) for name, values in self.added.items()},
                        self.var)

    def build(self):
        data = self.base[self.rows] if len(self.rows) else self.base.clear()
        return data.with_columns([_series(name, values)
                                  for name, values in self.added.items()])


class _Resolver:
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

    def signal(self, signal):
        """A strategy's score, as a literal column the rest of an expression can
        be arithmetic on."""
        name = f"_signal_{len(self.extra)}"
        arrays = tuple(self.frame._df[column].to_numpy() for column in signal.columns)
        self.extra[name] = pl.Series(name, signal.values(self.frame.graph, arrays),
                                     dtype=pl.Float64)
        return pl.col(name)

    def ids(self, values):
        return self.frame.graph.ids_of(values)

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


class _GroupBy:
    """What `group_by` returns: only `agg` follows it, so this is all of it."""

    def __init__(self, frame, by, maintain_order):
        self.frame = frame
        self.by = by
        self.maintain_order = maintain_order

    def agg(self, *exprs, **named):
        resolver = _Resolver(self.frame)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        grouped = (resolver.attach(self.frame._df)
                   .group_by(self.by, maintain_order=self.maintain_order))
        return self.frame._wrap(resolver.detach(grouped.agg(*columns, **keyed)))

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
    return Frame(frames[0].graph, pl.concat(_aligned(frames), how=how), variables)


def _aligned(frames):
    """The frames with their empty columns typed like the ones that have rows.

    A branch that matched nothing still has to line up with one that did, and an
    empty column of unknown type is polars' Null -- which refuses to stack onto
    a String. So a column that is Null in one frame takes the type another frame
    gives it."""
    known = {}
    for frame in frames:
        for name, dtype in frame.pl.schema.items():
            if dtype != pl.Null and name not in known:
                known[name] = dtype
    out = []
    for frame in frames:
        recast = [pl.col(name).cast(known[name]) for name, dtype in frame.pl.schema.items()
                  if dtype == pl.Null and name in known]
        out.append(frame.pl.with_columns(recast) if recast else frame.pl)
    return out


def _walk(graph, nodes, spec):
    """One step, over whatever relations it names.

    Returns the four arrays a traversal produces plus, when every row walked the
    same relation the same way, the name of it -- a constant that needs no
    column to say so."""
    specs = _relations(spec)
    if specs is None:                                   # any relation, either way
        return traverse.expand(graph, nodes, None, None, normalized=True) + (None,)
    parts = [traverse.expand(graph, nodes, name, backwards, normalized=True)
             for name, backwards in specs]
    single = None
    if len(specs) == 1:
        name, backwards = specs[0]
        single = reverse(name) if backwards else name
    if len(parts) == 1:
        return parts[0] + (single,)
    return tuple(np.concatenate(column) for column in zip(*parts)) + (single,)


def _relations(spec):
    """A step's relations as [(name, reverse)], or None for the wildcard.

    An empty collection is the wildcard: no constraint on the relation is the
    empty set of constraints, which is also why there is no magic string."""
    if isinstance(spec, str):
        spec = (spec,)
    specs = tuple(spec)
    if not specs:
        return None
    return [direction(name) for name in specs]


def _distinct(graph, rows, nodes):
    """The first of each (row, node) pair, in order -- the fold between steps."""
    if not len(rows):
        return np.zeros(0, dtype=np.int64)
    key = rows * graph.n_nodes + nodes
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def _folded(rows, columns):
    """The first row of each distinct (source row, named columns) combination."""
    if not len(rows):
        return np.zeros(0, dtype=np.int64)
    stacked = np.stack([rows] + [np.asarray(column, dtype=np.int64)
                                 for column in columns])
    _, first = np.unique(stacked, axis=1, return_index=True)
    return np.sort(first)


def _resolved(item, resolver):
    """One thing a verb was given, as a polars expression. A jerboas Expr is
    resolved against the frame; everything else is already polars', or a plain
    value polars reads as a literal."""
    if isinstance(item, Expr):
        return item.resolve(resolver)
    return item


def _root_paths(item):
    """The name paths a raw polars expression reads, so one can be pushed into a
    hop as readily as a jerboas expression."""
    if isinstance(item, pl.Expr):
        return tuple(tuple(name.split(".")) for name in item.meta.root_names())
    return ()


def _all(exprs):
    combined = exprs[0]
    for one in exprs[1:]:
        combined = combined & one
    return combined


def _series(name, values):
    return values if isinstance(values, pl.Series) else pl.Series(
        name, values, dtype=pl.String if isinstance(values, list) else None)


def _take(values, keep):
    """Index an array or a python column, by mask or by position."""
    if not isinstance(values, list):
        return values[keep]
    if keep.dtype == bool:
        return [value for value, take in zip(values, keep.tolist()) if take]
    return [values[position] for position in keep.tolist()]


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
