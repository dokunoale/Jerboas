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

The frame is eager, so `print(g.nodes("movie"))` is a table rather than a plan.
Inside `optimize` a hop is deferred instead, and runs in slices when something
reads the frame (plan/).
"""

import operator
from functools import reduce

import numpy as np
import polars as pl

from . import traverse
from .expr import (INCLUSION, SCORE, VIA, Col, Expr, direction, is_shadow,
                   name_of, reverse, shadow, shadowed)
from ..store.keys import Key
from ..plan.optimize import budget
from ..plan.plan import Plan
from ..plan.planner import row_local
from .resolve import FOLD, Pending, Resolver, _series, take
from .step import Step, root_of

RELATION = "relation"


class Frame:
    """A table of node ids, joined to the graph that gave them meaning."""

    def __init__(self, graph, data, variables=None, pending=None, constants=None,
                 plan=None):
        self.graph = graph
        self._data = (None if data is None else
                      data if isinstance(data, pl.DataFrame) else pl.DataFrame(data))
        # {column: type or None} for the columns that hold node ids, in the
        # order they were introduced. A hop with no `from_` reads the last one.
        self.vars = dict(variables or {})
        # {(kind, column): value} for the provenance that says the same thing
        # about every row -- the one relation a named hop walked, the one needle
        # a search asked for. A constant needs no array, and `v.x.via` should
        # answer all the same
        self.constants = dict(constants or {})
        # a hop that has walked but not built its rows yet (see resolve.Pending).
        # The only thing that reads it is `filter`, which may be able to apply
        # itself to the arrays instead of to the rows they would become
        self._pending = pending
        # a hop that has not even walked: inside `optimize`, so that the walk
        # can run a batch at a time with the filters that follow it (plan/plan.py)
        self._plan = plan

    @property
    def _df(self):
        """The rows. Building them is what a deferred hop was putting off, so
        anything that touches this pays for it -- and `filter` gets first
        refusal."""
        if self._plan is not None:
            plan, self._plan = self._plan, None
            self._data = plan.build()
        if self._pending is not None:
            self._data = self._pending.build()
            self._pending = None
        return self._data

    # --- construction --------------------------------------------------------

    def _wrap(self, data, variables=None, constants=None):
        variables = self.vars if variables is None else variables
        constants = self.constants if constants is None else constants
        kept = {name: type_ for name, type_ in variables.items() if name in data.columns}
        held = {(kind, column): value
                for (kind, column), value in constants.items()
                if column in data.columns or column.partition(".")[0] in data.columns}
        return Frame(self.graph, data, kept, constants=held)

    # --- what it is ----------------------------------------------------------

    @property
    def pl(self):
        """The polars DataFrame, for anything this class does not forward.

        Without the shadows: they are attributes of columns rather than columns,
        and a frame handed to something else should carry what it says it holds.
        `.raw` is the whole thing, bookkeeping included."""
        return self.visible

    @property
    def raw(self):
        """The polars DataFrame exactly as it is, shadows and all."""
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

        A keyword names the column the step's arrivals are kept in; a positional
        step is walked and not kept, which is what lets it be folded -- two
        routes through something nobody named are one answer. The last step must
        be named, because where the walk ends is what the frame holds.

        A step is a relation, `~name` for it read backwards, or a collection for
        any of several. The empty collection is any relation at all, either way.
        `step(...)` says the same and can budget it: `step("~rated").top(20)`
        follows the 20 best edges of each row rather than all of them (step.py).

        Walking leaves from the rightmost column of nodes; to leave from
        another, `select` it and `join` the result back.

        Only the named columns are added. What the step measured is each one's
        confidence (`v.genre.score`) and which relation it walked is
        `v.genre.via`.
        """
        steps = [(Step.of(spec), None) for spec in through]
        steps += [(Step.of(spec), name) for name, spec in named.items()]
        if not named:
            raise ValueError(
                "the last step of a hop must be named: where the walk ends is "
                "what the frame holds. hop(..., rec=\"has_genre\")")
        for spec, name in steps:
            if name is not None:
                self._claim(name)
            elif spec.measure is not None or spec.watched:
                raise ValueError(
                    "a step that measures must be named: two routes through an "
                    "unnamed step are folded into one, and the walk would lose "
                    "what the other measured")
            spec.known(self.graph)

        current = budget()
        if current is not None:
            return self._planned(steps, current)
        return self._hop_eager(steps)

    def _planned(self, steps, current):
        """The walk described rather than taken: it runs when something reads
        the frame, in slices and from whichever end is cheaper (plan/plan.py).
        A hop after a hop that has not run yet extends the same plan, so the
        middle of the walk never exists whole."""
        plan = (self._plan.extended(steps) if self._plan is not None
                else Plan.of(self, steps, current))
        variables, constants = dict(self.vars), dict(self.constants)
        for spec, name in steps:
            if name is None:
                continue
            variables[name] = spec.target_type(self.graph)
            single = spec.single()
            if single is not None:
                constants[(VIA, name)] = single
        return Frame(self.graph, None, variables, constants=constants, plan=plan)

    def _hop_eager(self, steps, toward=None):
        """The walk itself, taken here and now -- from the far end when the
        planner found that cheaper (`toward`, for a single step)."""
        graph = self.graph
        base = self._df
        source = self._rightmost()
        rows = np.arange(base.height, dtype=np.int64)
        nodes = base[source].to_numpy()
        added, variables = {}, dict(self.vars)
        constants = dict(self.constants)
        # the confidence of the node each row stands on: what a probability
        # step multiplies its transition by, so that its arrival's confidence is
        # the probability the walk got there (None: 1.0 everywhere)
        held = shadow(SCORE, source)
        carried = (base[held].fill_null(1.0).to_numpy() if held in base.columns
                   else None)

        for spec, name in steps:
            walked, targets, codes, weights, pi = spec.walk(
                graph, nodes, name, self._context(base, rows, added, variables),
                None if toward is None else toward.reaches)
            if spec.measure is not None and carried is not None:
                weights = weights * carried[walked]
            carried = weights
            single = spec.single()
            rows = rows[walked]
            added = {column: take(values, walked) for column, values in added.items()}
            nodes = targets
            if name is None:
                # nothing names these, so nothing tells two routes through them
                # apart: keep one of each and carry that forward
                keep = _distinct(graph, rows, nodes)
                rows, nodes, carried = rows[keep], nodes[keep], carried[keep]
                added = {column: take(values, keep) for column, values in added.items()}
                continue
            added[name] = nodes.astype(np.int32)
            if len(weights) and not (weights == 1.0).all():
                added[shadow(SCORE, name)] = weights
            # a draw's own chances, on the column it landed in: the chance of
            # the row is the product of these, which is what an estimate reads
            if pi is not None and len(pi) and not (pi == 1.0).all():
                added[shadow(INCLUSION, name)] = pi
            if single is None and len(codes):
                added[shadow(VIA, name)] = self._names(codes)
            elif single is not None:
                # every row walked the same relation, so one value says it
                constants[(VIA, name)] = single
            variables[name] = self._one_type(nodes)

        if any(name is None for _spec, name in steps):
            # a row that differs only where nothing was named is not a different
            # row: two routes through an unnamed step are one answer
            keep = _folded(rows, [added[name] for _spec, name in steps if name])
            rows = rows[keep]
            added = {column: take(values, keep) for column, values in added.items()}

        last = steps[-1][1]
        return Frame(graph, None, variables, pending=Pending(base, rows, added, last),
                     constants=constants)

    def _context(self, base, rows, added, variables):
        """What a step's map may read of the frame the walk leaves from: the
        columns of the variables it names, at the rows the candidates came from
        -- and only those, so a map that reads a user builds one column, not
        the frame."""
        def context(roots, walked):
            names = [name for name in base.columns if root_of(name) in roots]
            columns = base.select(names)[rows[walked]] if names else pl.DataFrame()
            carried = [_series(name, take(values, walked))
                       for name, values in added.items() if root_of(name) in roots]
            if carried:
                columns = columns.hstack(carried) if names else pl.DataFrame(carried)
            return columns, {root: variables[root] for root in roots if root in variables}
        return context

    def _nodes(self):
        """The ids a walk leaves from."""
        return self._df[self._rightmost()].to_numpy()

    def _slices(self, degree, budget):
        """This frame cut so one step out of each piece makes about `budget`
        rows, given how many rows each node's step makes (`degree`). A slice
        out of a hub is shorter than one out of a leaf, which is the whole
        reason to count what a step produces rather than what it is given."""
        expansion = degree[self._nodes()].astype(np.int64)
        running = np.cumsum(expansion)
        if not len(running) or running[-1] <= budget:
            yield self
            return
        # a node starts a new slice when its running total passes a mark, so a
        # slice never makes more than the budget unless one node alone does
        marks = np.arange(budget, int(running[-1]), budget)
        edges = np.unique(np.searchsorted(running, marks, side="right"))
        starts = np.concatenate([[0], edges])
        stops = np.concatenate([edges, [len(expansion)]])
        for start, stop in zip(starts.tolist(), stops.tolist()):
            if stop > start:
                yield self._wrap(self._df.slice(start, stop - start))

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
        """One column as an int array -- what makes a frame usable as a set of
        nodes anywhere ids are.

        With one column of nodes there is nothing to say; with several there is,
        so it asks rather than picking one."""
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
        if frame._plan is not None and predicates:
            frame, predicates = frame._defer(predicates)
        if frame._pending is not None and predicates:
            frame, predicates = frame._pushdown(predicates)
        if not predicates and not named:
            return frame
        resolver = Resolver(frame)
        exprs = [_resolved(one, resolver) for one in _flat(predicates)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        data = resolver.attach(frame._df).filter(*exprs, **keyed)
        return resolver.wrap(resolver.detach(data))

    def with_columns(self, *exprs, **named):
        resolver = Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        data = resolver.attach(self._df).with_columns(*columns, **keyed)
        # what was asked for stays; what was only read to compute it does not
        asked = set(named) | {one.meta.output_name() for one in columns
                              if isinstance(one, pl.Expr)}
        return resolver.wrap(resolver.detach(data, keep=asked))

    def select(self, *exprs, **named):
        """Choose columns. A column's confidence and provenance travel with it,
        because they are attributes of it and not columns of their own."""
        resolver = Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        data = resolver.attach(self._df).select(*columns, **keyed)
        carried = [name for name in self._df.columns
                   if is_shadow(name) and shadowed(name)[1] in data.columns
                   and name not in data.columns]
        # only when the selection kept the rows it was given: a selection that
        # aggregates has no row to hang a per-row confidence on
        if carried and data.height == self._df.height:
            data = data.hstack(resolver.attach(self._df).select(carried))
        return resolver.wrap(data)

    def sort(self, *by, descending=False, nulls_last=True):
        resolver = Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(by)]
        data = resolver.attach(self._df).sort(*columns, descending=descending,
                                              nulls_last=nulls_last)
        return resolver.wrap(resolver.detach(data))

    def unique(self, subset=None, keep="first", maintain_order=True):
        """Fold duplicate rows. The confidence of the row that is kept is kept
        with it -- `keep="first"` by default and the order maintained, so which
        row that is, is the one you can see rather than one of several.

        To fold by a rule instead, `group_by(...).agg(...)`, which says what
        becomes of a confidence rather than inheriting one."""
        if self._plan is not None and keep in ("first", "last", "any"):
            # folded a slice at a time and again across them: the first of a
            # row in the first slice holding it is its first anywhere
            from ..plan import stream
            return stream.unique(self, subset, keep, maintain_order)
        columns = None if subset is None else [name_of(one) for one in _listed(subset)]
        return self._wrap(self._df.unique(subset=columns, keep=keep,
                                          maintain_order=maintain_order))

    def coherent(self, column=None, *, by, through, connection="meet", passes=3):
        """One row per group, choosing the combination most connected to itself.

            candidates.coherent(by=v.asked, through=reverse("contains"))

        A name resolved alone has only its own popularity to go on; a *set* of
        names has the company they keep. `by` groups the candidates -- what
        `v.x.needle` leaves -- and `through` is one step to where two of them
        meet.

        `connection` is what a meeting is worth: "meet" (that they met at all),
        "count" (how often, which favours the popular), or "share"/"damped"
        (the count divided by how far each reaches, which leans to the obscure).

        The choice is `k**n` combinations and is not enumerated. Starting from
        the frame's own order, each group takes the candidate best connected to
        what the others hold, repeatedly -- a local maximum, in two or three
        passes. When nothing is connected the order stands, so sorting by
        whatever you would have used alone makes this an improvement on it
        rather than a replacement.
        """
        node = self._var(column)
        groups = self._df[name_of(by)].to_list()
        rows = self._df.height
        if rows <= 1:
            return self

        weights = self._connections(node, through, rows, connection)
        chosen = _assign(groups, weights, passes)
        return self._wrap(self._df[sorted(chosen)])

    def _connections(self, node, through, rows, connection="count"):
        """How often each pair of candidates meets, as a dense (rows, rows)
        matrix -- small, being one per name times a handful.

        One step out to the meeting places and a self-join there, rather than
        walking back and visiting everything else those places hold."""
        marked = self._wrap(self._df.with_row_index(_ROW))
        met = marked.hop(**{_MEET: through}).pl.select([_ROW, _MEET])
        paired = (met.join(met, on=_MEET, suffix="_other")
                  .group_by([_ROW, _ROW + "_other"]).len())
        weights = np.zeros((rows, rows))
        for one, other, count in paired.rows():
            if one != other:
                weights[one, other] = 1.0 if connection == "meet" else count
        if connection == "meet":
            return weights

        if connection == "count":
            return weights                  # what the crowd said, popularity and all
        if connection not in ("share", "damped"):
            raise ValueError(f"unknown connection {connection!r}; expected "
                             f"'share', 'damped' or 'count'")
        reach = Step.of(through).degree(self.graph)[self._df[node].to_numpy()].astype(float)
        power = 0.5 if connection == "share" else 0.25
        scale = np.outer(reach, reach) ** power
        return np.divide(weights, scale, out=np.zeros_like(weights), where=scale > 0)

    def confidence(self, rule="min", into="confidence"):
        """Every column's confidence, reduced to one number per row.

        Confidence does not compose on its own: `v.tag.score` is the step that
        revealed `tag` and `v.rec.score` the step after, and a string similarity,
        an edge weight and a model's plausibility are not the same measure. So
        reducing them is a decision, and this is where it is written down.

            .confidence("min")        # as certain as the least certain step
            .confidence("product")    # independent evidence, multiplied
            .confidence(lambda *scores: 0.7 * scores[0] + 0.3 * scores[1])

        Columns nothing measured count as 1.0 and are left out, so a frame in
        which nothing was in doubt reduces to 1.0 rather than to nothing."""
        if not callable(rule) and rule not in _REDUCE:
            raise ValueError(f"unknown confidence rule {rule!r}; expected one of "
                             f"{sorted(_REDUCE)} or a callable")
        scores = [pl.col(name) for name in self._df.columns if is_shadow(name)
                  and shadowed(name)[0] == SCORE]
        if not scores:
            return self._wrap(self._df.with_columns(pl.lit(1.0).alias(into)))
        reduced = rule(*scores) if callable(rule) else _REDUCE[rule](scores)
        return self._wrap(self._df.with_columns(reduced.alias(into)))

    def batches(self):
        """The frame a slice at a time, as frames.

        A walk planned inside `optimize` hands out each slice as it is walked,
        filtered and reduced, and none is kept once the next is asked for -- so
        an answer that does not fit can still be written out, or folded, as it
        comes. Anything else is one batch, itself.

            with jb.optimize():
                walk = seeds.hop(peer="~has_interact", rec="has_interact")
            for part in walk.batches():
                part.pl.write_parquet(...)

        Each call walks again: a plan is a description, and reading it twice
        takes it twice."""
        if self._plan is None:
            yield self
            return
        yield from self._plan.parts()

    def chunked(self, size):
        """The frame in slices of `size` rows, as frames.

        The manual counterpart of `batches`, for when a walk out of the whole
        frame would not fit: here the caller chooses the size."""
        for start in range(0, max(self._df.height, 1), size):
            part = self._wrap(self._df.slice(start, size))
            if len(part):
                yield part

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
        """Rename columns, and with each one whatever hangs off it.

        A shadow whose column was renamed would otherwise go on naming a column
        that is not there: `v.rec.score` would read 1.0 and the measurement
        would still be in the frame under the old name."""
        full = dict(mapping)
        for name in self._df.columns:
            kind_column = shadowed(name)
            if kind_column and kind_column[1] in mapping:
                full[name] = shadow(kind_column[0], mapping[kind_column[1]])
        variables = {mapping.get(name, name): type_ for name, type_ in self.vars.items()}
        constants = {(kind, mapping.get(column, column)): value
                     for (kind, column), value in self.constants.items()}
        return self._wrap(self._df.rename(full), variables, constants)

    def join(self, other, on=None, how="inner", **kwargs):
        right = other._df if isinstance(other, Frame) else other
        columns = None if on is None else [name_of(one) for one in _listed(on)]
        variables = dict(self.vars)
        if isinstance(other, Frame):
            variables.update(other.vars)
        constants = dict(self.constants)
        if isinstance(other, Frame):
            constants = {**other.constants, **constants}
        return self._wrap(self._df.join(right, on=columns, how=how, **kwargs),
                          variables, constants)

    def group_by(self, *by, maintain_order=True, confidence="mean"):
        """Fold rows together. What becomes of their confidence is `confidence`:
        by default the mean of the rows folded in, so a group of rows nobody
        doubted stays certain and a group of weak matches says so.

        `"min"` reads a group as its weakest member, `"max"` as its best,
        `"product"` as independent evidence multiplied, `"sum"` as the mass of
        the rows folded -- the probability of any of several walks, when each
        row's confidence is the probability of its own (`step.probability()`)
        -- and `None` drops it. A callable is given the shadow's expression and
        returns whatever it should become."""
        return _GroupBy(self, [name_of(one) for one in _flat(by)], maintain_order,
                        confidence)

    def top(self, n, by=None, descending=True, over=None, spread=None, temperature=0.0,
            seed=None):
        """The n best rows. `by` defaults to a `score` column when there is one,
        because that is what the frame was building up to.

        `over` makes it the n best *per group* -- one user's ten films, one
        walk's five most promising steps -- which is a window function rather
        than a sort, so it costs one pass instead of one query per group:

            .top(10, by="score", over="user")

        The k best edges out of each row *during* a walk are a budgeted step
        instead (`step(...).top(k)`, step.py): cutting here, after the hop,
        pays for the whole expansion first.

        `spread` makes it n in all, taken in turns from each group: the best of
        every group, then the second best of every group, and so on, each turn
        in order of score. A group with little to offer runs out and the others
        fill in, so an answer about several things covers all of them without
        a quota to size -- the partition form of choosing for diversity.

            .top(10, by="score", spread="part")

        `temperature` makes the choice a sample rather than a maximum: at zero
        the n best, above it the n drawn in proportion to `exp(score / t)`. The
        trick is Gumbel's -- perturb each score by `-log(-log(u))` and take the
        top n -- which is exactly sampling without replacement from that
        distribution, at the cost of one array of noise."""
        if over is not None and spread is not None:
            raise ValueError("top() takes n per group (`over`) or n across groups "
                             "(`spread`), not both")
        if (self._plan is not None and not temperature and spread is None
                and (by is None or row_local(by, self.graph))):
            # the n best of the whole are among the n best of some slice, so
            # keeping n per slice is enough -- and all a planned walk keeps
            from ..plan import stream
            return stream.top(self, n, by, descending, over)
        if by is None:
            by = "score" if "score" in self._df.columns else None
        if by is None:
            return self.head(n) if over is None else self
        resolver = Resolver(self)
        ordering = _resolved(by, resolver)
        if isinstance(ordering, str):          # polars reads a bare name as a column
            ordering = pl.col(ordering)
        if temperature:
            ordering = _perturbed(ordering, temperature, resolver, self._df.height, seed)
        data = resolver.attach(self._df)
        if spread is not None:
            groups = [name_of(one) for one in _flat([spread])]
            turn = ordering.rank("ordinal", descending=descending).over(groups)
            data = data.sort([turn, ordering], descending=[False, descending]).head(n)
            return resolver.wrap(resolver.detach(data))
        if over is None:
            data = data.sort(ordering, descending=descending).head(n)
            return resolver.wrap(resolver.detach(data))
        groups = [name_of(one) for one in _flat([over])]
        ranked = ordering.rank("ordinal", descending=descending).over(groups)
        data = (data.filter(ranked <= n)
                .sort(groups + [ordering], descending=[False] * len(groups) + [descending]))
        return resolver.wrap(resolver.detach(data))

    def _defer(self, predicates):
        """Hand conditions to a walk that has not happened, so the slice each
        runs in is the slice it filters.

        All of them or none: a condition whose verdict depends on other rows (a
        mean, a rank, a `like` keeping the k best) has to see the whole answer,
        and the others written beside it have to wait with it -- applied first,
        slice by slice, they would change which rows its mean is taken over."""
        flat = list(_flat(predicates))
        if not all(row_local(one, self.graph) for one in flat):
            return self, predicates
        return Frame(self.graph, None, dict(self.vars),
                     constants=dict(self.constants),
                     plan=self._plan.narrowed(flat)), []

    # --- pushing a predicate into a hop that has not built its rows ----------

    def _pushdown(self, predicates):
        """Apply what can be applied to the arrays, and hand back the rest.

        A predicate is pushable when everything it reads is about the node the
        hop just reached, or about the step itself."""
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
        resolver = Resolver(narrow)
        exprs = [_resolved(one, resolver) for one in pushed]
        keep = (resolver.attach(narrow._df)
                .select(_all(exprs).fill_null(False).alias("keep"))["keep"].to_numpy())
        # what the predicate measured stays, as it would on the rows: a `like`'s
        # closeness is the column's confidence from here on
        measured = pending.with_added(resolver.keep)
        return Frame(self.graph, None, self.vars, pending=measured.masked(keep),
                     constants={**self.constants, **resolver.constants}), stays

    def _var(self, variable=None):
        if variable is not None:
            return name_of(variable)
        nodes = [name for name in self._df.columns if name in self.vars]
        if not nodes:
            raise ValueError("this frame has no column of nodes")
        if len(nodes) > 1:
            raise ValueError(
                f"this frame has {len(nodes)} columns of nodes -- "
                f"{', '.join(nodes)} -- so which one is a question: name it.")
        return nodes[0]

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
        present = self._plan.names() if self._plan is not None else self._df.columns
        if target in present:
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


_REDUCE = {
    "min": lambda scores: pl.min_horizontal(scores),
    "max": lambda scores: pl.max_horizontal(scores),
    "mean": lambda scores: pl.mean_horizontal(scores),
    "product": lambda scores: reduce(operator.mul, scores),
    "sum": lambda scores: pl.sum_horizontal(scores),
}




class _GroupBy:
    """What `group_by` returns: only `agg` and `len` follow it."""

    def __init__(self, frame, by, maintain_order, confidence="mean"):
        self.frame = frame
        self.by = by
        self.maintain_order = maintain_order
        self.confidence = confidence

    def agg(self, *exprs, **named):
        if self.frame._plan is not None:
            from ..plan import stream
            folded = stream.agg(self, exprs, named)
            if folded is not None:
                return folded
        resolver = Resolver(self.frame)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        keyed.update(self._confidence())
        grouped = (resolver.attach(self.frame._df)
                   .group_by(self.by, maintain_order=self.maintain_order))
        return resolver.wrap(resolver.detach(grouped.agg(*columns, **keyed)))

    def len(self, name="len"):
        if self.frame._plan is not None:
            from ..plan import stream
            return stream.agg(self, (), {}, length=name)
        frame = self.frame
        grouped = frame._df.group_by(self.by, maintain_order=self.maintain_order)
        pi = _inclusion_of(frame)
        if pi is None:
            return frame._wrap(grouped.agg(pl.len().alias(name), **self._confidence()))
        # the walk these rows came from drew: each kept row stands for 1/pi of
        # the exact walk's rows, so the group's size is estimated rather than
        # truncated (Horvitz-Thompson) -- and the estimate's standard error is
        # the count's confidence
        return frame._wrap(grouped.agg(
            (1 / pi).sum().alias(name),
            ((1 - pi) / pi ** 2).sum().sqrt().alias(shadow(SCORE, name)),
            **self._confidence()))

    def _confidence(self):
        """What the grouped columns' confidence becomes.

        Only the columns being grouped on: everything else is leaving anyway, and
        inventing a confidence for a column that did not exist before the
        aggregation would be inventing one for a number nobody measured."""
        if self.confidence is None:
            return {}
        fold = self.confidence if callable(self.confidence) else FOLD.get(self.confidence)
        if fold is None:
            raise ValueError(f"unknown confidence rule {self.confidence!r}; expected "
                             f"one of {sorted(FOLD)}, a callable, or None")
        return {shadow(SCORE, column): fold(pl.col(shadow(SCORE, column)))
                for column in self.by
                if shadow(SCORE, column) in self.frame._df.columns}


def _inclusion_of(frame):
    """The chance each row had of being kept by the walk's draws, as one
    expression -- the product of the inclusion shadows the frame carries -- or
    None when nothing drew. Absent shadows are the 1.0 they stand for."""
    columns = [name for name in frame._df.columns
               if (shadowed(name) or ("",))[0] == INCLUSION]
    if not columns:
        return None
    return reduce(operator.mul, (pl.col(name).fill_null(1.0) for name in columns))


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
    constants = {}
    for frame in frames:
        constants.update(frame.constants)
    return Frame(frames[0].graph, pl.concat(_aligned(frames), how=how), variables,
                 constants=constants)


def _aligned(frames):
    """The frames with their empty columns typed like the ones that have rows.

    A branch that matched nothing still has to line up with one that did, and an
    empty column of unknown type is polars' Null -- which refuses to stack onto
    a String. So a column that is Null in one frame takes the type another frame
    gives it."""
    known, mixed = {}, set()
    for frame in frames:
        for name, dtype in frame._df.schema.items():
            if dtype == pl.Null:
                continue
            if name in known and known[name] != dtype:
                # an Enum and a String hold the same words in two encodings, and
                # only one of them stacks onto the other
                mixed.add(name)
            known.setdefault(name, dtype)
    out = []
    for frame in frames:
        recast = [pl.col(name).cast(pl.String if name in mixed else known[name])
                  for name, dtype in frame._df.schema.items()
                  if name in known and (dtype == pl.Null or name in mixed)]
        out.append(frame._df.with_columns(recast) if recast else frame._df)
    return out


_ROW, _MEET = "__coherent_row", "__coherent_meet"
_JITTER = "__top_jitter"


def _assign(groups, weights, passes):
    """One row per group, hill-climbing on how connected the choices are.

    Starts from the first row of each group, which is why the frame's own order
    is the fallback: with nothing connected, nothing moves."""
    order, members = [], {}
    for row, group in enumerate(groups):
        if group not in members:
            members[group] = []
            order.append(group)
        members[group].append(row)
    chosen = {group: rows[0] for group, rows in members.items()}
    if len(order) < 2:
        return set(chosen.values())

    for _pass in range(passes):
        moved = False
        for group in order:
            others = [chosen[one] for one in order if one != group]
            scores = [weights[row, others].sum() for row in members[group]]
            best = members[group][int(np.argmax(scores))]
            if best != chosen[group]:
                chosen[group] = best
                moved = True
        if not moved:
            break
    return set(chosen.values())


def _perturbed(ordering, temperature, resolver, height, seed):
    """A score with Gumbel noise: the top n of it is a sample of n, drawn in
    proportion to `exp(score / temperature)`.

    Measured in the scores' own spread rather than their units, so a temperature
    means the same thing whether the column holds a PageRank around 0.01 or a
    count around 600: at 1 the noise is as large as the spread it disturbs, at
    0.1 it only shuffles what was nearly tied.

    The noise is a column rather than a literal, because `over` evaluates a
    group at a time and a literal of the frame's height means nothing there."""
    rng = np.random.default_rng(seed)
    uniform = rng.random(height)
    resolver.extra[_JITTER] = pl.Series(
        _JITTER, -np.log(-np.log(np.clip(uniform, 1e-12, 1.0 - 1e-12))))
    spread = ordering.std().fill_null(0.0)
    return pl.when(spread > 0) \
             .then(ordering / (spread * temperature) + pl.col(_JITTER)) \
             .otherwise(pl.col(_JITTER))


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
