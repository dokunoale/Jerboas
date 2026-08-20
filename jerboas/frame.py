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

import operator
from functools import reduce

import numpy as np
import polars as pl

from . import traverse
from .expr import (SCORE, VIA, Col, Expr, direction, is_shadow, name_of,
                   reverse, shadow, shadowed)
from .keys import Key
from .optimize import row_budget
from .resolve import Pending, Plan, Resolver, take

RELATION = "relation"

# Every column carries a confidence, and where something measured one it is kept
# here: an ordinary polars column under a reserved prefix, so filter, sort, join
# and group_by keep it aligned with its values without a line of code from us.
# Hidden from `columns` and from `print`, because it is an attribute of a column
# rather than a column -- `v.rec.score` is how it is read.
#
# Absent means 1.0. A graph with no weights and a query with no fuzzy matching
# therefore allocate nothing at all.
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
        # can run a batch at a time with the filters that follow it (resolve.Plan)
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
        for spec, name in steps:
            if name is not None:
                self._claim(name)
            self._known(spec)

        budget = row_budget()
        if budget is not None and self._produces(steps[0][0]) > budget:
            # described rather than taken: the walk runs when something reads the
            # frame, in slices cut by what each step produces (see optimize.py)
            variables = dict(self.vars)
            for spec, name in steps:
                if name is not None:
                    variables[name] = self._target_type(spec)
            return Frame(self.graph, None, variables, constants=dict(self.constants),
                         plan=Plan(self, steps, budget))
        return self._hop_eager(steps)

    def _hop_eager(self, steps):
        """The walk itself, taken here and now."""
        graph = self.graph
        base = self._df
        source = self._rightmost()
        rows = np.arange(base.height, dtype=np.int64)
        nodes = base[source].to_numpy()
        added, variables = {}, dict(self.vars)
        constants = dict(self.constants)

        for spec, name in steps:
            walked, targets, codes, weights, single = _walk(graph, nodes, spec)
            rows = rows[walked]
            added = {column: take(values, walked) for column, values in added.items()}
            nodes = targets
            if name is None:
                # nothing names these, so nothing tells two routes through them
                # apart: keep one of each and carry that forward
                keep = _distinct(graph, rows, nodes)
                rows, nodes = rows[keep], nodes[keep]
                added = {column: take(values, keep) for column, values in added.items()}
                continue
            added[name] = nodes.astype(np.int32)
            if len(weights) and not (weights == 1.0).all():
                added[shadow(SCORE, name)] = weights
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

    def _known(self, spec):
        """Refuse a step over a relation the graph has never seen.

        Walking one matches nothing, which is a defensible answer to a question
        about a relation that exists elsewhere and an indefensible one to a
        typo -- and a hop names its relation on purpose, so a name the graph
        does not have is the second."""
        for name, _backwards in (_relations(spec) or ()):
            if self.graph.relation_code(name) is None:
                raise ValueError(
                    f"no relation {name!r} in this graph; it has: "
                    f"{', '.join(self.graph.relations)}")

    def _produces(self, spec):
        """Exactly how many rows one step out of this frame would make.

        Not an estimate: a node's degree is a number the graph keeps, so the
        size of an expansion is known before a step is taken. It is what decides
        whether a walk is worth deferring, and where its slices are cut."""
        nodes = self._df[self._rightmost()].to_numpy()
        return int(self._step_degree(spec)[nodes].sum())

    def _step_degree(self, spec):
        """Per node, how many edges one step would follow."""
        relations = _relations(spec)
        if relations is None:                 # any relation, either way
            return self.graph.degree(None, False) + self.graph.degree(None, True)
        total = None
        for name, backwards in relations:
            counts = self.graph.degree(name, backwards)
            total = counts if total is None else total + counts
        return total

    def _slices(self, spec, budget):
        """This frame cut so one step out of each piece makes about `budget`
        rows. A slice out of a hub is shorter than one out of a leaf, which is
        the whole reason to count what a step produces rather than what it is
        given."""
        nodes = self._df[self._rightmost()].to_numpy()
        expansion = self._step_degree(spec)[nodes].astype(np.int64)
        running = np.cumsum(expansion)
        if not len(running) or running[-1] <= budget:
            yield self
            return
        # cut wherever the running total crosses another budget's worth, and
        # never leave a slice empty: a node bigger than the budget on its own is
        # a slice of one, which is as small as a walk can be made
        marks = np.arange(budget, int(running[-1]), budget)
        edges = np.unique(np.searchsorted(running, marks, side="left") + 1)
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

    def _target_type(self, spec):
        """What a step will land in, before it is taken: a schema fact, since
        there is no data yet to read one off."""
        relations = _relations(spec)
        if relations is None or len(relations) != 1:
            return None
        name, backwards = relations[0]
        return self.graph.target_types(name, backwards)

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
        columns = None if subset is None else [name_of(one) for one in _listed(subset)]
        return self._wrap(self._df.unique(subset=columns, keep=keep,
                                          maintain_order=maintain_order))

    def coherent(self, column=None, *, by, through, connection="meet", passes=3):
        """One row per group, choosing the combination most connected to itself.

        A name resolved on its own has only its own popularity to go on. A *set*
        of names has more: the things somebody names together tend to sit near
        each other, so the assignment to prefer is the one whose choices are
        most connected. `Wonderwall` beside `Champagne Supernova` is Oasis;
        `Wonderwall` beside `Come Pick Me Up` is Ryan Adams -- neither is the
        more popular in the abstract, and what decides is the company.

            candidates.coherent(by=v.asked, through=reverse("contains"))

        `by` groups the candidates -- `v.x.needle` is what a search leaves for
        exactly this -- and `through` is one step to where two of them *meet*:
        two songs are connected by the playlists holding both. So the connection
        is one hop and a self-join rather than a walk out and back, which is the
        difference between a second and a minute -- the way back multiplies by
        everything else the meeting place holds, and then throws all of it away.

        `connection` is what that meeting is worth, and the default is that they
        met at all. Counting the meetings instead, or dividing that count by how
        far each candidate reaches, are both worse -- measured, on real
        playlists, by giving back the titles of songs the playlist held and
        seeing how many resolve to the songs it actually held:

            titles      most played     meet      count    count/reach
                 3            66.7%    100.0%     91.7%          75.0%
                16            81.2%     98.4%     89.1%          77.5%

        Counting favours the popular, since a song in seventeen hundred
        playlists meets more of everything; dividing the count overshoots the
        other way, handing the choice to whichever obscure candidate shares a
        large fraction of its few neighbours. Whether two candidates keep
        company at all is neither, and it is the thing being asked.

        The choice itself is `k**n` combinations, so it is not enumerated:
        starting from the frame's own order, each group takes the candidate best
        connected to what the others currently hold, repeatedly. Two or three
        passes settle it, and the result is a local maximum rather than the
        maximum -- which is the usual trade and worth saying out loud.

        **When the graph says nothing the frame's order decides**, so sorting by
        whatever you would have used alone -- popularity, closeness -- makes
        this an improvement on that rather than a replacement for it.
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

        One step out to the meeting places and a self-join there. Walking back
        would visit everything else those places hold, which is a million rows
        to keep a few hundred."""
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
        reach = self._step_degree(through)[self._df[node].to_numpy()].astype(float)
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

    def chunked(self, size):
        """The frame in slices of `size` rows, as frames.

        What a caller does by hand when a walk out of the whole thing would not
        fit -- the benchmark expands its users a block at a time for exactly
        this reason. A planner would choose the size; until there is one, the
        caller does."""
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
        `"product"` as independent evidence multiplied, `None` drops it. A
        callable is given the shadow's expression and returns whatever it
        should become."""
        return _GroupBy(self, [name_of(one) for one in _flat(by)], maintain_order,
                        confidence)

    def top(self, n, by=None, descending=True, over=None):
        """The n best rows. `by` defaults to a `score` column when there is one,
        because that is what the frame was building up to.

        `over` makes it the n best *per group* -- one user's ten films, one
        walk's five most promising steps -- which is a window function rather
        than a sort, so it costs one pass instead of one query per group:

            .top(10, by="score", over="user")
            .hop(mid=()).top(5, by=v.pr, over="seed").hop(rec=())

        That second line is a beam search: keep the k most promising partial
        walks at each step and expand only those. It was an engine once."""
        if by is None:
            by = "score" if "score" in self._df.columns else None
        if by is None:
            return self.head(n) if over is None else self
        resolver = Resolver(self)
        ordering = _resolved(by, resolver)
        if isinstance(ordering, str):          # polars reads a bare name as a column
            ordering = pl.col(ordering)
        data = resolver.attach(self._df)
        if over is None:
            data = data.sort(ordering, descending=descending).head(n)
            return resolver.wrap(resolver.detach(data))
        groups = [name_of(one) for one in _flat([over])]
        ranked = ordering.rank("ordinal", descending=descending).over(groups)
        data = (data.filter(ranked <= n)
                .sort(groups + [ordering], descending=[False] * len(groups) + [descending]))
        return resolver.wrap(resolver.detach(data))

    def _defer(self, predicates):
        """Hand a condition to a walk that has not happened, so the batch it
        runs in is the batch it filters."""
        plan = self._plan
        target = next(name for _spec, name in reversed(plan.steps) if name is not None)
        joins, stays = [], []
        for one in predicates:
            paths = one.reads() if isinstance(one, Expr) else _root_paths(one)
            (joins if paths and all(path[0] == target for path in paths)
             else stays).append(one)
        if not joins:
            return self, stays
        return Frame(self.graph, None, dict(self.vars),
                     constants=dict(self.constants),
                     plan=plan.narrowed(joins)), stays

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
        resolver = Resolver(narrow)
        exprs = [_resolved(one, resolver) for one in pushed]
        keep = (resolver.attach(narrow._df)
                .select(_all(exprs).fill_null(False).alias("keep"))["keep"].to_numpy())
        return Frame(self.graph, None, self.vars, pending=pending.masked(keep)), stays

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


_REDUCE = {
    "min": lambda scores: pl.min_horizontal(scores),
    "max": lambda scores: pl.max_horizontal(scores),
    "mean": lambda scores: pl.mean_horizontal(scores),
    "product": lambda scores: reduce(operator.mul, scores),
    "sum": lambda scores: pl.sum_horizontal(scores),
}


FOLD = {
    "mean": lambda expr: expr.mean(),
    "min": lambda expr: expr.min(),
    "max": lambda expr: expr.max(),
    "product": lambda expr: expr.product(),
    "first": lambda expr: expr.first(),
}


class _GroupBy:
    """What `group_by` returns: only `agg` and `len` follow it."""

    def __init__(self, frame, by, maintain_order, confidence="mean"):
        self.frame = frame
        self.by = by
        self.maintain_order = maintain_order
        self.confidence = confidence

    def agg(self, *exprs, **named):
        resolver = Resolver(self.frame)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        keyed.update(self._confidence())
        grouped = (resolver.attach(self.frame._df)
                   .group_by(self.by, maintain_order=self.maintain_order))
        return resolver.wrap(resolver.detach(grouped.agg(*columns, **keyed)))

    def len(self, name="len"):
        grouped = self.frame._df.group_by(self.by, maintain_order=self.maintain_order)
        return self.frame._wrap(grouped.agg(pl.len().alias(name), **self._confidence()))

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
