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
from .expr import ATTR, REL, Col, Expr, Relation, name_of
from .keys import Key

RELATION = "relation"


class Frame:
    """A table of node ids, joined to the graph that gave them meaning."""

    def __init__(self, graph, data, variables=None, pending=None):
        self.graph = graph
        self._data = (None if data is None else
                      data if isinstance(data, pl.DataFrame) else pl.DataFrame(data))
        # {column: type or None} for the columns that hold node ids, in the
        # order they were introduced. A hop with no `from_` reads the last one.
        self.vars = dict(variables or {})
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
            from_=None, norm=False, where=None):
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

        `type` and `where` say *in advance* which nodes are worth landing on,
        and they exist for one reason: cost. A hop's expensive half is not the
        walk -- that is a gather over CSR slices -- but building the result,
        which drags every column the frame already had along to every new row.
        Both are the same restriction a `filter` would apply afterwards, applied
        instead to the arrays the walk produced, so the rows that will not
        survive are never built.

        `where` is a frame of admissible nodes, which is also how an attribute
        predicate gets in -- a frame is where one is written:

            recent = g.nodes("movie").attrs(movie="year").filter(v.movie.year >= 1990)
            frame.hop("has_genre", to="rec", reverse=True, where=recent)

        This is the compiler's old admission mask, back where it belongs. It
        never makes the answer different; it makes it cheaper -- measured on
        MovieLens, 34% off a hop admitting 3% of what it walked, tapering to
        nothing as the admission widens, because the alternative is a hash probe
        against every edge walked.

        There is deliberately no such parameter for the edge's weight. It was
        written, measured, and removed: filtering the weight array before
        building the frame beats polars' own comparison only below about 1%
        selectivity, and costs twice as much at 39%. A knob whose right setting
        requires knowing the selectivity curve is worse than no knob, so a
        weight stays what it is -- a column -- and `.filter(v.x.score >= 3)`
        after the hop is both the spelling and the fast path.
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

        keep = self._admitted(targets, type, where)
        if keep is not None:
            rows, targets, codes, weights = (rows[keep], targets[keep],
                                             codes[keep], weights[keep])

        edge = as_ or relation or target
        self._claim(target, edge, relation)
        added = {target: targets.astype(np.int32), f"{edge}.score": weights}
        if relation is None:
            added[f"{edge}.rel"] = self._names(codes)
        variables = dict(self.vars)
        variables[target] = type
        # not built yet: a filter written next may be able to apply itself to
        # these arrays, which is the difference between admitting a candidate
        # and building a row only to drop it
        pending = _Pending(self._df, rows, added, target, type)
        return Frame(self.graph, None, variables, pending=pending)

    def _admitted(self, targets, type_, where):
        """Which of the nodes just reached are worth building a row for.

        One boolean array, intersected from whatever was said in advance, and
        None when nothing was -- so the common case allocates nothing."""
        keep = None
        if type_ is not None:
            low, high = self.graph.block(type_)
            keep = (targets >= low) & (targets < high)
        if where is not None:
            # a mask over every node, read at the targets: admission is then an
            # array lookup rather than a search, however the set was named
            admissible = np.zeros(self.graph.n_nodes, dtype=bool)
            admissible[self.graph.ids_of(where)] = True
            reached = admissible[targets]
            keep = reached if keep is None else (keep & reached)
        return keep

    def paths(self, relation=None, *, to, hops=(1, 2), type=None, where=None,
              through=None, keep_via=False, from_=None):
        """Walks of more than one length, as one frame.

            .paths(to="rec", type="movie", hops=(1, 2))
            .paths(to="rec", type="movie", hops=(1, 2), through="person")

        `hops` is an inclusive range of lengths, or an int for exactly one. Each
        length is a branch, the branches are concatenated diagonally, and a
        `hops` column says which one a row came from -- so "one hop or two" is
        one frame rather than two and a concat.

        The reason it is a verb and not sugar is the `unique` between the steps.
        A two-hop walk expands the first hop by the degree of the second, and two
        sources reaching the same intermediate carry the same rows onward: that
        product is what makes a naive bridge query exhaust memory, and folding it
        is what the old engine's memoized sub-path search was for. So by default
        the intermediates are dropped and deduplicated at every level.

        `keep_via=True` keeps them, as `via_1`, `via_2`, ... -- and then nothing
        can be deduplicated, because the columns that would collapse are the
        answer. Use it when the walk itself is what you are after.

        `type` and `where` admit the *destination*; `through` admits the
        intermediates, and takes a type name as well as a set of nodes."""
        low, high = (hops, hops) if isinstance(hops, int) else hops
        if low < 1 or high < low:
            raise ValueError(f"hops must be a length or an increasing range, got {hops!r}")
        wanted = range(low, high + 1)
        arrival = [to, f"{to}.score"] + ([] if relation is not None else [f"{to}.rel"])

        chain, branches = self, []
        for step in range(1, high + 1):
            start = None if step > 1 else from_
            if step in wanted:                       # a branch that ends here
                branch = chain.hop(relation, to=to, type=type, where=where,
                                   as_=to, from_=start)
                if not keep_via:
                    # what got here is not what this branch is about: keep where
                    # it started and where it arrived, and fold the routes
                    branch = branch.select(*self.columns, *arrival).unique()
                branches.append(branch.with_columns(hops=pl.lit(step, dtype=pl.Int32)))
            if step == high:
                break
            chain = chain.hop(relation, to=f"via_{step}", as_=f"via_{step}", from_=start,
                              where=through if not isinstance(through, str) else None,
                              type=through if isinstance(through, str) else None)
            if not keep_via:
                # the fold: two routes that met here carry identical rows from
                # now on, and only one of them is worth walking
                chain = chain.select(*self.columns, f"via_{step}").unique()
        return concat(*branches) if branches else self.head(0)

    def degree(self, relation=None, *, of=None, reverse=None, as_=None):
        """How many edges a node has, as a column.

        A fact about the graph, not about what this frame matched -- the same
        number whatever the query asked, which is exactly what distinguishes it
        from `group_by(...).agg(count)`. Read off a per-node array the graph
        computes once, so it costs a gather.

            .degree("has_genre", of="movie")     # -> movie.has_genre_count
        """
        var = self._source_var(of)
        counts = self._degrees(relation, reverse)
        name = as_ or f"{var}.{relation or 'edge'}_count"
        ids = self._df[var].to_numpy()
        return self._wrap(self._df.with_columns(pl.Series(name, counts[ids])))

    def having(self, relation=None, *, where=None, reverse=None, of=None, absent=False):
        """Keep the rows whose node has such an edge -- without walking it.

            .having("directed_by")                     # directs anything at all
            .having("directed_by", where=people)       # directs one of these
            .missing("has_interact", where=watched)    # the negation

        This is a filter, not a traversal: no row is added and none of the frame
        is expanded. With no `where` it reads the node's arity off the graph's
        own count. With one, it walks the *given* set backwards and collects
        what reaches it -- so the cost is the degree of `where`, not of the
        frame. Pass the smaller side.
        """
        var = self._source_var(of)
        admissible = self._reachable(relation, where, reverse)
        keep = admissible[self._df[var].to_numpy()]
        return self._wrap(self._df.filter(pl.Series(~keep if absent else keep)))

    def missing(self, relation=None, *, where=None, reverse=None, of=None):
        """The rows `having` would have dropped."""
        return self.having(relation, where=where, reverse=reverse, of=of, absent=True)

    def _reachable(self, relation, where, reverse):
        """Which nodes of the graph have such an edge, as one boolean array.

        With no `where` that is the node's arity. With one it is read from the
        other end -- what has an edge into this set is what this set reaches
        when walked the other way -- so the cost is the degree of `where` and
        the frame is never expanded."""
        if reverse is None and relation is not None:
            reverse = False
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

    def _degrees(self, relation, reverse):
        """Per-node arity, in the direction a hop would have read.

        A named relation counts forwards unless told otherwise; the wildcard has
        no natural direction and counts both, the way it walks both."""
        graph = self.graph
        if reverse is None and relation is not None:
            reverse = False
        if reverse is None:
            return graph.degree(relation, False) + graph.degree(relation, True)
        return graph.degree(relation, reverse)

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
        keyed = {name: self._signal(name, value, resolver)
                 for name, value in named.items()}
        data = resolver.attach(self._df).with_columns(*columns, **keyed)
        return self._wrap(resolver.detach(data))

    def select(self, *exprs, **named):
        resolver = _Resolver(self)
        columns = [_resolved(one, resolver) for one in _flat(exprs)]
        keyed = {name: _resolved(value, resolver) for name, value in named.items()}
        data = resolver.attach(self._df).select(*columns, **keyed)
        return self._wrap(resolver.detach(data))

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
        column = name_of(by)
        if over is None:
            return self.sort(column, descending=descending).head(n)
        groups = [name_of(one) for one in _flat([over])]
        ranked = pl.col(column).rank("ordinal", descending=descending).over(groups)
        return self._wrap(self._df.filter(ranked <= n)
                          .sort(groups + [column], descending=[False] * len(groups)
                                + [descending]))

    # --- internals -----------------------------------------------------------

    def _signal(self, name, value, resolver):
        """A Signal is a strategy that has not met a graph yet; here it does."""
        if isinstance(value, Signal):
            arrays = tuple(self._df[column].to_numpy() for column in value.columns)
            return pl.Series(name, value.values(self.graph, arrays), dtype=pl.Float64)
        return _resolved(value, resolver)

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
            if paths and all(path[0] == pending.var or ".".join(path) in pending.added
                             for path in paths):
                pushed.append(one)
            else:
                stays.append(one)
        if not pushed:
            return self, stays

        narrow = Frame(self.graph, pending.narrow(), {pending.var: pending.type})
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

    def _source_var(self, from_):
        if from_ is not None:
            return name_of(from_)
        if not self.vars:
            raise ValueError("hop(...) needs a column of nodes to leave from")
        return list(self.vars)[-1]          # the last one introduced

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

    def _claim(self, target, edge, relation):
        """Refuse a hop whose columns would land on ones already there.

        Suffixing the second `has_tag.score` to `has_tag.score_2` is what a
        dataframe would do, and it is wrong here: the two are the weights of two
        different steps, and a filter written against the obvious name would
        silently read the other one. Two steps of one relation are two things,
        so they are named -- which is what `as_` is for."""
        if target in self._df.columns:
            raise ValueError(
                f"this hop would land on {target!r}, which the frame already has. "
                f"Give the new column its own name: hop(..., to=\"...\")")
        clash = [name for name in (f"{edge}.score",
                                   None if relation is not None else f"{edge}.rel")
                 if name is not None and name in self._df.columns]
        if clash:
            raise ValueError(
                f"this hop would land on {', '.join(repr(name) for name in clash)}, "
                f"which the frame already has. Two steps of one relation are two "
                f"different things, and suffixing the second would leave a filter "
                f"written against the obvious name reading the other one -- so name "
                f"this step: hop({relation!r}, to={target!r}, as_=\"...\")")

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


class _Pending:
    """A hop that has walked the graph but not yet built its rows.

    The walk produced four parallel arrays; turning them into a frame means
    taking every column the old frame had at `rows`, which is the expensive half
    of a hop. Holding them one step lets a filter about the node just reached be
    applied here -- to arrays -- instead of to rows that would then be dropped.

    Nothing else is deferred: any other verb reads `_df` and pays for it. This
    is a peephole, not a query planner."""

    __slots__ = ("base", "rows", "added", "var", "type")

    def __init__(self, base, rows, added, var, type_):
        self.base = base            # the frame before the hop
        self.rows = rows            # which base row each candidate came from
        self.added = added          # {column: array} the step produced
        self.var = var              # the new node column
        self.type = type_

    def narrow(self):
        """Just what the step produced -- the frame a pushable predicate is
        evaluated against."""
        return pl.DataFrame({name: _series(name, values)
                             for name, values in self.added.items()})

    def masked(self, keep):
        return _Pending(self.base, self.rows[keep],
                        {name: _take(values, keep) for name, values in self.added.items()},
                        self.var, self.type)

    def build(self):
        data = self.base[self.rows] if len(self.rows) else self.base.clear()
        return data.with_columns([_series(name, values)
                                  for name, values in self.added.items()])


class _Resolver:
    """What turns a name into a column, with the graph in hand.

    Three answers, in this order: a column the frame already has, an attribute
    of that variable's type, or a relation of the graph. The first is free; the
    second reads one array out of the graph; the third has no per-row value at
    all and comes back as a Relation, for `.count()` or `.is_in(...)` to make
    sense of.

    Anything read on demand is attached to the frame for the length of one verb
    and taken off again, because a filter filters rows -- it does not quietly
    widen the frame. `.attrs(...)` is how a column stays."""

    def __init__(self, frame):
        self.frame = frame
        self.extra = {}                    # name -> Series, for this verb only

    # -- the surface an Expr resolves against --

    def lookup(self, path):
        name = ".".join(path)
        if name in self.frame._df.columns or name in self.extra:
            return pl.col(name)
        if len(path) >= 2 and path[0] in self.frame.vars:
            return self._below(path[0], path[1:])
        raise ValueError(
            f"no column {name!r}, and {path[0]!r} is not a node column of this "
            f"frame -- it has: {', '.join(self.frame._df.columns)}")

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

    def ids(self, values):
        return self.frame.graph.ids_of(values)

    # -- attaching and detaching what was read on demand --

    def attach(self, data):
        return data.with_columns(list(self.extra.values())) if self.extra else data

    def detach(self, data):
        stale = [name for name in self.extra if name in data.columns]
        return data.drop(stale) if stale else data

    # -- internals --

    def _below(self, var, rest):
        forced = None
        if rest[0] in (ATTR, REL) and len(rest) > 1:
            forced, rest = rest[0], rest[1:]
        name = ".".join(rest)
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
        name = f"{var}.{attribute}"
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
    if isinstance(values, list):
        return [value for value, take in zip(values, keep.tolist()) if take]
    return values[keep]


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
