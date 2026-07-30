"""Query: the pipeline, and the compiler that binds passive objects to a graph.

Query owns the two responsibilities the object families deliberately keep out of
themselves:

  1. The pipeline verbs -- select / where / rank / top / groupby / using. The
     only methods a user calls; each returns self, and nothing runs until the
     result is iterated (fully lazy).

  2. Compilation -- the inner _Compiler turns the passive refs/conditions into
     the neutral IR (ir.py). It implements core.Compiler: the single place that
     knows the schema, allocates variables, and decides whether a Compare seeds a
     variable or adds a predicate. Conditions describe themselves to it; it owns
     the "how".

Compilation now *resolves* rather than defers: because a node's constraints are
all binding-independent, the compiler evaluates them against the graph's typed
columns and hands the engine one boolean mask per variable. Nothing about a
candidate is re-decided per row.

Rows carry integer node ids the whole way through -- search, anti-joins,
ranking, grouping -- and become Key/Rel values only in _render, at the edge.
"""

import numpy as np

from .core import Compiler, Strategy
from .conditions import Or
from .keys import Rel
from .columns import Column, column_mask
from .refs import Node, Attr, Edge, Path, Degree, PathStep, PathStepAttr
from .strategies import Score, ExprStrategy
from .engine import Default
from .ir import NodeSpec, EdgeSpec, Output, SearchSpec


def _projection_var(projection):
    """The pattern variable underneath a projection (rec.title -> rec, path -> path)."""
    if isinstance(projection, (Attr, Degree)):
        return projection.node
    if isinstance(projection, (PathStep, PathStepAttr)):
        return projection.path
    if isinstance(projection, (Node, Path)):
        return projection
    return None   # Score() and the like have no variable


class _Scope:
    """One ranking level: a (possibly empty) sequence of strategies and a limit."""

    def __init__(self):
        self.rank = []
        self.limit = None


class Query:
    def __init__(self, graph, *projections):
        self.graph = graph
        self.projections = projections
        # distinct selected variables (Node/Path), in projection order -> one
        # visible column each. Column roles are fixed by this order, known before
        # the search runs, so strategies can resolve their columns up front.
        self.nodes = self._select_vars(projections)
        self.node_columns = {}     # type -> first visible column of that type
        self.primary_column = None # first Node column
        self.path_column = None    # first Path column
        self._columns = {}         # id(ref) -> result column index (visible now, hidden added at compile)
        for col, var in enumerate(self.nodes):
            self._columns[id(var)] = col
            if isinstance(var, Node):
                if self.primary_column is None:
                    self.primary_column = col
                if var.type is not None and var.type not in self.node_columns:
                    self.node_columns[var.type] = col
            elif isinstance(var, Path) and self.path_column is None:
                self.path_column = col

        self.conditions = []
        self.engine = Default()
        self.group_var = None
        self.scopes = [_Scope()]   # innermost-last; scopes[0] is the outer/global level
        self._current = 0
        self.visited_nodes = set()
        self._row_scores = {}
        self._visible_count = len(self.nodes)
        self._has_hidden = False
        self._antijoins = []
        self._paths = {}
        self._results = None

    def _select_vars(self, projections):
        seen, variables = set(), []
        for projection in projections:
            var = _projection_var(projection)
            if var is not None and id(var) not in seen:
                seen.add(id(var))
                variables.append(var)
        return tuple(variables)

    # --- pipeline verbs ------------------------------------------------------

    def where(self, *conditions):
        self.conditions.extend(conditions)
        return self

    def rank(self, *strategies):
        # a second rank(...) pops to the outer level (see groupby)
        if self.scopes[self._current].rank:
            self._current = max(0, self._current - 1)
        self.scopes[self._current].rank = [self._as_strategy(s) for s in strategies]
        return self

    def top(self, n):
        if self.scopes[self._current].limit is not None:
            self._current = max(0, self._current - 1)
        self.scopes[self._current].limit = n
        return self

    def groupby(self, var):
        self.group_var = var
        self.scopes.append(_Scope())
        self._current = len(self.scopes) - 1
        return self

    def using(self, engine):
        self.engine = engine
        return self

    @staticmethod
    def _as_strategy(s):
        return s if isinstance(s, Strategy) else ExprStrategy(s)

    def _soft_signals(self):
        """The 'soft where': any where-condition that is also a Strategy (a Like)
        contributes its graded membership as a ranking signal, on top of the
        widened crisp admission its compile() already emits."""
        return [c for c in self.conditions if isinstance(c, Strategy)]

    # --- lazy result protocol ------------------------------------------------

    def __iter__(self):
        return iter(self._ensure_run())

    def __len__(self):
        return len(self._ensure_run())

    def __getitem__(self, i):
        return self._ensure_run()[i]

    def __repr__(self):
        return repr(self._ensure_run())

    def _ensure_run(self):
        if self._results is None:
            self._results = self.run()
        return self._results

    # --- execution -----------------------------------------------------------

    def run(self):
        self._row_scores = {}
        self._fit_strategies()

        ors = [c for c in self.conditions if isinstance(c, Or)]
        if ors:
            rows = self._run_or(ors[0])
        else:
            spec = self._build_spec(self.conditions)
            result = self.engine.search(self.graph, spec, self._guide())
            self.visited_nodes = result.visited
            rows = result.rows

        rows = self._apply_antijoins(rows)
        rows = self._dedupe_hidden(rows)
        rows = self._rank_and_limit(rows)
        return self._render(rows)

    def _run_or(self, or_condition):
        shared = [c for c in self.conditions if not isinstance(c, Or)]
        rows, visited = set(), set()
        for branch in or_condition.operands:
            spec = self._build_spec(shared + [branch])
            result = self.engine.search(self.graph, spec, self._guide())
            visited |= result.visited
            rows |= set(result.rows)
        self.visited_nodes = visited
        return sorted(rows)

    def _all_strategies(self):
        seen, out = set(), []
        for scope in self.scopes:
            for s in scope.rank:
                if id(s) not in seen:
                    seen.add(id(s)); out.append(s)
        for s in self._soft_signals():
            if id(s) not in seen:
                seen.add(id(s)); out.append(s)
        return out

    def _fit_strategies(self):
        for s in self._all_strategies():
            s.fit(self.graph)

    def _guide(self):
        for scope in reversed(self.scopes):
            for s in scope.rank:
                if s.supports_guidance:
                    return s
        for s in self._soft_signals():
            if s.supports_guidance:
                return s
        return None

    # --- compilation ---------------------------------------------------------

    def _build_spec(self, conditions):
        compiler = _Compiler(self)
        spec = compiler.build(conditions)
        # carry the compile-time artifacts the rest of run() needs
        self._columns = compiler.columns
        self._antijoins = compiler.antijoins
        self._visible_count = compiler.visible_count
        self._has_hidden = compiler.has_hidden
        self._paths = compiler.paths
        return spec

    # --- post-search filters -------------------------------------------------

    def _apply_antijoins(self, rows):
        if not self._antijoins:
            return rows
        graph = self.graph
        checks = []
        for source, relations, target, reverse in self._antijoins:
            codes = [graph.relation_code(r) for r in relations] or [None]
            checks.append((self._columns[id(source)], self._columns[id(target)], codes, reverse))
        kept = []
        for row in rows:
            for src_col, tgt_col, codes, reverse in checks:
                source, target = row[src_col], row[tgt_col]
                if any(graph.has_edge(source, target, code, reverse) for code in codes):
                    break                       # anti-join: the edge must NOT exist
            else:
                kept.append(row)
        return kept

    def _dedupe_hidden(self, rows):
        if not self._has_hidden:
            return rows
        seen = {}
        for row in rows:
            seen.setdefault(row[:self._visible_count], row)
        return list(seen.values())

    # --- ranking / grouping --------------------------------------------------

    def _outer_rank(self):
        # the outer scope's strategies plus any soft-where signal not already there
        rank = list(self.scopes[0].rank)
        for s in self._soft_signals():
            if s not in rank:
                rank.append(s)
        return rank

    def _rank_and_limit(self, rows):
        if self.group_var is None:
            return self._apply_scope(self._outer_rank(), self.scopes[0].limit, rows)

        inner, outer = self.scopes[-1], self.scopes[0]
        group_col = self._columns[id(self.group_var)]
        groups, order = {}, []
        for row in rows:
            key = row[group_col]
            if key not in groups:
                groups[key] = []; order.append(key)
            groups[key].append(row)
        for key in order:
            groups[key] = self._apply_scope(inner.rank, inner.limit, groups[key])

        outer_rank = self._outer_rank()
        if outer_rank:
            reps = [groups[k][0] for k in order if groups[k]]
            keys = [k for k in order if groups[k]]
            scores = self._combine_scores(outer_rank, reps)
            order = [k for _, k in sorted(zip(scores, keys), key=lambda x: x[0], reverse=True)]
        if outer.limit is not None:
            order = order[:outer.limit]

        result = []
        for key in order:
            result.extend(groups[key])
        return result

    def _apply_scope(self, rank, limit, rows):
        if rank:
            scores = self._combine_scores(rank, rows)
            for row, score in zip(rows, scores):
                self._row_scores[row] = score
            rows = [row for _, row in sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def _combine_scores(self, strategies, rows):
        # each strategy min-max normalized, then averaged: a soft signal rewards
        # rather than hard-filters, and the combined score stays in [0, 1] however
        # many signals are stacked
        combined = [0.0] * len(rows)
        if not strategies:
            return combined
        for strategy in strategies:
            scores = strategy.score(self, rows)
            low, high = (min(scores), max(scores)) if scores else (0.0, 0.0)
            span = (high - low) or 1.0
            for i, value in enumerate(scores):
                combined[i] += (value - low) / span
        return [total / len(strategies) for total in combined]

    # --- rendering -----------------------------------------------------------

    def _render(self, rows):
        """The one place integers become Key/Rel: everything upstream stays
        numeric so it can index arrays directly."""
        if not self.projections:
            graph = self.graph
            return [graph.key(row[0]) if len(row) == 1 else tuple(map(graph.key, row))
                    for row in rows]
        extractors = [self._extractor(p) for p in self.projections]
        out = []
        for row in rows:
            values = tuple(extract(row) for extract in extractors)
            out.append(values[0] if len(values) == 1 else values)
        return out

    def _path_value(self, walk):
        """A path row (node, code, node, ...) as alternating Key and Rel."""
        graph = self.graph
        return tuple(graph.key(step) if i % 2 == 0 else _rel(graph, step)
                     for i, step in enumerate(walk))

    def _extractor(self, projection):
        graph = self.graph
        if isinstance(projection, Score):
            return lambda row: self._row_scores.get(row, 0.0)
        if isinstance(projection, Attr):
            col, name = self._columns[id(projection.node)], projection.name
            return lambda row: graph.value(row[col], name)
        if isinstance(projection, Degree):
            col = self._columns[id(projection.node)]
            degrees = graph.degree(projection.relation, projection.reverse)
            return lambda row: int(degrees[row[col]])
        if isinstance(projection, Node):
            col = self._columns[id(projection)]
            return lambda row: graph.key(row[col])
        if isinstance(projection, Path):
            col = self._columns[id(projection)]
            return lambda row: self._path_value(row[col])
        if isinstance(projection, PathStep):
            col, idx = self._columns[id(projection.path)], (0 if projection.index == 0 else -1)
            return lambda row: graph.key(row[col][idx])
        if isinstance(projection, PathStepAttr):
            col = self._columns[id(projection.path)]
            idx = 0 if projection.index == 0 else -1
            name = projection.name
            return lambda row: graph.value(row[col][idx], name)
        raise TypeError(f"unsupported projection: {projection!r}")


def _rel(graph, code):
    """A relation code back into a named, directed Rel. Reverse traversals are
    stored as ~code, so one integer carries both the relation and its direction."""
    if code is None:
        return None
    return Rel(graph.relations[~code], True) if code < 0 else Rel(graph.relations[code], False)


class _Compiler(Compiler):
    """The Query-owned implementation of core.Compiler.

    Holds the mutable pattern under construction ({var: NodeSpec}, edges) and the
    var-allocation table keyed by Node identity, so two conditions naming the same
    Node share a variable. Every graph/schema decision the Conditions defer lives
    here -- including, now, evaluating each constraint against the graph's columns
    and intersecting it into the variable's admission mask.
    """

    def __init__(self, query, negate=False):
        self.query = query
        self.graph = query.graph
        self._nodes = {}          # var -> NodeSpec
        self._edges = []          # [EdgeSpec]
        self._var_of = {}         # id(Node) -> var
        self._node_of = {}        # var -> Node, for diagnostics only
        self._counter = 0
        self._negate = negate
        # results the Query reads back after build()
        self.columns = dict(query._columns)   # id(ref) -> column (seed with visible)
        self.antijoins = []                   # [(source Node, relations, target Node, reverse)]
        self.paths = {}                       # id(Path) -> (seq vars, [EdgeSpec])
        self.visible_count = len(query.nodes)
        self.has_hidden = False

    # -- entry point --

    def build(self, conditions):
        # pass 1: give every selected Node its variable before emitting relations,
        # so a select var reused as a relation target keeps one variable (a join,
        # not a cross product). Also expand Node(**attrs) sugar.
        for var in self.query.nodes:
            if isinstance(var, Node):
                self._bind_node(var)
        # pass 2: each condition describes itself
        for condition in conditions:
            condition.compile(self)
        # pass 3: antijoin endpoints not selected become hidden columns so the
        # post-search filter can read them
        for source, _relations, target, _reverse in self.antijoins:
            for node in (source, target):
                self._ensure_column(node)
        self._reject_disconnected()
        return SearchSpec(self._nodes, self._edges, self._build_outputs())

    def _reject_disconnected(self):
        """Refuse a variable that never joins the selected pattern.

        Such a variable does not filter the result -- it multiplies it, once per
        binding, or empties it when it has none. That is an existential
        quantifier, and it is almost never what was meant. What was meant, nearly
        every time, is that two identical-looking Nodes were the same variable:

            g.select(Node("person")).where(Node("person").name == "Ada")

        Refs have identity semantics on purpose, so those are two variables and
        the constraint lands on the one nobody selected -- silently returning
        every person. Failing here is the difference between a typo and a wrong
        answer that looks plausible.
        """
        projected = set()
        for var in self.query.nodes:
            if isinstance(var, Node):
                projected.add(self.var(var))
            else:                                   # a Path: its sequence variables
                sequence, _edges = self.paths.get(id(var), ([], []))
                projected.update(sequence)
        if not projected:
            return

        joined = {}
        def link(one, other):
            joined.setdefault(one, set()).add(other)
            joined.setdefault(other, set()).add(one)
        for edge in self._edges:
            link(edge.source, edge.target)
        for source, _relations, target, _reverse in self.antijoins:
            # an anti-join is a join too: it constrains a pair, so an endpoint
            # reached only this way is connected, not orphaned
            link(self.var(source), self.var(target))

        reached, queue = set(projected), list(projected)
        while queue:
            for other in joined.get(queue.pop(), ()):
                if other not in reached:
                    reached.add(other)
                    queue.append(other)

        orphans = [var for var in self._nodes if var not in reached]
        if orphans:
            described = ", ".join(sorted(self._describe(var) for var in orphans))
            raise ValueError(
                f"{described} is constrained but never joined to the selected pattern, "
                f"so it would multiply the results rather than filter them. Two "
                f"identical-looking Nodes are two different variables: reuse the same "
                f"object in select() and where(), or relate them with an edge."
            )

    def _describe(self, var):
        node = self._node_of.get(var)
        if node is None or node.type is None:
            return "Node()"
        return f'Node("{node.type}")'

    # -- Compiler surface the Condition family calls --

    def constrain(self, expr, op, value):
        if isinstance(expr, Degree):
            spec = self.spec(expr.node)
            degrees = self.graph.degree(expr.relation, expr.reverse)
            self._restrict(spec, column_mask(Column(degrees), op, value))
            return
        spec = self.spec(expr.node)                # Attr
        self._restrict(spec, self._attr_mask(spec, expr.name, op, value))

    def member_of(self, ref, values):
        if isinstance(ref, Node):                  # identity membership
            spec = self.spec(ref)
            mask = np.zeros(self.graph.n_nodes, dtype=bool)
            for value in values:
                index = self.graph.lookup(value)
                if index is not None:
                    mask[index] = True
            self._restrict(spec, mask)
            return
        spec = self.spec(ref.node)                 # attribute membership
        self._restrict(spec, self._attr_mask(spec, ref.name, "in", set(values)))

    def edge(self, source, relation, target, reverse=False):
        relations = self._relation_tuple(relation)
        if self._negate:                           # anti-join: applied after the search
            self.antijoins.append((source, relations, target, reverse))
            self.spec(source); self.spec(target)
            return
        self.spec(source); self.spec(target)
        name = relations[0] if len(relations) == 1 else None   # single relation or wildcard
        self._edges.append(EdgeSpec(self.var(source), self._code(name),
                                    self.var(target), self._direction(name, reverse)))

    def path(self, path_ref, pattern):
        items = self._normalize_pattern(pattern)
        seq = [self.var(item) for item in items if isinstance(item, Node)]
        for item in items:
            if isinstance(item, Node):
                self._bind_node(item)
        path_edges = []
        for i in range(0, len(items) - 2, 2):
            source, edge, target = items[i], items[i + 1], items[i + 2]
            code, reverse = self._edge_relation(edge)
            spec = EdgeSpec(self.var(source), code, self.var(target), reverse)
            self._edges.append(spec)
            path_edges.append(spec)
        self.paths[id(path_ref)] = (seq, path_edges)

    def schema(self):
        return self.graph.schema()

    def negated(self):
        clone = _Compiler.__new__(_Compiler)
        clone.__dict__.update(self.__dict__)
        clone._negate = not self._negate
        return clone

    # -- mask construction: where a described constraint meets the data --

    def _restrict(self, spec, mask):
        """Intersect a constraint into the variable's admission set. Negation is
        a set difference, so Not needs no per-condition support."""
        spec.mask &= ~mask if self._negate else mask

    def _attr_mask(self, spec, name, op, value):
        """A predicate on a named attribute, over whichever type blocks can have
        it. An untyped variable is tested against every type that has a column of
        that name -- the graph-wide reading the old per-node attrs dict gave."""
        graph = self.graph
        mask = np.zeros(graph.n_nodes, dtype=bool)
        types = [spec.type] if spec.type is not None else graph.types
        for type_ in types:
            column = graph.column(type_, name)
            if column is None:
                continue
            low, high = graph.block(type_)
            mask[low:high] = column_mask(column, op, value)
        return mask

    def _type_mask(self, type_):
        mask = np.zeros(self.graph.n_nodes, dtype=bool)
        if type_ is None:
            mask[:] = True
        else:
            low, high = self.graph.block(type_)
            mask[low:high] = True
        return mask

    def _code(self, name):
        """A relation name as a code. An unknown name gets -1, which no stored
        edge carries, so the pattern simply matches nothing."""
        if name is None:
            return None
        code = self.graph.relation_code(name)
        return -1 if code is None else code

    def _direction(self, name, reverse):
        """Which way to walk: True backwards, False forwards, None either.

        A named relation reads forwards by default (`movie.directed_by` goes
        movie -> person) and `.inverse` flips it. The wildcard has no natural
        direction, so it walks both -- which is what a bridge pattern like
        [movie, Edge(), genre, Edge(), movie] needs, and precisely what the old
        `rv=True` edge duplication existed to provide."""
        if reverse:
            return True
        return False if name is not None else None

    # -- internals --

    def _fresh(self):
        self._counter += 1
        return f"_v{self._counter}"

    def var(self, node):
        if id(node) not in self._var_of:
            var = self._var_of[id(node)] = self._fresh()
            self._node_of[var] = node        # so a diagnostic can name what was written
        return self._var_of[id(node)]

    def spec(self, node):
        return self._bind_node(node)

    def _bind_node(self, node):
        var = self.var(node)
        spec = self._nodes.get(var)
        if spec is None:
            spec = NodeSpec(var, type=node.type, mask=self._type_mask(node.type))
            self._nodes[var] = spec
            for name, value in getattr(node, "attrs", {}).items():   # Node(**attrs) sugar
                spec.mask &= self._attr_mask(spec, name, "eq", value)
        return spec

    def _ensure_column(self, node):
        if id(node) in self.columns:
            return
        self.spec(node)
        self.columns[id(node)] = None    # placeholder; real index assigned in _build_outputs
        self.has_hidden = True

    def _relation_tuple(self, relation):
        if relation is None:
            return ()
        if isinstance(relation, str):
            return (relation,)
        return tuple(relation)

    def _edge_relation(self, edge):
        if isinstance(edge, Edge):
            if len(edge.relations) > 1:
                raise ValueError("a path edge may carry at most one relation")
            name = edge.relations[0] if edge.relations else None
            return self._code(name), self._direction(name, edge.reverse)
        return None, None

    def _normalize_pattern(self, pattern):
        # insert a wildcard Edge() between two consecutive Nodes with no explicit edge
        normalized = [pattern[0]]
        for item in pattern[1:]:
            if isinstance(item, Node) and isinstance(normalized[-1], Node):
                normalized.append(Edge())
            normalized.append(item)
        return normalized

    def _build_outputs(self):
        outputs = [None] * self.visible_count
        for var in self.query.nodes:
            col = self.columns[id(var)]
            if isinstance(var, Node):
                outputs[col] = Output("node", self.var(var))
            else:                                   # Path
                seq, edges = self.paths.get(id(var), ([], []))
                outputs[col] = Output("path", seq, edges=edges)
        # hidden columns (antijoin endpoints), appended after the visible ones
        column = self.visible_count
        for node_id, col in self.columns.items():
            if col is None:
                self.columns[node_id] = column
                var = self._var_of[node_id]
                outputs.append(Output("node", var))
                column += 1
        return outputs
