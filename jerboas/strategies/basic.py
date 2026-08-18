"""The projection marker and the two strategies with no model behind them.

Score is not a Strategy: it reads back what rank(...) computed. ExprStrategy
adapts a bare scalar Expr -- a Degree or an Aggregate -- so rank(expr) works
with no wrapping.
"""

from collections import defaultdict

from ..core import Strategy
from ..refs import Aggregate, Attr
from ..ir import aliased

# how the values of one group become one number
FOLD = {
    "sum": lambda values: float(sum(values)),
    "count": lambda values: float(len(values)),
    "mean": lambda values: float(sum(values) / len(values)),
    "min": lambda values: float(min(values)),
    "max": lambda values: float(max(values)),
}


class Score:
    """A projection marker: select(rec, Score()) yields the row's final ranking
    score. Not a Strategy -- it reads back what rank(...) computed."""


class ExprStrategy(Strategy):
    """Adapts a scalar Expr into a Strategy.

    A Degree is a fact about the graph, true whatever the query asked, so it is
    read once per row. An Aggregate is a fact about the *matches*, so it is read
    once per group of rows -- and every row of a group is given the group's
    value, which is what lets the Query collapse them afterwards without having
    to choose between them.
    """

    def __init__(self, expr):
        self.expr = expr        # a Degree (has .node, .relation, .reverse) or an Aggregate

    def score(self, query, rows):
        if isinstance(self.expr, Aggregate):
            return self._folded(query, rows)
        if isinstance(self.expr, Attr):
            return self._column(query, rows)
        col = query._columns.get(id(self.expr.node), query.primary_column)
        degrees = query.graph.degree(self.expr.relation, self.expr.reverse)
        return [float(degrees[row[col]]) for row in rows]

    def _column(self, query, rows):
        """A stored column as a ranking signal.

        A number ranks by its value, the way a degree does. Text has no value to
        rank by, so it ranks by its position among the values these rows hold --
        which is what lets `rank(Ascending(movie.title), PageRank(...))` combine
        at all, both sides being quantities by the time they are normalised."""
        col = query._columns.get(id(self.expr.node), query.primary_column)
        graph, name = query.graph, self.expr.name
        aliases = getattr(self.expr.node, "_aliases", None)
        values = [graph.value(row[col], aliased(aliases, name, graph.type_of(row[col])))
                  for row in rows]
        # the *present* values decide which kind of column this is: one node
        # without a year must not turn a numeric ranking into a lexicographic
        # one. A node with no value scores below every node that has one, so it
        # lands last -- and first under Ascending, which is what reversing means
        present = [value for value in values if value is not None]
        if all(isinstance(value, (int, float)) for value in present):
            floor = float(min(present)) - 1.0 if present else 0.0
            return [floor if value is None else float(value) for value in values]
        order = {text: position for position, text in enumerate(sorted(map(str, set(present))))}
        return [-1.0 if value is None else float(order[str(value)]) for value in values]

    def _folded(self, query, rows):
        plan = query._aggregates.get(id(self.expr))
        if plan is None or not rows:
            return [0.0] * len(rows)
        matched = defaultdict(dict)          # group -> {the match: what it is worth}
        for row in rows:
            key, value = self._match(query, plan, row)
            matched[query.group_key(row)][key] = value
        fold = FOLD[self.expr.how]
        folded = {group: fold(list(values.values())) for group, values in matched.items()}
        return [folded[query.group_key(row)] for row in rows]

    def _match(self, query, plan, row):
        """One match of the aggregated thing, as (what it is, what it is worth).

        Keyed by what it is, so a match that appears twice -- because some other
        variable of the pattern moved underneath it -- is still one match."""
        if plan[0] == "node":
            return row[query._columns[id(plan[1])]], 1.0
        _kind, source, target, relation, reverse, normalized = plan
        head, tail = row[query._columns[id(source)]], row[query._columns[id(target)]]
        return (head, tail), query.graph.weight_of(head, tail, relation, reverse, normalized)


class _Directed(Strategy):
    """Shared body of the two direction words."""

    sign = 1.0

    def __init__(self, expr):
        self.expr = expr
        # anything scoreable: a Degree, an Aggregate, a column -- or another
        # Strategy, since reversing one is the same operation
        self.inner = expr if isinstance(expr, Strategy) else ExprStrategy(expr)

    def score(self, query, rows):
        return [self.sign * value for value in self.inner.score(query, rows)]

    def __repr__(self):
        return f"{type(self).__name__}({self.expr!r})"


class Descending(_Directed):
    """Most first -- what `rank(expr)` already means, said out loud."""


class Ascending(_Directed):
    """Least first: `Ascending(movie.title)` is A to Z, `Ascending(movie.year)`
    is the oldest, `Ascending(node.rel.count())` the least connected.

    `rank(...)` scores rather than sorts, and a score means "more is better" --
    which a degree implies on its own and a column does not. So the direction is
    said here rather than guessed from what kind of column it is."""

    sign = -1.0
