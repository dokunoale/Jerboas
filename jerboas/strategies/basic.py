"""The projection marker and the two strategies with no model behind them.

Score is not a Strategy: it reads back what rank(...) computed. ExprStrategy
adapts a bare scalar Expr (a Degree) so rank(expr) works with no wrapping.
"""

from ..core import Strategy


class Score:
    """A projection marker: select(rec, Score()) yields the row's final ranking
    score. Not a Strategy -- it reads back what rank(...) computed."""


class ExprStrategy(Strategy):
    """Adapts a scalar Expr (currently a Degree) into a Strategy: scores each row
    by evaluating the expression against its binding."""

    def __init__(self, expr):
        self.expr = expr        # a Degree: has .node, .relation, .reverse

    def score(self, query, rows):
        col = query._columns.get(id(self.expr.node), query.primary_column)
        degrees = query.graph.degree(self.expr.relation, self.expr.reverse)
        return [float(degrees[row[col]]) for row in rows]


class Alphabetical(Strategy):
    """Rank rows by the label of their primary node, A to Z."""

    def score(self, query, rows):
        graph, col = query.graph, query.primary_column
        labels = [str(graph.label_of(row[col])) for row in rows]
        order = sorted(range(len(rows)), key=lambda i: labels[i])
        scores = [0.0] * len(rows)
        for position, i in enumerate(order):
            scores[i] = float(-position)
        return scores
