"""What a plan may do with a condition, read off the condition itself.

Two questions, both about a predicate written after a deferred walk:

*Is it row-local?* A walk run in slices applies its conditions a slice at a
time, which is the same as applying them to the whole only when a row's verdict
depends on that row alone. `v.rec.year >= 1990` does; `v.rec.score >=
v.rec.score.mean()` does not, and neither does a `like` that keeps the k best
or a `norm` that rescales by the column's extremes. Those wait for the whole
answer. The test is an allowlist -- a node the planner does not recognise is
not local -- because the wrong answer here is a quiet one.

*Does it name where the walk must land?* `v.rec.is_in(wanted)` does, and when
the set costs less to walk than the frame, the walk is taken from the set's end
(query/traverse.py, `Reach`).
"""

import polars as pl

from ..query.expr import Col, Expr, _Binary, _Contains, _In, _Like, _Method, _Norm, _Unary

# Methods whose value on a row depends on other rows: aggregates, windows,
# ranks, and whatever reorders or drops rows inside an expression.
_AGGREGATING = frozenset({
    "sum", "count", "n_unique", "mean", "min", "max", "std", "first", "last",
    "sort_by", "unique", "head", "tail", "rank", "over", "any", "all",
})

# The same idea for a raw polars expression, which cannot be walked: its printed
# form names every method it calls, so a word from this list anywhere in it is
# reason enough to wait for the whole answer.
_POLARS_AGGREGATING = tuple(f".{name}(" for name in sorted(_AGGREGATING | {
    "median", "quantile", "var", "len", "cum_sum", "cum_count", "cum_max",
    "cum_min", "cum_prod", "shift", "diff", "rolling", "is_duplicated",
    "is_unique", "is_first_distinct", "is_last_distinct", "arg_max", "arg_min",
    "implode", "gather", "top_k", "bottom_k", "pct_change", "ewm_mean",
    "rolling_mean", "rolling_sum", "mode", "entropy", "value_counts",
}))


def row_local(predicate, graph):
    """Whether a row's verdict under `predicate` depends on that row alone."""
    if isinstance(predicate, pl.Expr):
        text = str(predicate)
        return not any(name in text for name in _POLARS_AGGREGATING)
    if not isinstance(predicate, Expr):
        return True                                   # a literal
    if isinstance(predicate, Col):
        return True
    if isinstance(predicate, _Binary):
        return row_local(predicate.left, graph) and row_local(predicate.right, graph)
    if isinstance(predicate, (_Unary,)):
        return row_local(predicate.operand, graph)
    if isinstance(predicate, (_In, _Contains)):
        return row_local(predicate.target, graph)
    if isinstance(predicate, (_Like, _Norm)):
        return False                 # the k best, the column's extremes: the whole
    if isinstance(predicate, _Method):
        if predicate.method in _AGGREGATING:
            # one exception: a relation's count is its arity, a fact about the
            # node rather than about the frame
            return predicate.method == "count" and _names_relation(predicate.target, graph)
        return (row_local(predicate.target, graph)
                and all(row_local(one, graph) for one in predicate.args)
                and all(row_local(one, graph) for one in predicate.kwargs.values()))
    signal = getattr(predicate, "strategy", None)
    if signal is not None:
        # a strategy scores each row from its own node ids -- unless it is
        # rescaled by the column's extremes, which are the whole frame's
        return not predicate.normalized
    return False


def roots(predicate):
    """The columns a predicate reads, by their root name, or None when it does
    not say (a raw polars expression, a strategy's ids)."""
    if isinstance(predicate, Expr):
        columns = getattr(predicate, "columns", None)
        if getattr(predicate, "strategy", None) is not None:
            return set(columns)
        paths = predicate.reads()
        return {path[0] for path in paths} if paths else None
    if isinstance(predicate, pl.Expr):
        names = predicate.meta.root_names()
        return {name.partition(".")[0] for name in names} if names else None
    return set()


def landing(predicates, target, graph):
    """The node set a walk to `target` is required to land in, or None.

    Only a membership written on its own -- a conjunct of the filter, over the
    column itself -- says that; one inside an `|` or behind a `~` does not."""
    for one in predicates:
        if isinstance(one, _Binary) and one.op == "__and__":
            found = landing([one.left, one.right], target, graph)
            if found is not None:
                return found
        if (isinstance(one, _In) and isinstance(one.target, Col)
                and one.target._path == (target,)):
            return graph.ids_of(one.values)
    return None


def _names_relation(target, graph):
    return (isinstance(target, Col) and len(target._path) >= 2
            and target._path[-1] in graph.relations)
