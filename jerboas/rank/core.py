"""The one interface left: Strategy, and the signal it produces.

A predicate is a polars expression, a traversal is a verb, an aggregate is a
group_by. What none of those can be is a score computed *from the graph* -- a
walk, a factorization, an embedding -- and that is what a Strategy is.

    class PageRank(Strategy):
        def fit(self, graph): ...                  # once, before scoring
        def scores(self, graph, columns): ...      # id arrays in, floats out

`columns` is the tuple of node-id arrays named in `on(...)`, in that order: the
first is what is scored, the rest are context (whose taste it is, which seed it
was reached from).

    frame.with_columns(pr=PageRank(to=seeds).on("rec").norm())

`on` returns a Signal: a strategy bound to columns, not yet to a frame. The
frame resolves it, because the frame is the thing that knows the graph.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..query.expr import Expr


class Signal(Expr):
    """A strategy aimed at named columns. Passive: the Frame computes it.

    An expression like any other, so a score is combined by arithmetic that is
    written down -- `0.7 * kg.norm() + 0.3 * walk.norm()` -- rather than by a
    rule the pipeline applies out of sight."""

    __slots__ = ("strategy", "columns", "normalized")

    def __init__(self, strategy, columns, normalized=False):
        self.strategy = strategy
        self.columns = tuple(columns)
        self.normalized = normalized

    def norm(self):
        """Min-max into [0, 1], so signals on different scales can be added."""
        return Signal(self.strategy, self.columns, normalized=True)

    def resolve(self, ctx):
        return ctx.signal(self)

    def reads(self):
        return ()          # it reads node ids, which no filter can be pushed past

    def values(self, graph, arrays):
        self.strategy.fit(graph)
        scores = np.asarray(self.strategy.scores(graph, arrays), dtype=np.float64)
        if not self.normalized or not len(scores):
            return scores
        low, high = float(scores.min()), float(scores.max())
        span = high - low
        return np.ones(len(scores)) if span == 0 else (scores - low) / span

    def __repr__(self):
        return (f"{type(self.strategy).__name__}.on{self.columns}"
                f"{'.norm()' if self.normalized else ''}")


class Strategy(ABC):
    """A score computed from the graph, one float per row.

    This is the "non-deterministic-first" half of the library: a filter keeps or
    drops a row, a strategy says how much it is worth, and several combine by
    plain arithmetic on the columns they produce -- explicitly weighted, because
    how much each signal counts is a question only the caller can answer.
    """

    def on(self, *columns):
        """Aim this strategy at the frame's columns. The first is what is being
        scored; the rest are context, and which is which is the strategy's own
        documented reading."""
        if not columns:
            raise TypeError(f"{type(self).__name__}.on() needs the column being scored")
        return Signal(self, [str(c) for c in columns])

    def fit(self, graph):
        """Called once per frame before scoring. Override to precompute
        graph-derived state; must be idempotent. Default: no-op."""

    @abstractmethod
    def scores(self, graph, columns):
        """One float per row, in row order. `columns` is a tuple of int arrays,
        one per name given to `on(...)`."""

    def cached(self, graph, key, builder):
        """Memoize for (this strategy instance, this graph, key). Lives on the
        strategy because the result usually depends on its hyperparameters."""
        cache = self.__dict__.setdefault("_strategy_cache", {})
        cache_key = (id(graph), key)
        if cache_key not in cache:
            cache[cache_key] = builder()
        return cache[cache_key]
