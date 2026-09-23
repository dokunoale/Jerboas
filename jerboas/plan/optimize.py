"""Deferring a walk, so its cost has a ceiling.

    with jb.optimize(rows=5_000_000):
        frame = seeds.hop(rec=()).filter(v.rec.type == "movie")

A hop expands a frame by the degree of what it walks, and the expansion happens
before any filter can reduce it. Inside `optimize` a hop describes itself
instead, the filters written after it join the description, and the whole runs
when something reads the frame -- in slices. Same answer, and the peak is a
slice instead of the lot.

`rows` budgets what a step *produces*, and it is exact rather than estimated:
the graph knows every degree, so an expansion's size is `degree[nodes].sum()`
before a step is taken. Slices are cut where that running total crosses the
budget, so a walk out of a hub takes a shorter slice than one out of a leaf, a
walk that fits is not deferred, and the budget applies again at every step --
which is why `hop(a=..., b=...)` and `hop(a=...).hop(b=...)` cost the same.

Two things follow from working in slices: a predicate that aggregates sees its
slice, and the answer is accumulated rather than streamed, so a query whose
result does not fit is not helped.
"""

from contextlib import contextmanager
from contextvars import ContextVar

# None outside a context: a hop takes place where it is written, which is the
# behaviour that is easy to reason about and the one worth defaulting to.
_ROWS = ContextVar("jerboas_rows", default=None)

DEFAULT_ROWS = 5_000_000


@contextmanager
def optimize(rows=DEFAULT_ROWS):
    """Run a walk in slices, each cut so one step produces about `rows` rows."""
    if rows is not None and rows < 1:
        raise ValueError(f"the row budget must be at least one row, got {rows}")
    token = _ROWS.set(rows)
    try:
        yield
    finally:
        _ROWS.reset(token)


def row_budget():
    """How many rows one step may produce at a time, or None for all of them."""
    return _ROWS.get()
