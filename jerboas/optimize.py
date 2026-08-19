"""Deferring a walk, so its cost has a ceiling.

A hop expands a frame by the degree of what it walks, and the expansion happens
before any filter can reduce it. On a small frame that is nothing; on a large one
it is the whole problem -- a two-hop wildcard over a big graph builds hundreds of
millions of rows on the way to a few thousand.

Inside `optimize`, a hop describes itself instead of taking place. The filters
written after it join the description, and the whole of it runs when something
reads the frame -- in slices, each walked, filtered and reduced before the next
one starts. The answer is the same; the peak is a slice instead of the lot.

    with jb.optimize(rows=5_000_000):
        frame = seeds.hop(rec=()).filter(v.rec.type == "movie")

`rows` is a budget on *what a step produces*, not on what it is given, and it is
not an estimate: the graph knows every node's degree, so the exact size of an
expansion is `degree[nodes].sum()` before a step is taken. The slices are cut
where that running total crosses the budget, which is why a walk out of a hub
takes a smaller slice than one out of a leaf, and why a walk that fits is not
deferred at all.

The budget is applied again at every step, so `hop(a=..., b=...)` and
`hop(a=...).hop(b=...)` cost the same -- the second hop of either is cut against
the middle it actually landed on.

Two things still follow from working in slices: a predicate that aggregates
(`>= v.rec.score.mean()`) sees its slice rather than the whole result, and the
surviving rows are accumulated rather than streamed, so a query whose *answer*
does not fit is not helped.
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
