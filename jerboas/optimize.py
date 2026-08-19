"""Deferring a walk, so its cost has a ceiling.

A hop expands a frame by the degree of what it walks, and the expansion happens
before any filter can reduce it. On a small frame that is nothing; on a large one
it is the whole problem -- a two-hop wildcard over a big graph builds hundreds of
millions of rows on the way to a few thousand.

Inside `optimize`, a hop describes itself instead of taking place. The filters
written after it join the description, and the whole of it runs when something
finally reads the frame -- a batch of source rows at a time, each batch walked,
filtered and reduced before the next one starts. The answer is the same; the
peak is a batch instead of the lot.

    with jb.optimize(batch=100_000):
        frame = seeds.hop(rec=()).filter(v.rec.type == "movie")

This is a peephole with a ceiling, not a planner. It defers exactly one hop and
the conditions about the node it reaches; anything else runs the walk first. Two
things follow from batching that are worth knowing: a predicate that aggregates
(`>= v.rec.score.mean()`) sees its batch rather than the whole result, and row
order is the batches' order, which is the frame's own.
"""

from contextlib import contextmanager
from contextvars import ContextVar

# None outside a context: a hop takes place where it is written, which is the
# behaviour that is easy to reason about and the one worth defaulting to.
_BATCH = ContextVar("jerboas_batch", default=None)

DEFAULT_BATCH = 100_000


@contextmanager
def optimize(batch=DEFAULT_BATCH):
    """Defer walks and run them a batch of source rows at a time."""
    if batch is not None and batch < 1:
        raise ValueError(f"batch must be at least one row, got {batch}")
    token = _BATCH.set(batch)
    try:
        yield
    finally:
        _BATCH.reset(token)


def batch_size():
    """How many source rows a deferred walk takes at once, or None for all."""
    return _BATCH.get()
