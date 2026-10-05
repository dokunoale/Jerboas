"""Deferring a walk, so its cost has a ceiling.

    with jb.optimize():
        frame = seeds.hop(rec=()).filter(v.rec.type == "movie")

A hop expands a frame by the degree of what it walks, and the expansion happens
before any filter can reduce it. Inside `optimize` a hop describes itself
instead, the filters written after it join the description, and the whole runs
when something reads the frame -- in slices, and from whichever end is cheaper.
Same answer, and the peak is a slice instead of the lot.

The budget is on what a step *produces*, and the size of that is exact rather
than estimated: the graph knows every degree, so an expansion's size is
`degree[nodes].sum()` before a step is taken. Slices are cut where that running
total crosses the budget, so a walk out of a hub takes a shorter slice than one
out of a leaf, and the budget applies again at every step -- which is why
`hop(a=..., b=...)` and `hop(a=...).hop(b=...)` cost the same.

The budget is said one of two ways:

    jb.optimize()                  # half the memory that is free right now
    jb.optimize(memory=2 << 30)    # two gigabytes
    jb.optimize(rows=5_000_000)    # five million rows a step, however wide

Memory is turned into rows per step and per frame, since a row of a frame
carrying fourteen columns costs more than one carrying two. `rows` skips the
conversion, for a caller who has measured.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

# None outside a context: a hop takes place where it is written, which is the
# behaviour that is easy to reason about and the one worth defaulting to.
_BUDGET = ContextVar("jerboas_budget", default=None)

# What one produced row costs beyond the frame's own width, at the peak of a
# step: the walk's parallel arrays (row, position, target, code, weight, and the
# cumulative sum `ranges` builds) and, per named step, the id, confidence and
# relation the new rows carry -- once as arrays and once as the frame built
# from them.
WALK_BYTES = 48
STEP_BYTES = 2 * (4 + 8 + 8)


class Budget:
    """How much one step may produce at a time: a number of rows, or a number
    of bytes that becomes one once the frame's width is known."""

    __slots__ = ("fixed", "memory")

    def __init__(self, rows=None, memory=None):
        if rows is not None and memory is not None:
            raise ValueError("say the budget as rows or as memory, not both")
        if rows is not None and rows < 1:
            raise ValueError(f"the row budget must be at least one row, got {rows}")
        if memory is not None and memory < 1:
            raise ValueError(f"the memory budget must be at least one byte, got {memory}")
        if rows is None and memory is None:
            memory = available_memory() // 2
        self.fixed, self.memory = rows, memory

    def rows(self, frame=None, steps=()):
        """Rows one step out of `frame` may produce: the fixed number, or what
        fits in the memory once each row is priced at the frame's width."""
        if self.fixed is not None:
            return self.fixed
        return max(1, self.memory // row_bytes(frame, steps))

    def __repr__(self):
        return (f"Budget(rows={self.fixed})" if self.fixed is not None
                else f"Budget(memory={self.memory})")


@contextmanager
def optimize(rows=None, memory=None):
    """Plan the walks written inside: deferred, sliced, and walked from the
    cheaper end. `rows` or `memory` bounds one slice; neither means half the
    memory free when the context opens."""
    token = _BUDGET.set(Budget(rows, memory))
    try:
        yield
    finally:
        _BUDGET.reset(token)


def budget():
    """The budget in force, or None outside `optimize`."""
    return _BUDGET.get()


def row_budget():
    """How many rows one step may produce at a time when that is a fixed
    number, None otherwise."""
    current = _BUDGET.get()
    return None if current is None else current.fixed


def row_bytes(frame, steps=()):
    """What one row produced by a step out of `frame` costs at the peak: the
    frame's own row, taken again for every edge, plus what the walk carries."""
    width = 0
    if frame is not None:
        data = frame._df
        if data.height:
            width = data.estimated_size() // data.height
        else:
            width = 8 * len(data.columns)
    named = sum(1 for _spec, name in steps if name is not None) or 1
    return width + WALK_BYTES + STEP_BYTES * named


def available_memory():
    """Bytes free right now, as well as the platform will say.

    Linux says it in /proc/meminfo; elsewhere `sysconf` may know the free pages;
    failing both, half the physical memory is the guess -- a machine is rarely
    idler than that."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    page = os.sysconf("SC_PAGE_SIZE")
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * page
    except (ValueError, OSError):
        return os.sysconf("SC_PHYS_PAGES") * page // 2
