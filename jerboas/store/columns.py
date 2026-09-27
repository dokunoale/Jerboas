"""Typed attribute columns.

A column is typed once at load instead of every value staying text. That is what
lets `year >= 1990` be one array comparison rather than a per-candidate string
coercion. Reading one into a frame is a gather at its local ids; comparing it
is polars' business from there.

Absence is carried by an explicit `present` mask rather than by NaN. NaN would
force an integer column to float, and a caller asking for a movie's year would
get 1995.0 -- a loader artifact leaking into their output. Gaps are normal: a
node can appear in the graph and have no row in any attribute file.
"""

import numpy as np


class Column:
    """One attribute, for every node of one type, indexed by local id."""

    __slots__ = ("values", "present")

    def __init__(self, values, present=None):
        self.values = values      # ndarray: int64, float64 or object
        self.present = present    # bool ndarray, or None when the column is complete

    def __len__(self):
        return len(self.values)

    def get(self, index):
        if self.present is not None and not self.present[index]:
            return None
        return _plain(self.values[index])


def build(values):
    """A raw text column (None where absent) as the narrowest type that holds it."""
    present = None
    if any(value is None for value in values):
        present = np.array([value is not None for value in values], dtype=bool)
    filled = [value for value in values if value is not None]

    for parse, dtype in ((int, np.int64), (float, np.float64)):
        try:
            numbers = [parse(value) for value in filled]
        except ValueError:
            continue
        typed = np.zeros(len(values), dtype=dtype)
        typed[slice(None) if present is None else present] = numbers
        return Column(typed, present)

    return Column(np.array(values, dtype=object))


def _plain(value):
    """A stored cell as a plain Python value, not a numpy scalar."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
