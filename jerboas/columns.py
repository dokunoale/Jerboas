"""Typed attribute columns, and the masks you get from them.

A column is typed once at load instead of every value staying text. That is what
lets `year >= 1990` be one array comparison rather than a per-candidate string
coercion -- and it removes a class of bug the text-only loader had, where
`year == 1994` and `year == '1994'` disagreed unless every operator remembered
to coerce.

Absence is carried by an explicit `present` mask rather than by NaN. NaN would
force an integer column to float, and a caller asking for a movie's year would
get 1995.0 -- a loader artifact leaking into their output. Gaps are normal: a
node can appear in the graph and have no row in any attribute file.
"""

import operator

import numpy as np

_ORDER = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


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


def column_mask(column, op, value):
    """Which rows of `column` satisfy `<op> value`, as a boolean array."""
    values = column.values
    result = (_text_mask(values, op, value) if values.dtype == object
              else _numeric_mask(values, op, value))
    if column.present is None:
        return result
    # an absent value equals nothing, and differs from everything
    return (result | ~column.present) if op == "ne" else (result & column.present)


def _numeric_mask(column, op, value):
    if op == "in":
        numbers = [n for n in (_number(v) for v in value) if n is not None]
        return np.isin(column, numbers) if numbers else np.zeros(len(column), bool)
    if op in ("contains", "contains_any"):
        return _text_mask(column.astype(object), op, value)
    number = _number(value)
    if number is None:
        # nothing numeric equals a non-numeric value -- and everything differs
        return np.ones(len(column), bool) if op == "ne" else np.zeros(len(column), bool)
    if op == "eq":
        return column == number
    if op == "ne":
        return column != number
    handler = _ORDER.get(op)
    return handler(column, number) if handler else np.zeros(len(column), bool)


def _text_mask(column, op, value):
    size = len(column)
    if op == "in":
        allowed = {_text(v) for v in value}
        return np.fromiter((_text(v) in allowed for v in column), bool, size)
    if op == "contains":
        needle = str(value).lower()
        return np.fromiter((v is not None and needle in str(v).lower() for v in column), bool, size)
    if op == "contains_any":
        needles = [str(v).lower() for v in value]
        return np.fromiter(
            (v is not None and any(n in str(v).lower() for n in needles) for v in column),
            bool, size)
    if op in ("eq", "ne"):
        wanted = _text(value)
        match = operator.eq if op == "eq" else operator.ne
        return np.fromiter((match(_text(v), wanted) for v in column), bool, size)
    handler = _ORDER.get(op)
    if handler is None:
        return np.zeros(size, bool)
    wanted = str(value)
    return np.fromiter((v is not None and handler(str(v), wanted) for v in column), bool, size)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value):
    return None if value is None else str(value)


def _plain(value):
    """A stored cell as a plain Python value, not a numpy scalar."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
