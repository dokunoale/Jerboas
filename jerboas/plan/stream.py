"""Reductions that take a planned walk a slice at a time.

A walk planned inside `optimize` is a generator of slices (plan.py). Building
the whole of it and then reducing is what an eager frame does; when the
reduction can be taken per slice and then across slices, the whole never has to
exist:

    top(n)           the n best of the whole are among the n best of a slice
    unique()         the first of a row is the first in the first slice with it
    group_by().agg   sum, count, min, max, first, last and mean decompose

Each reduces a slice, sets it aside, and folds what is set aside whenever it
outgrows what was already folded -- so the peak is about twice the answer plus
one slice, and the work stays linear. A reduction that does not decompose (a
median, a `norm`, a sample) is not taken here: the frame builds the whole and
reduces it, which is slower and never wrong.
"""

import polars as pl

from ..query.expr import SCORE, Col, _Method, shadow
from ..query.frame import _resolved
from ..query.resolve import FOLD, Resolver
from .plan import stack
from .planner import _names_relation, row_local

# What one slice's partial result becomes across slices.
_MERGE = {"sum": "sum", "count": "sum", "min": "min", "max": "max",
          "first": "first", "last": "last"}
DECOMPOSABLE = frozenset(_MERGE) | {"mean"}


def top(frame, n, by, descending, over):
    def local(part):
        return part.top(n, by=by, descending=descending, over=over)

    def merge(parts):
        return joined(parts).top(n, by=by, descending=descending, over=over)

    return _reduce(frame.batches(), local, merge)


def unique(frame, subset, keep, maintain_order):
    def local(part):
        return part.unique(subset, keep=keep, maintain_order=maintain_order)

    def merge(parts):
        return joined(parts).unique(subset, keep=keep, maintain_order=maintain_order)

    return _reduce(frame.batches(), local, merge)


def agg(grouped, exprs, named, length=None):
    """`group_by(...).agg(...)` over a planned walk, a slice at a time -- or
    None when an aggregation does not decompose, for the frame to do whole."""
    frame, by = grouped.frame, grouped.by
    wanted = _decompose(exprs, named, frame)
    if wanted is None:
        return None
    rule = grouped.confidence
    if rule is not None and (callable(rule) or rule not in _CONFIDENCE):
        if not callable(rule) and rule not in FOLD:
            raise ValueError(f"unknown confidence rule {rule!r}; expected one of "
                             f"{sorted(FOLD)}, a callable, or None")
        return None
    seen = set()                       # grouped columns some slice had a confidence for

    def local(part):
        resolver = Resolver(part)
        columns = []
        for index, (_name, method, target) in enumerate(wanted):
            expr = _resolved(target, resolver)
            if method == "mean":
                columns += [expr.sum().alias(f"__jb_s{index}"),
                            expr.count().alias(f"__jb_n{index}")]
            else:
                columns.append(getattr(expr, method)().alias(f"__jb_a{index}"))
        if length is not None:
            columns.append(pl.len().alias("__jb_len"))
        data = resolver.attach(part._df)
        if rule is not None:
            for column in by:
                name = shadow(SCORE, column)
                if name in data.columns:
                    seen.add(column)
                    measured = pl.col(name)
                else:
                    measured = pl.repeat(1.0, pl.len())
                columns += _CONFIDENCE[rule][0](measured, column)
        return data.group_by(by, maintain_order=True).agg(columns)

    def merge(parts):
        data = pl.concat(parts, how="vertical_relaxed")
        schema = data.schema
        folded = [getattr(pl.col(name), _MERGE_OF(name, wanted, rule))().alias(name)
                  for name in data.columns if name not in by]
        merged = data.group_by(by, maintain_order=True).agg(folded)
        # a sum of counts is a count: keep the dtype one slice gave it
        return merged.cast({name: schema[name] for name in merged.columns
                            if name not in by})

    partial = _reduce(_partials(frame, local), lambda one: one, merge)

    out = [pl.col(column) for column in by]
    for index, (name, method, _target) in enumerate(wanted):
        if method == "mean":
            total, count = pl.col(f"__jb_s{index}"), pl.col(f"__jb_n{index}")
            out.append(pl.when(count > 0).then(total / count).otherwise(None).alias(name))
        else:
            out.append(pl.col(f"__jb_a{index}").alias(name))
    if length is not None:
        out.append(pl.col("__jb_len").alias(length))
    if rule is not None:
        out += [_CONFIDENCE[rule][1](column).alias(shadow(SCORE, column))
                for column in by if column in seen]
    return frame._wrap(partial.select(out))


def _partials(frame, local):
    for part in frame.batches():
        yield local(part)


def _decompose(exprs, named, frame):
    """[(output name, method, target)] when every aggregation is one of the
    decomposable methods on a row-local expression, else None."""
    graph, names = frame.graph, set(frame._plan.names())
    wanted = []
    for one in exprs:
        target = _decomposable(one, graph)
        if target is None or not isinstance(target, Col):
            return None
        name = ".".join(target._path)
        if name not in names:
            # a positional aggregate is named by what polars resolves it to,
            # which for anything but a plain column is not decided until then
            return None
        wanted.append((name, one.method, target))
    for name, one in named.items():
        target = _decomposable(one, graph)
        if target is None:
            return None
        wanted.append((name, one.method, target))
    if len({name for name, _method, _target in wanted}) != len(wanted):
        return None
    return wanted


def _decomposable(one, graph):
    if not isinstance(one, _Method) or one.method not in DECOMPOSABLE:
        return None
    if one.args or one.kwargs or not row_local(one.target, graph):
        return None
    if one.method == "count" and _names_relation(one.target, graph):
        return None                    # an arity, not an aggregate
    return one.target


# confidence rule -> (partial columns for one grouped column, final expression)
_CONFIDENCE = {
    "mean": (lambda measured, column: [measured.sum().alias(f"__jb_cs_{column}"),
                                       pl.len().alias(f"__jb_cn_{column}")],
             lambda column: pl.col(f"__jb_cs_{column}") / pl.col(f"__jb_cn_{column}")),
    "min": (lambda measured, column: [measured.min().alias(f"__jb_c_{column}")],
            lambda column: pl.col(f"__jb_c_{column}")),
    "max": (lambda measured, column: [measured.max().alias(f"__jb_c_{column}")],
            lambda column: pl.col(f"__jb_c_{column}")),
    "product": (lambda measured, column: [measured.product().alias(f"__jb_c_{column}")],
                lambda column: pl.col(f"__jb_c_{column}")),
    "sum": (lambda measured, column: [measured.sum().alias(f"__jb_c_{column}")],
            lambda column: pl.col(f"__jb_c_{column}")),
    "first": (lambda measured, column: [measured.first().alias(f"__jb_c_{column}")],
              lambda column: pl.col(f"__jb_c_{column}")),
}


def _MERGE_OF(name, wanted, rule):
    """How one partial column folds across slices."""
    if name.startswith("__jb_a"):
        return _MERGE[wanted[int(name[len("__jb_a"):])][1]]
    if name.startswith(("__jb_s", "__jb_n", "__jb_len", "__jb_cs_", "__jb_cn_")):
        return "sum"
    if name.startswith("__jb_c_"):
        return rule                    # min, max, product, sum, first: themselves
    raise AssertionError(name)


def joined(frames):
    """Frames of one walk as one, their shadows reconciled (plan.stack)."""
    first = frames[0]
    variables = {}
    for one in frames:
        for name, type_ in one.vars.items():
            if variables.get(name) is None:
                variables[name] = type_
    return first._wrap(stack([one.raw for one in frames]), variables)


def _reduce(parts, local, merge):
    """Reduce each part, and fold what is waiting into what was folded whenever
    it has grown as large -- amortized linear, and never more than about twice
    the answer held at once."""
    merged, waiting, pending = None, [], 0
    for part in parts:
        reduced = local(part)
        waiting.append(reduced)
        pending += len(reduced)
        if pending >= max(len(merged) if merged is not None else 0, 1):
            merged = merge(([merged] if merged is not None else []) + waiting)
            waiting, pending = [], 0
    if waiting:
        merged = merge(([merged] if merged is not None else []) + waiting)
    return merged
