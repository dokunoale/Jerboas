"""A walk described but not taken, and how it is run.

Held only inside `optimize` (see optimize.py): the frame a deferred hop returns
carries a `Plan` instead of rows, the filters written after it join the plan,
and the whole runs when something reads the frame.
"""

import polars as pl


class Plan:
    """A walk described but not taken, and the conditions about where it lands.

    Held only inside `optimize`. Running it takes the same walk and applies the
    same conditions, in slices cut by what a step produces -- and cut again at
    every step, so a walk written as one hop and the same walk written as two
    cost the same. The answer is what it would have been; the peak is a slice.
    """

    __slots__ = ("frame", "steps", "budget", "predicates")

    def __init__(self, frame, steps, budget, predicates=()):
        self.frame = frame              # the frame the walk leaves from
        self.steps = steps              # [(relation spec, column or None)]
        self.budget = budget            # rows one step may produce at a time
        self.predicates = list(predicates)

    def narrowed(self, predicates):
        """The same plan, with more said about where the walk may land."""
        return Plan(self.frame, self.steps, self.budget,
                    self.predicates + list(predicates))

    def build(self):
        return _run(self.frame, _groups(self.steps), self.predicates, self.budget)


def _groups(steps):
    """The steps in runs that each end at a named one.

    A step nobody named produces no column, so it cannot end a run: what folds
    two routes through it is the named step after it. Python already puts the
    unnamed ones first, so the first run is all of them plus one, and the rest
    are one apiece."""
    first = next(i for i, (_spec, name) in enumerate(steps) if name is not None)
    return [steps[:first + 1]] + [[one] for one in steps[first + 1:]]


def _run(frame, groups, predicates, budget):
    """One run of steps at a time, each in slices, recursing for the rest."""
    group, rest = groups[0], groups[1:]
    parts = []
    for piece in frame._slices(group[0][0], budget):
        part = piece._hop_eager(group)
        if rest:
            part = _run(part, rest, predicates, budget)
            parts.append(part)
            continue
        if predicates:
            part = part.filter(*predicates)
        parts.append(part.raw)
    if not parts:
        empty = frame._wrap(frame._df.clear())._hop_eager(group)
        return _run(empty, rest, predicates, budget) if rest else empty.raw
    if len(parts) == 1:
        return parts[0]
    # rechunk=False keeps the slices' own buffers instead of copying them into
    # one: the answer exists once rather than twice, which on a walk whose
    # result is most of its cost is the difference between finishing and not
    return pl.concat(parts, how="vertical", rechunk=False)
