"""A walk described but not taken, and how it is run.

Inside `optimize` (see optimize.py) a hop returns a frame that carries a `Plan`
instead of rows. The conditions written after it join the plan, a hop after it
extends the plan, and the whole runs when something reads the frame.

A plan is a list of stages. Each ends at a named step and carries the
conditions that can be decided once that step has landed -- the earliest point
at which everything they read exists, so a condition on the middle of a walk
prunes the middle before the next step multiplies it. Running a stage decides
two things per stage, with the numbers the graph already keeps:

*Which end to walk from.* A stage whose conditions say where it must land
(`v.rec.is_in(wanted)`) can be walked backwards from that set when the set's
edges are fewer than the frame's -- same rows, same order.

*How big a slice.* The budget, turned into rows at this frame's width, cut
against what this step out of these nodes will produce.

The slices are a generator (`parts`), which is what `Frame.batches()` hands out
and what the streamed reductions consume; `build` is the one place they are
accumulated.
"""

import polars as pl

from ..query.expr import SCORE, shadowed
from .planner import landing, roots


class Plan:
    """A walk described but not taken, and the conditions about where it lands.

    Running it takes the same walk and applies the same conditions, in slices
    cut by what a step produces -- and cut again at every step, so a walk
    written as one hop and the same walk written as two cost the same. The
    answer is what it would have been; the peak is a slice.
    """

    __slots__ = ("frame", "stages", "budget")

    def __init__(self, frame, stages, budget):
        self.frame = frame              # the frame the walk leaves from
        self.stages = stages            # [(steps, predicates)], each ending named
        self.budget = budget            # optimize.Budget

    @classmethod
    def of(cls, frame, steps, budget):
        return cls(frame, [(group, []) for group in _groups(steps)], budget)

    def extended(self, steps):
        """The same plan, walked further."""
        return Plan(self.frame, self.stages + [(group, []) for group in _groups(steps)],
                    self.budget)

    def narrowed(self, predicates):
        """The same plan, with more said about where the walk may land. Each
        condition joins the earliest stage after which everything it reads
        exists; one that does not say what it reads waits for the last."""
        stages = [(steps, list(conditions)) for steps, conditions in self.stages]
        base = {name.partition(".")[0] for name in self.frame._df.columns}
        for one in predicates:
            wanted = roots(one)
            index = len(stages) - 1
            if wanted is not None:
                seen = set(base)
                for position, (steps, _conditions) in enumerate(stages):
                    seen |= {name for _spec, name in steps if name is not None}
                    if wanted <= seen:
                        index = position
                        break
            stages[index][1].append(one)
        return Plan(self.frame, stages, self.budget)

    def names(self):
        """The columns the frame will have, as far as a hop needs to know --
        the ones it may not land on."""
        return list(self.frame._df.columns) + [
            name for steps, _conditions in self.stages
            for _spec, name in steps if name is not None]

    def parts(self):
        """The answer a slice at a time, as frames."""
        yield from _parts(self.frame, self.stages, self.budget)

    def build(self):
        return stack([part.raw for part in self.parts()])


def _groups(steps):
    """The steps in runs that each end at a named one.

    A step nobody named produces no column, so it cannot end a run: what folds
    two routes through it is the named step after it. Python already puts the
    unnamed ones first, so the first run is all of them plus one, and the rest
    are one apiece."""
    first = next(i for i, (_spec, name) in enumerate(steps) if name is not None)
    return [steps[:first + 1]] + [[one] for one in steps[first + 1:]]


def _parts(frame, stages, budget):
    """One stage at a time, each in slices, recursing for the rest."""
    (steps, conditions), rest = stages[0], stages[1:]
    toward = direction(frame, steps, conditions)
    degree = toward.degree() if toward is not None else frame._step_degree(steps[0][0])
    for piece in frame._slices(degree, budget.rows(frame, steps)):
        part = piece._hop_eager(steps, toward=toward)
        if conditions:
            part = part.filter(*conditions)
        if rest:
            yield from _parts(part, rest, budget)
        else:
            yield part


class Toward:
    """One step's edges into a set, found from the set's end once and gathered
    onto every slice (query/traverse.py, `Reach`)."""

    __slots__ = ("reaches",)

    def __init__(self, reaches):
        self.reaches = reaches          # one per relation the step names

    def degree(self):
        total = self.reaches[0].degree()
        for reach in self.reaches[1:]:
            total = total + reach.degree()
        return total


def direction(frame, steps, conditions):
    """A `Toward` when this stage is cheaper walked from where it must land,
    None to walk it forwards.

    Both costs are exact: forwards is the frame's degree along the step,
    backwards the set's degree along it read the other way, plus sorting the
    frame's nodes once to match the edges found onto its rows."""
    if len(steps) != 1:
        return None
    spec, name = steps[0]
    wanted = landing(conditions, name, frame.graph)
    if wanted is None:
        return None
    forwards = frame._produces(spec)
    backwards = frame._arriving(spec, wanted) + frame._df.height
    if backwards >= forwards:
        return None
    return Toward(frame._reaches(spec, wanted))


def stack(frames):
    """Slices as one frame, without copying them into one buffer.

    The slices of one walk can disagree about their shadows: a confidence
    column exists only where some weight in the slice was not 1.0, so one slice
    may carry it and the next not. The union is taken, the missing confidence
    is the 1.0 it stood for, and the columns come back in the order a single
    walk would have put them in."""
    frames = [one for one in frames]
    if len(frames) == 1:
        return frames[0]
    order = _union(frames)
    # rechunk=False keeps the slices' own buffers instead of copying them into
    # one, so the answer exists once rather than twice
    data = pl.concat(frames, how="diagonal_relaxed", rechunk=False)
    missing = [name for name in order
               if (shadowed(name) or ("",))[0] == SCORE
               and any(name not in one.columns for one in frames)]
    if missing:
        data = data.with_columns(pl.col(missing).fill_null(1.0))
    return data.select(order)


def _union(frames):
    """Every column any slice has, each placed after the column it follows in
    the slice that has it."""
    order = list(max(frames, key=lambda one: one.width).columns)
    for one in frames:
        previous = None
        for name in one.columns:
            if name not in order:
                order.insert(order.index(previous) + 1 if previous else 0, name)
            previous = name
    return order
