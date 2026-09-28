"""A step: which relations a walk follows, and how many of their edges.

Every argument of `hop` becomes one. A plain step follows every edge, so a hop
costs what its nodes' degrees sum to and a walk multiplies them. A budgeted step
follows at most n per row -- a selector, as decoding has them -- and selectors
chain into a cascade:

    .hop(peer="~has_interact")                                   # every edge
    .hop(peer=step("~has_interact").top(20))                     # the 20 heaviest
    .hop(peer=step("~has_interact").sample(20, seed=0))          # 20 drawn by weight
    .hop(peer=step("~has_interact").top_p(0.9))                  # the nucleus
    .hop(rec=step("contains").top(500, by=specific)              # cheap, then
                             .top(50, by=taste))                 # expensive

`by` is the map a selector ranks by (`top`), draws in proportion to
(`sample`) or shares out (`top_p`): an expression over the step's arrival -- `v.peer.score`, the edge's
weight, by default; a number weighs every edge alike. A map that reads nothing
but the step is a fact about the graph, so it is computed once over every edge
of the relations and fused into the store (traverse.Fused): the hop then costs
n per row, whatever the degree. A map that also reads the frame the walk leaves
from -- whose taste, which seed -- is evaluated on each row's candidates,
without building a row of the frame for any of them.

`probability()` makes what the step measured the probability a random walk
gets there: each kept edge's share of its row's mass under a map, times the
confidence of the node it left from. A seed's arrivals then share one unit
between them however many there are, the last step's confidence is the
probability of the whole walk, and `group_by(confidence="sum")` is the chance
of ending on each node.

The trade a budget makes is the only one it makes: an answer the walk would
have reached through an edge it did not follow is not in the frame. What a
model then scores is a subset of the exact candidates, never a different set.
"""

import numpy as np
import polars as pl

from .expr import REVERSED, SCORE, direction, is_shadow, shadow, shadowed
from . import traverse

_BY = "__jb_by"


def step(*relations):
    """A step over these relations (`~name` backwards); none at all is any
    relation, either way."""
    if len(relations) == 1 and not isinstance(relations[0], str):
        relations = tuple(relations[0])
    return Step.of(relations)


class Select:
    """One selector: per row, the `width` best edges by `by`, `width` drawn in
    proportion to it, or the fewest best holding `share` of its mass."""

    __slots__ = ("width", "by", "draw", "seed", "share")

    def __init__(self, width=None, by=None, draw=False, seed=None, share=None):
        if width is not None and width < 1:
            raise ValueError(f"a budget keeps at least one edge, not {width}")
        if share is not None and not 0 < share <= 1:
            raise ValueError(f"a share of a row's mass is in (0, 1], not {share}")
        self.width = None if width is None else int(width)
        self.by, self.draw, self.seed, self.share = by, draw, seed, share

    def __repr__(self):
        if self.share is not None:
            return f"top_p({self.share})"
        return f"{'sample' if self.draw else 'top'}({self.width})"


class Step:
    """Relations read one way or another, walked in full or through selectors.

    Passive, like every expression: the frame walks it, and the planner reads
    off it what the walk will cost."""

    __slots__ = ("specs", "stages", "measure")

    def __init__(self, specs, stages=(), measure=None):
        self.specs = specs              # [(relation, backwards)], None for any
        self.stages = stages            # Select, in the order they apply
        self.measure = measure          # (by,) for probability(), else None

    @classmethod
    def of(cls, spec):
        """What `hop` was given, as a step: a Step, a relation, `~relation`, or
        a collection of them -- the empty one being any relation at all."""
        if isinstance(spec, Step):
            return spec
        specs = (spec,) if isinstance(spec, str) else tuple(spec)
        if not all(isinstance(one, str) for one in specs):
            raise TypeError(f"a step is a relation or a collection of them, not {spec!r}")
        return cls([direction(one) for one in specs] if specs else None)

    # -- building one --

    def top(self, n, by=None):
        """The n edges of each row ranking highest by `by`."""
        return Step(self.specs, self.stages + (Select(n, by),), self.measure)

    def sample(self, n, by=None, seed=None):
        """n edges of each row drawn with replacement in proportion to `by`,
        which must not be negative, and folded to the distinct ones.

        With a seed the draw is a function of the node (traverse.Ranked.draw):
        the same answer every time and however the walk is sliced. Without one
        every walk draws afresh."""
        return Step(self.specs, self.stages + (Select(n, by, True, seed),), self.measure)

    def top_p(self, p, by=None):
        """The fewest best edges of each row holding `p` of its mass under
        `by` -- a nucleus, as a decoder keeps one: few edges where one
        dominates, many where none does."""
        return Step(self.specs, self.stages + (Select(by=by, share=p),), self.measure)

    def probability(self, by=None):
        """What the step measured becomes the probability a random walk gets
        there: each kept edge's share of its row's mass under `by` (the edge's
        weight by default, `1` for every edge alike), times the confidence of
        the node the row stands on -- 1.0 for a node nothing measured, so a
        walk starts with one unit per row. Renormalized over the edges kept, as
        a decoder renormalizes over its top k.

        Along several such steps the last confidence is the walk's probability,
        and `group_by(confidence="sum")` adds up the walks that end together."""
        return Step(self.specs, self.stages, (by,))

    # -- what the planner reads off it --

    def relations(self):
        """[(relation, backwards)], or None for any relation either way."""
        return self.specs

    def single(self):
        """The one relation it walks, as `v.x.via` names it, or None."""
        if self.specs is None or len(self.specs) != 1:
            return None
        name, backwards = self.specs[0]
        return REVERSED + name if backwards else name

    @property
    def width(self):
        """The most edges a row can keep, or None when it may keep them all."""
        widths = [one.width for one in self.stages if one.width is not None]
        return min(widths, default=None)

    def known(self, graph):
        """Refuse a relation the graph has never seen: walking one matches
        nothing, which is an indefensible answer to a typo."""
        for name, _backwards in self.specs or ():
            if graph.relation_code(name) is None:
                raise ValueError(
                    f"no relation {name!r} in this graph; it has: "
                    f"{', '.join(graph.relations)}")

    def degree(self, graph, flipped=False):
        """Per node, how many rows one step out of it makes -- or, `flipped`,
        how many edges of the step arrive at it."""
        if self.specs is None:
            total = graph.degree(None, False) + graph.degree(None, True)
        else:
            total = sum(graph.degree(name, backwards != flipped)
                        for name, backwards in self.specs)
        width = self.width
        return total if flipped or width is None else np.minimum(total, width)

    def target_type(self, graph):
        """What it lands in, before it is taken, when that is one type."""
        if self.specs is None or len(self.specs) != 1:
            return None
        name, backwards = self.specs[0]
        return graph.target_types(name, backwards)

    def reaches(self, graph, targets):
        """Its edges into `targets`, found from that end: one Reach per
        relation, in the order `walk` concatenates them."""
        if self.specs is None:
            return [traverse.Reach(graph, targets)]
        return [traverse.Reach(graph, targets, name, backwards)
                for name, backwards in self.specs]

    # -- walking it --

    def walk(self, graph, nodes, name, context=None, reaches=None):
        """(rows, targets, codes, weights), as traverse.expand returns them.

        `context(roots, rows)` hands a map the columns it reads of the frame
        the walk leaves from, at the rows given; only a map reading beyond the
        step asks for it."""
        nodes = np.asarray(nodes, dtype=np.int64)
        stages = self.stages
        if reaches is not None:
            arrays = _merge([reach.gather(nodes, normalized=True) for reach in reaches])
        elif stages and self._fusable(stages[0], name):
            arrays = self._fused(graph, nodes, name, stages[0])
            stages = stages[1:]
        else:
            arrays = self._expand(graph, nodes)
        for stage in stages:
            arrays = self._select(graph, nodes, name, context, arrays, stage)
        if self.measure is not None:
            arrays = self._probability(graph, nodes, name, context, arrays)
        return arrays

    def _expand(self, graph, nodes):
        if self.specs is None:
            return traverse.expand(graph, nodes, None, None, normalized=True)
        return _merge([traverse.expand(graph, nodes, name, backwards, normalized=True)
                       for name, backwards in self.specs])

    def _fusable(self, stage, name):
        """A first selector whose map reads only the step, over relations read
        one way, is folded into the store."""
        return (self.specs is not None
                and len({backwards for _name, backwards in self.specs}) == 1
                and not _outside(stage.by, name))

    def _fused(self, graph, nodes, name, stage):
        backwards = self.specs[0][1]
        codes = tuple(sorted(graph.relation_code(one) for one, _b in self.specs))
        # keyed by the map itself, which the cache holds on to, so the key
        # cannot outlive it and come to name another
        by = stage.by if stage.by is None or isinstance(stage.by, (int, float)) else id(stage.by)
        _held, fused = graph.cached(
            ("fused", codes, backwards, name, by),
            lambda: (stage.by, traverse.Fused(
                graph, codes, backwards, self._ranking(graph, codes, backwards, name, stage))))
        if stage.share is not None:
            rows, positions = fused.nucleus(nodes, stage.share)
        elif stage.draw:
            rows, positions = fused.draw(nodes, stage.width, _seed(stage))
        else:
            rows, positions = fused.best(nodes, stage.width)
        return traverse.gathered(graph, rows, positions, backwards, normalized=True)

    def _ranking(self, graph, codes, backwards, name, stage):
        """The map over every edge of the relations, in store order."""
        members = traverse.Fused.members(graph, codes, backwards)
        indices = graph.in_indices if backwards else graph.out_indices
        weights = graph.weights(normalized=True)[1 if backwards else 0]
        values = _values(stage.by, graph, name, None, indices[members],
                         weights[members], None)
        _check(values, stage)
        return values.astype(np.float32)

    def _select(self, graph, nodes, name, context, arrays, stage):
        """One selector over the candidates each row already has."""
        rows, targets, codes, weights = arrays
        values = _values(stage.by, graph, name, rows, targets, weights, context)
        _check(values, stage)
        ranked = traverse.Ranked(np.bincount(rows, minlength=len(nodes)), values)
        segments = np.arange(len(nodes))
        if stage.share is not None:
            _kept, index = ranked.nucleus(segments, stage.share)
        elif stage.draw:
            _kept, index = ranked.draw(segments, stage.width, _seed(stage), nodes)
        else:
            _kept, index = ranked.best(segments, stage.width)
        return rows[index], targets[index], codes[index], weights[index]

    def _probability(self, graph, nodes, name, context, arrays):
        rows, targets, codes, weights = arrays
        (by,) = self.measure
        mass = _values(by, graph, name, rows, targets, weights, context)
        if (mass < 0).any():
            raise ValueError("a probability's weights must not be negative")
        totals = np.bincount(rows, weights=mass, minlength=len(nodes))[rows]
        share = np.divide(mass, totals, out=np.zeros(len(mass)), where=totals > 0)
        return rows, targets, codes, share

    def __repr__(self):
        spec = "" if self.specs is None else ", ".join(
            repr(REVERSED + n if b else n) for n, b in self.specs)
        chain = "".join(f".{one!r}" for one in self.stages)
        return f"step({spec}){chain}{'.probability()' if self.measure else ''}"


def _merge(parts):
    return parts[0] if len(parts) == 1 else traverse.interleave(parts)


def _seed(stage):
    if stage.seed is not None:
        return stage.seed
    return int(np.random.default_rng().integers(2**63))


def _check(values, stage):
    if (stage.draw or stage.share is not None) and (values < 0).any():
        raise ValueError("a map drawn from or shared out must not be negative")


def _outside(by, name):
    """The variables a map reads beyond the step's own arrival."""
    if by is None or isinstance(by, (int, float)):
        return set()
    from ..plan.planner import roots
    return (roots(by) or set()) - {name}


def _values(by, graph, name, rows, targets, weights, context):
    """A map over candidate edges: one float per edge.

    Evaluated as a frame of the edges' arrivals and what the step measured,
    beside whatever the map reads of the frame the walk leaves from."""
    if by is None:
        return np.asarray(weights, dtype=np.float64)
    if isinstance(by, (int, float)):
        return np.full(len(targets), float(by))
    from .frame import Frame, _resolved
    from .resolve import Resolver

    tags = np.unique(graph._type_tag_of[targets])
    data = pl.DataFrame({name: np.asarray(targets, dtype=np.int32),
                         shadow(SCORE, name): np.asarray(weights, dtype=np.float64)})
    variables = {name: graph.types[int(tags[0])] if len(tags) == 1 else None}
    outside = _outside(by, name)
    try:
        if outside:
            if context is None:
                raise ValueError(f"it reads {sorted(outside)}, which only a walk "
                                 "out of a frame has")
            columns, types = context(outside, rows)
            data = data.hstack(columns)
            variables.update(types)
        edges = Frame(graph, data, variables)
        resolver = Resolver(edges)
        expr = _resolved(by, resolver)
    except (ValueError, pl.exceptions.ColumnNotFoundError) as error:
        raise ValueError(
            f"a step's map reads its own arrival ({name!r} and what it carries) "
            f"and the frame it leaves from: {error}") from None
    return (resolver.attach(edges._df).with_columns(expr.alias(_BY))[_BY]
            .cast(pl.Float64).fill_null(0.0).to_numpy())


def root_of(column):
    """The variable a frame column belongs to, shadows included."""
    if is_shadow(column):
        column = shadowed(column)[1]
    return column.partition(".")[0]
