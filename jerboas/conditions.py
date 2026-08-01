"""Conditions: the hard filters that go in where(...).

Every constraint the user can express is one Condition subclass, and they all
share the single `compile(ctx)` seam from core.py. That is what makes the family
open: a new matcher is a new class here, never an edit to the Query.

The hybrid rule in practice:
    operators on an Expr   -> Compare        (node.year >= 1990)
    a Ref on the right of ==-> Has           (movie.directed_by == Node("person"))
    set / fuzzy / edge     -> In / Like / Has (things operators can't say cleanly)
    a path shape           -> Match          (path == [...])
    boolean glue           -> And / Or / Not (from core.py, re-exported below)

compile bodies here are intentionally thin: a Condition only *describes* itself
to the Compiler (ctx), which owns all graph/schema/seed decisions. The bodies
below are illustrative of that contract; the engine-specific choices they defer
to ctx (id-seeding, anti-joins, label-column resolution) are implemented in
query.py in the porting phase.
"""

import math
from difflib import SequenceMatcher

from .core import Condition, Strategy, And, Or, Not   # re-exported so `from .conditions import Not` works


class Compare(Condition):
    """A comparison between an expression (Attr or Degree) and a value.

        node.year >= 1990        -> Compare(Attr(node,'year'), 'ge', 1990)
        node.title == "Akira"    -> Compare(Attr(node,'title'), 'eq', 'Akira')
        node.rel.count() >= 5    -> Compare(Degree(node,'rel'), 'ge', 5)

    The Query reads `left` to tell a stored-attribute compare (Attr) from a
    structural one (Degree), so this one class covers both without a separate
    DegreePredicate type.
    """

    OPS = ("eq", "ne", "lt", "le", "gt", "ge", "contains")

    def __init__(self, left, op, value):
        self.left = left          # Attr or Degree
        self.op = op
        self.value = value

    def compile(self, ctx):
        # Describe "constrain left.node's <attr/degree> with op/value" to ctx;
        # ctx decides whether it becomes a seed (id ==/in), a stored-attr
        # predicate, or a structural predicate.
        ctx.constrain(self.left, self.op, self.value)


class Like(Condition, Strategy):
    """Fuzzy membership: a graded [0,1] version of a crisp Condition. THE
    non-deterministic-first primitive.

    Like wraps *exactly one* Condition and softens it -- one uniform signature,
    no special cases. A crisp Condition returns membership in {0, 1}; Like
    returns it in [0, 1], the smooth "leaky" generalization:

        Like(person.name.is_in(args.people))            # -> string similarity
        Like(movie.year < 3, mode="gaussian", t=0.8)    # -> soft threshold (leaks past 3)
        Like(movie.year == 1994, width=5)               # -> soft equality (+/- 5)

    The membership shape is inferred from the wrapped condition's operator
    (mode="auto"): a set/equality on a categorical attribute softens to
    similarity, a threshold softens to a gaussian/sigmoid leak. Pass `mode`
    explicitly to override.

    Like is the bridge between the two families -- it IS both a Condition and a
    Strategy:

      * as a Condition (`support()` -> compile): a *widened crisp* version of the
        wrapped condition, so the engine only ever sees a hard admission set
        (mu > eps) and never chases a gaussian across the whole graph.
      * as a Strategy (`membership()` -> score): the graded weight, a soft
        ranking signal.

    Placement follows from that duality (decided for the project):
      * in where(...): BOTH -- widened admission AND automatic ranking weight
        (the "soft where"). The Query auto-registers any where-condition that is
        also a Strategy into the rank scope; you don't repeat it in rank(...).
      * in rank(...): the weight only (everything is already admitted).
    """

    def __init__(self, condition, *, mode="auto", t=1.0, width=None, eps=0.6, k=1):
        self.condition = condition  # the crisp Condition being softened (always a Condition)
        self.mode = mode            # "auto" | "gaussian" | "sigmoid" | "linear" | "leaky"
        self.t = t                  # temperature: how far membership diffuses past the edge
        self.width = width          # explicit spread for equality/range softening
        self.eps = eps              # similarity below which a match is not a match
        self.k = k                  # how many best matches each needle admits

    # -- Condition side: the widened crisp admission --

    def _sigma(self):
        return self.width if self.width is not None else self.t

    def _operand(self):
        """(attr, op, value) of the wrapped condition. `op` is 'in' for a set
        membership, otherwise the comparison operator."""
        if isinstance(self.condition, In):
            return self.condition.ref, "in", self.condition.values
        if isinstance(self.condition, Compare):
            return self.condition.left, self.condition.op, self.condition.value
        return None, None, None

    def support(self):
        """The crisp Condition the engine admits on: the wrapped condition widened
        to (roughly) the region where membership exceeds eps. Returns a plain
        Condition so the IR/engine stay crisp -- a set softens to substring
        admission, a threshold widens by ~3 sigma."""
        attr, op, value = self._operand()
        if op == "in":                                   # string set -> substring admission
            return Compare(attr, "contains_any", list(value))
        margin = 3.0 * self._sigma()
        if op in ("lt", "le"):
            return Compare(attr, op, value + margin)
        if op in ("gt", "ge"):
            return Compare(attr, op, value - margin)
        if op == "eq":                                   # soft equality -> a band
            return And(Compare(attr, "ge", value - margin), Compare(attr, "le", value + margin))
        return self.condition                            # already crisp-broad (e.g. contains)

    def compile(self, ctx):
        attr, op, needles = self._operand()
        if op == "in" and needles and all(isinstance(n, str) for n in needles):
            # a set of strings is a search box, not a filter: admit the k values
            # closest to each needle rather than a region around them, judged by
            # the same measure the graded weight uses, so the two cannot disagree
            ctx.best_match(attr, needles, self.k, self.eps, self.closeness)
            return
        self.support().compile(ctx)

    # -- Strategy side: the graded weight --

    def _membership(self, op, value, raw):
        if op == "in" or op == "contains":               # string similarity to any needle
            needles = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
            return self._similarity(raw, needles)
        try:
            x, v = float(raw), float(value)
        except (TypeError, ValueError):
            return 0.0
        sigma = self._sigma() or 1.0
        inside = {"lt": x < v, "le": x <= v, "gt": x > v, "ge": x >= v, "eq": x == v}.get(op, False)
        if op == "eq":                                   # symmetric bump around v
            return self._decay(abs(x - v), sigma)
        if inside:
            return 1.0
        d = (x - v) if op in ("lt", "le") else (v - x)   # distance past the boundary (>0 outside)
        return self._decay(d, sigma)

    def _decay(self, d, sigma):
        r = d / sigma
        if self.mode == "linear":
            return max(0.0, 1.0 - r / 3.0)
        if self.mode == "sigmoid":
            return 1.0 / (1.0 + math.exp(r - 3.0))
        if self.mode == "leaky":                         # gaussian with a small linear leak
            return max(0.05 * max(0.0, 1.0 - r / 6.0), math.exp(-0.5 * r * r))
        return math.exp(-0.5 * r * r)                    # gaussian (also "auto")

    def closeness(self, needle, text):
        """How close one stored value is to one needle, in [0, 1].

        Containment counts as a perfect match, so "tarantino" finds "Quentin
        Tarantino" the way a search box would -- except when the needle is empty,
        which is contained in everything and would make a blank search the
        broadest one possible instead of the narrowest."""
        n = str(needle).lower()
        return 1.0 if n and n in text else SequenceMatcher(None, n, text).ratio()

    def _similarity(self, raw, needles):
        if raw is None:
            return 0.0
        text = str(raw).lower()
        return max((self.closeness(n, text) for n in needles), default=0.0)

    def score(self, query, rows):
        attr, op, value = self._operand()
        if attr is None:
            return [0.0] * len(rows)
        col = query._columns.get(id(attr.node), query.primary_column)
        graph, name = query.graph, attr.name
        return [self._membership(op, value, graph.value(row[col], name)) for row in rows]


class In(Condition):
    """Set membership. Two readings, decided by what `ref` is:

        In(node, keys)            # node identity: bind only to these full keys
        In(node.year, years)      # attribute: node.year is one of these values

    `~In(...)` (via Not) excludes instead. This is the object the Polars-style
    `.is_in()` method builds (node.name.is_in(...)); the fluent form is the one
    users reach for, this constructor is the underlying representation.
    """

    def __init__(self, ref, values):
        self.ref = ref            # Node (identity) or Attr (attribute)
        self.values = values

    def compile(self, ctx):
        ctx.member_of(self.ref, set(self.values))


class Has(Condition):
    """A relation between two nodes, i.e. an edge in the pattern.

        movie.directed_by == Node("person")   -> Has(movie, 'directed_by', person)
        Has(user, "has_interact", rec)         # explicit form
        ~Has(user, "has_interact", rec)        # anti-join: the edge must NOT exist

    A positive Has is a normal edge in the conjunctive pattern; a negated one is
    a cross-row anti-join applied after the search (two already-bound endpoints).
    `relation` may be a name, a tuple of names (any of them), or None (wildcard).
    `reverse` walks the relation from target to source.
    """

    def __init__(self, source, relation, target, reverse=False):
        self.source = source      # Node
        self.relation = relation  # str | tuple[str, ...] | None
        self.target = target      # Node
        self.reverse = reverse

    def compile(self, ctx):
        ctx.edge(self.source, self.relation, self.target, self.reverse)


class Match(Condition):
    """A path constrained to a shape: `path == [rec, Edge(), seed]`.

    The pattern list interleaves nodes and Edges (a bare Node-Node gap means a
    wildcard edge). Two Matches OR together for alternative shapes, which the
    Query turns into a union of searches (it can't be one conjunctive pattern).
    """

    def __init__(self, path, pattern):
        self.path = path
        self.pattern = pattern    # [Node, Edge, Node, ...]

    def __or__(self, other):
        return Or(self, other)

    def compile(self, ctx):
        ctx.path(self.path, self.pattern)


__all__ = ["Condition", "Compare", "Like", "In", "Has", "Match", "And", "Or", "Not"]
