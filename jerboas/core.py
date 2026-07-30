"""The four core interfaces of the library, in one place.

Everything the user writes is an instance of one of these families, and the rule
that separates them is deliberately blunt:

    METHODS are only the pipeline verbs  -> select / where / rank / top / groupby / using
    OBJECTS are everything you pass to them.

Methods live only on Graph and Query (see graph.py / query.py); they are the only
things that ever touch the graph. Everything else here is a passive object: it
describes *what* you want, never *how* to get it. The Query is the single place
that binds these passive objects to a concrete Graph and compiles them into the
neutral IR (ir.py) an Engine consumes.

The families:

    Ref        a passive reference into the pattern (a noun). Goes in select(...).
               Expr is the subset of Ref that evaluates to a scalar and therefore
               supports comparison operators.
    Condition  a hard constraint (a filter). Goes in where(...). Deterministic.
    Strategy   a soft score (the non-deterministic-first core). Goes in rank(...).
    Engine     the execution backend. Goes in using(...).

Adding a new matcher = a new Condition subclass. Adding a new ranker = a new
Strategy. Neither requires editing the Query: each object knows how to describe
itself to the compiler (Condition.compile) or to the ranker (Strategy.score).

Design principle -- the predicate surface follows Polars. References are
column-style (`node.name`), predicates are built by operators and fluent methods
on them (`node.year >= 1990`, `node.name.is_in(...)`, `node.name.contains(...)`,
`node.year.is_between(...)`), and booleans combine with `& | ~`. The Condition
constructors these produce (Compare, In, ...) are the underlying objects, not the
spelling a user reaches for. The library's own, non-Polars primitives -- the
non-deterministic-first ones (Like, the strategies, Score) -- are objects that
*wrap* those Polars-style expressions. Familiar where it can be, novel only where
it must be.

Graded membership is the underlying primitive. A crisp Condition returns
membership in {0, 1}; the non-deterministic-first heart is that it can be in
[0, 1]. An object that is BOTH a Condition and a Strategy is a *soft* one (see
conditions.Like): its compile() emits a widened crisp admission (so engines stay
crisp) while its score() contributes the graded weight. The Query treats any
where-condition that is also a Strategy as a "soft where" -- it admits AND
weights from a single object.
"""

from abc import ABC, abstractmethod


# --------------------------------------------------------------------------- #
# References (nouns): the things you can project and constrain.                #
# --------------------------------------------------------------------------- #

class Ref(ABC):
    """A passive reference into a query pattern.

    A Ref carries no knowledge of any Graph: it is pure syntax that the Query
    resolves at compile time. Node, Edge and Path are Refs; Attr and Degree are
    the Expr subset below.

    Refs are what you put in select(...). They have identity semantics on
    purpose (compared by `is`, hashed by id): two `Node("movie")` values are two
    distinct pattern variables, even though they look the same.
    """


class Expr(Ref):
    """A Ref that evaluates to a scalar, so it supports comparison operators.

    Comparisons never touch the graph: they build a Compare Condition (see
    conditions.py) that the Query compiles later. This is the hybrid rule the
    project settled on -- operators where Python allows them, named Condition
    objects (Like, In, Has) where it does not.

    A special case lives in Attr.__eq__: when the right-hand side is itself a
    Ref, `==` means "there is a relation to that node" (a Has edge) rather than
    a value equality. That is what makes `movie.directed_by == Node("person")`
    work without the Node ever knowing the schema.
    """

    # NOTE: concrete operator bodies live on Attr / Degree in refs.py so each
    # can decide value-vs-relation semantics. The abstract shape is documented
    # here; see conditions.Compare for the object produced.


# --------------------------------------------------------------------------- #
# Conditions (hard filters): what goes in where(...).                          #
# --------------------------------------------------------------------------- #

class Condition(ABC):
    """A hard, deterministic constraint on the pattern.

    Conditions are composable for free: `a & b`, `a | b`, `~a`. They stay fully
    passive -- the only behaviour they carry is `compile`, which *describes* the
    constraint to a Compiler (owned by the Query). The Compiler owns all
    graph/schema knowledge, so a Condition never resolves a column, seeds a
    variable, or reads the graph itself: it only says what it constrains and
    lets the Query decide how that lands in the engine IR.
    """

    @abstractmethod
    def compile(self, ctx):
        """Emit this constraint into the pattern under construction.

        `ctx` is a Compiler (see below): the Query-owned facade that resolves a
        Ref to a pattern variable, hands out its NodeSpec, adds edges, and
        exposes the graph schema. Returns nothing; it mutates `ctx`.
        """

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


# And / Or / Not are declared here (not in conditions.py) because Condition's
# operator methods need them; conditions.py re-exports them for discoverability.

class _Combinator(Condition):
    """Shared base for the boolean combinators."""

    def __init__(self, *operands):
        self.operands = operands


class And(_Combinator):
    def compile(self, ctx):
        for operand in self.operands:
            operand.compile(ctx)


class Or(_Combinator):
    """Disjunction. Unlike And it cannot be a single conjunctive pattern, so the
    Query searches each branch and unions the rows (see Query._run_or). compile
    is therefore never called directly on an Or -- the Query splits it first."""

    def __or__(self, other):
        return Or(*self.operands, other)

    def compile(self, ctx):
        raise TypeError("Or is expanded by the Query into separate searches, not compiled inline")


class Not(Condition):
    def __init__(self, operand):
        self.operand = operand

    def __invert__(self):
        return self.operand

    def compile(self, ctx):
        # Negation attaches to the wrapped constraint (exclusion / anti-join);
        # the concrete condition decides what "negated" means for it.
        self.operand.compile(ctx.negated())


# --------------------------------------------------------------------------- #
# Compiler: the Query-owned facade a Condition emits into.                     #
# --------------------------------------------------------------------------- #

class Compiler(ABC):
    """The compile-time context handed to Condition.compile.

    This facade is exactly the vocabulary of the Condition family: one method per
    kind of thing a Condition needs to say. That keeps Conditions passive (they
    only *describe* themselves) while the Query's implementation (query.py) owns
    every graph/schema decision -- how a description becomes a NodeSpec
    predicate, a seed, an edge, or a cross-row anti-join. Add a Condition and you
    reuse this vocabulary; you extend it only when a genuinely new *kind* of
    constraint appears.
    """

    @abstractmethod
    def constrain(self, expr, op, value):
        """Constrain an Expr (Attr or Degree) of a node: op in
        eq/ne/lt/le/gt/ge/contains. The Query decides if this seeds the variable
        (id ==/in), adds a stored-attribute predicate, or a structural (Degree)
        one. Called by Compare."""

    @abstractmethod
    def member_of(self, ref, values):
        """Set membership: a Node's identity keys, or an Attr's allowed values.
        Called by In."""

    @abstractmethod
    def edge(self, source, relation, target, reverse=False):
        """A relation edge between two Node refs, walked forwards or (reverse=True)
        backwards. Called by Has (and by path())."""

    @abstractmethod
    def path(self, path_ref, pattern):
        """Bind a Path ref to a [Node, Edge, Node, ...] shape. Called by Match."""

    @abstractmethod
    def schema(self):
        """The graph schema {type: {"columns": set, "relations": set}} -- the
        only graph knowledge a Condition may consult."""

    @abstractmethod
    def negated(self):
        """A view of this Compiler in which the next emitted constraint is
        negated (exclusion for a predicate, anti-join for an edge). Used by Not,
        so individual Conditions carry no negate flag of their own."""


# --------------------------------------------------------------------------- #
# Strategy (soft score): the non-deterministic-first core, goes in rank(...).  #
# --------------------------------------------------------------------------- #

class Strategy(ABC):
    """A ranking strategy: scores result rows, optionally precomputing
    graph-derived state once via fit().

    This is the heart of "non-deterministic-first": where a Condition hard-keeps
    or drops a row, a Strategy orders rows by a learned or computed affinity, and
    several strategies combine (each min-max normalized, then summed) so a soft
    filter rewards rather than excludes. The lifecycle (when fit runs, when score
    is called) is owned by the Query and documented per method.
    """

    supports_guidance = False   # set True + implement edge_weight to guide engines like Greedy

    def fit(self, graph):
        """Called by the Query exactly once per distinct strategy instance per
        run, before search/scoring. Override to precompute graph-derived state
        (a factorization, a reachability index). Must be idempotent. Default:
        no-op."""

    @abstractmethod
    def score(self, query, rows):
        """Return one float per row (same order), higher = better."""

    def edge_weight(self, source, relation, target):
        """Only called when supports_guidance is True. How promising traversing
        this edge is, for engines that prune (e.g. Greedy)."""
        raise NotImplementedError

    def cached(self, graph, key, builder):
        """Memoize a computation for (this strategy instance, this graph, key).
        Lives on the strategy, not the graph, because the result usually depends
        on the strategy's own hyperparameters, not just the graph."""
        cache = self.__dict__.setdefault("_strategy_cache", {})
        cache_key = (id(graph), key)
        if cache_key not in cache:
            cache[cache_key] = builder()
        return cache[cache_key]


# --------------------------------------------------------------------------- #
# Engine (execution backend): goes in using(...).                             #
# --------------------------------------------------------------------------- #

class Engine(ABC):
    """A search engine: resolves a compiled SearchSpec (ir.py) against a Graph."""

    @abstractmethod
    def search(self, graph, spec, guide=None):
        """Return a SearchResult. `guide`, if given, is a Strategy with
        supports_guidance=True whose fit() the Query has already called -- the
        engine may call guide.edge_weight(...) but not guide.fit(...)."""
