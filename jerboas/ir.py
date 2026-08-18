"""The neutral intermediate representation an Engine consumes.

The DSL (refs + conditions) is only surface syntax; the Query compiles it into a
SearchSpec -- a conjunctive pattern plus the output columns -- that any Engine
resolves without ever seeing a Ref or a Condition.

What used to live here and no longer does: AttrPredicate, StructuralPredicate
and the string/number coercion they needed. Every constraint a NodeSpec can
carry is node-local and independent of the binding, so the compiler evaluates
all of them once against the graph's typed columns and hands the engine a single
boolean mask. The engine's per-candidate test is `spec.mask[node]`.

Nodes are integers here, and stay integers all the way through ranking; they
become Key values only in Query._render.
"""

from dataclasses import dataclass, field


def aliased(aliases, name, type_):
    """The column a queried name means for one type.

    The only place a name is translated. `Node.alias(label="title")` is what
    puts something in `aliases`; with nothing declared the name *is* the column,
    which is why `graph.column()` is a dict lookup and not a resolution."""
    target = aliases.get(name, name) if aliases else name
    return target.get(type_, name) if isinstance(target, dict) else target


@dataclass(eq=False)
class NodeSpec:
    """A node constraint in the compiled pattern, as an admission mask.

    `mask[i]` is True when node i satisfies everything the query asked of this
    variable: its type, its identity set, its attribute predicates and its
    degree predicates, all folded together at compile time."""

    var: object
    type: object = None
    mask: object = None                 # boolean ndarray over every node
    aliases: object = None              # {queried name: column, or {type: column}}

    def column(self, name, type_):
        return aliased(self.aliases, name, type_)

    def admits(self, index):
        return bool(self.mask[index])


@dataclass(eq=False)
class EdgeSpec:
    """A relation constraint between two variables.

    `relation` is a relation code, or None for the wildcard (any relation, and
    -- since the store is directed with a transpose -- either direction).
    `reverse` picks the direction for a named relation.

    `admits` is the edge-level twin of NodeSpec.mask: an (out, in) pair of
    boolean arrays over stored edges, or None when the query says nothing about
    weights. A predicate on an edge's score is folded into it at compile time,
    so the engine reads a slice and never learns what a weight is."""

    source: object
    relation: object
    target: object
    reverse: bool = False
    admits: object = None


@dataclass(eq=False)
class Output:
    """One result column: a node variable, or an ordered path of variables
    interleaved with the relation codes actually traversed."""

    kind: str                     # "node" | "path"
    ref: object                   # variable name (node) | list of variables (path)
    edges: list = field(default_factory=list)

    def extract(self, binding, relations):
        if self.kind == "node":
            return binding[self.ref]
        seq = []
        for i, var in enumerate(self.ref):
            seq.append(binding[var])
            if i < len(self.edges):
                seq.append(relations.get(self.edges[i]))
        return tuple(seq)


@dataclass(eq=False)
class SearchSpec:
    """A conjunctive pattern plus the columns to return."""

    nodes: dict                   # {var: NodeSpec}
    edges: list                   # [EdgeSpec]
    outputs: list                 # [Output]


@dataclass(eq=False)
class SearchResult:
    """What an Engine returns: matched rows plus every node it walked through
    (the latter feeds ranking strategies that need the touched subgraph)."""

    rows: list
    visited: set
