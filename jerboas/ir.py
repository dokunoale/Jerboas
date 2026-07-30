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


class NodeSpec:
    """A node constraint in the compiled pattern, as an admission mask.

    `mask[i]` is True when node i satisfies everything the query asked of this
    variable: its type, its identity set, its attribute predicates and its
    degree predicates, all folded together at compile time."""

    def __init__(self, var, type=None, mask=None):
        self.var = var
        self.type = type
        self.mask = mask

    def admits(self, index):
        return bool(self.mask[index])


class EdgeSpec:
    """A relation constraint between two variables.

    `relation` is a relation code, or None for the wildcard (any relation, and
    -- since the store is directed with a transpose -- either direction).
    `reverse` picks the direction for a named relation."""

    def __init__(self, source, relation, target, reverse=False):
        self.source = source
        self.relation = relation
        self.target = target
        self.reverse = reverse


class Output:
    """One result column: a node variable, or an ordered path of variables
    interleaved with the relation codes actually traversed."""

    def __init__(self, kind, ref, edges=None):
        self.kind = kind          # "node" | "path"
        self.ref = ref            # variable name (node) | list of variables (path)
        self.edges = edges or []

    def extract(self, binding, relations):
        if self.kind == "node":
            return binding[self.ref]
        seq = []
        for i, var in enumerate(self.ref):
            seq.append(binding[var])
            if i < len(self.edges):
                seq.append(relations.get(self.edges[i]))
        return tuple(seq)


class SearchSpec:
    """A conjunctive pattern plus the columns to return."""

    def __init__(self, nodes, edges, outputs):
        self.nodes = nodes        # {var: NodeSpec}
        self.edges = edges        # [EdgeSpec]
        self.outputs = outputs    # [Output]


class SearchResult:
    """What an Engine returns: matched rows plus every node it walked through
    (the latter feeds ranking strategies that need the touched subgraph)."""

    def __init__(self, rows, visited):
        self.rows = rows
        self.visited = visited
