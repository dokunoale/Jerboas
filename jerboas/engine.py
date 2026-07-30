"""Search engines: resolve a SearchSpec (ir.py) against a Graph.

Engines only ever see the crisp IR -- an admission mask per variable and the
edges between them -- never a Ref or a Condition, and never anything fuzzy
(Like's softness lives in the ranking layer; the engine sees only its widened
crisp support).

Two things the engines used to do are gone. Testing a candidate is no longer a
sequence of type/id/attribute/degree checks but a single `mask[node]`, because
the compiler folded all of them into one array. And seeding a variable no longer
branches over key sets, id sets and type partitions: the mask already encodes
every one of those, so a root is `flatnonzero(mask)`.

What is left is the part that genuinely cannot be vectorized: matching a pattern
of arbitrary shape is a backtracking walk, and it stays a Python one.
"""

from itertools import product

import numpy as np

from .core import Engine
from .ir import SearchResult


def _plan(spec):
    incoming, children = {}, {}
    for edge in spec.edges:
        incoming[edge.target] = edge
        children.setdefault(edge.source, []).append(edge)
    roots = [v for v in spec.nodes if v not in incoming]
    return incoming, children, roots


def _order(spec, children, roots):
    order, seen, queue = [], set(), list(roots)
    while queue:
        var = queue.pop(0)
        if var in seen:
            continue
        seen.add(var)
        order.append(var)
        for edge in children.get(var, []):
            queue.append(edge.target)
    return order


def _seed(spec, var):
    """Every node this variable admits, straight off its mask."""
    return np.flatnonzero(spec.nodes[var].mask).tolist()


def _neighbors(graph, node, edge):
    """[(target, code)] for one pattern edge; a negative code marks a traversal
    made against the stored direction."""
    return graph.expand(node, edge.relation, edge.reverse)


class Default(Engine):
    """Exact search. Memoized over shared sub-paths for node-only outputs; falls
    back to full path enumeration when a path column must be materialized."""

    def search(self, graph, spec, guide=None):
        if any(output.kind == "path" for output in spec.outputs):
            return self._exhaustive(graph, spec)
        return self._memoized(graph, spec)

    def _exhaustive(self, graph, spec):
        incoming, children, roots = _plan(spec)
        order = _order(spec, children, roots)
        masks = {var: node.mask for var, node in spec.nodes.items()}

        rows, visited, binding, relations_used = set(), set(), {}, {}

        def candidates(var, edge):
            if edge is not None:
                return _neighbors(graph, binding[edge.source], edge)
            return [(node, None) for node in _seed(spec, var)]

        def assign(depth):
            if depth == len(order):
                rows.add(tuple(o.extract(binding, relations_used) for o in spec.outputs))
                return
            var = order[depth]
            edge = incoming.get(var)
            mask = masks[var]
            for node, relation in candidates(var, edge):
                if not mask[node]:
                    continue
                binding[var] = node
                if edge is not None:
                    relations_used[edge] = relation
                visited.add(node)
                assign(depth + 1)
            binding.pop(var, None)

        assign(0)
        return SearchResult(sorted(rows), visited)

    def _memoized(self, graph, spec):
        incoming, children, roots = _plan(spec)
        masks = {var: node.mask for var, node in spec.nodes.items()}

        col_of_var = {output.ref: col for col, output in enumerate(spec.outputs)}

        subtree_cols = {}
        def cols_of(var):
            if var in subtree_cols:
                return subtree_cols[var]
            cols = set()
            if var in col_of_var:
                cols.add(col_of_var[var])
            for edge in children.get(var, []):
                cols.update(cols_of(edge.target))
            subtree_cols[var] = sorted(cols)
            return subtree_cols[var]
        for var in spec.nodes:
            cols_of(var)

        visited = set()
        memo = {}

        def comp(var, node):
            if (var, node) in memo:
                return memo[(var, node)]
            if not masks[var][node]:
                memo[(var, node)] = frozenset()
                return memo[(var, node)]

            child_cols, child_sets, satisfiable = [], [], True
            for edge in children.get(var, []):
                reachable = set()
                for target, _relation in _neighbors(graph, node, edge):
                    reachable |= comp(edge.target, target)
                if not reachable:
                    satisfiable = False
                    break
                child_cols.append(subtree_cols[edge.target])
                child_sets.append(reachable)
            if not satisfiable:
                memo[(var, node)] = frozenset()
                return memo[(var, node)]

            visited.add(node)
            own_col = col_of_var.get(var)
            cols = subtree_cols[var]
            result = set()
            for combo in (product(*child_sets) if child_sets else [()]):
                slot = {own_col: node} if own_col is not None else {}
                for cs, tup in zip(child_cols, combo):
                    slot.update(zip(cs, tup))
                result.add(tuple(slot[c] for c in cols))
            memo[(var, node)] = frozenset(result)
            return memo[(var, node)]

        root_cols, root_sets = [], []
        for root in roots:
            reachable = set()
            for node in _seed(spec, root):
                reachable |= comp(root, node)
            if not reachable:
                return SearchResult([], visited)
            root_cols.append(subtree_cols[root])
            root_sets.append(reachable)

        width = len(spec.outputs)
        rows = set()
        for combo in (product(*root_sets) if root_sets else [()]):
            slot = {}
            for cs, tup in zip(root_cols, combo):
                slot.update(zip(cs, tup))
            rows.add(tuple(slot[c] for c in range(width)))
        return SearchResult(sorted(rows), visited)


class Greedy(Engine):
    """Approximate beam search: guided by a Strategy's edge weights, keep only the
    k most promising partial paths at each step. Falls back to Default when there
    is no guide to prune with."""

    def __init__(self, k=5):
        self.k = k

    def search(self, graph, spec, guide=None):
        if guide is None or not guide.supports_guidance:
            return Default().search(graph, spec, guide)

        incoming, children, roots = _plan(spec)
        order = _order(spec, children, roots)
        masks = {var: node.mask for var, node in spec.nodes.items()}

        rows, visited = set(), set()

        for seed in _seed(spec, order[0]):
            if not masks[order[0]][seed]:
                continue
            beam = [({order[0]: seed}, 0.0)]
            for var in order[1:]:
                edge = incoming[var]
                mask = masks[var]
                expanded = []
                for binding, score in beam:
                    source = binding[edge.source]
                    for target, relation in _neighbors(graph, source, edge):
                        if not mask[target]:
                            continue
                        step = guide.edge_weight(source, relation, target)
                        nxt = dict(binding)
                        nxt[var] = target
                        expanded.append((nxt, score + step))
                expanded.sort(key=lambda item: item[1], reverse=True)
                beam = expanded[:self.k]
            for binding, _score in beam:
                visited.update(binding.values())
                rows.add(tuple(output.extract(binding, {}) for output in spec.outputs))

        return SearchResult(sorted(rows), visited)
