"""The on-disk format of a trained embedding, and how it is bound back to a graph.

This module is deliberately torch-free and numpy-only: it is the seam between
training (jerboas.models, which needs torch) and serving (strategies.Embedding,
which does not). Installing the library without the `torch` extra still lets you
load a checkpoint someone else trained and rank with it.

Why identity is stored, not just weights
----------------------------------------
A node's integer id is an artifact of *how the graph was loaded*: blocks are laid
out in order of first appearance, so adding one movie, or listing the attribute
files in a different order, shifts the ids. A checkpoint keyed by position would
keep working after such a change and silently score the wrong entities -- the
worst kind of bug, because nothing raises.

So a checkpoint records the identity of every row: the type names, the raw ids
within each type, and the relation names in code order. Binding it to a graph
resolves those names to whatever ids that graph is using now, and reports what it
could not find. The identity table costs a few hundred KB against several MB of
weights.
"""

import numpy as np

FORMAT = 1


def save(path, model, factors, graph, tensors):
    """Write weights plus the identity needed to rebind them.

    `tensors` maps a name to an array whose rows are indexed by node id, except
    `relation` and `relation_vec`, which are indexed by relation code.
    """
    payload = {
        "format": np.asarray(FORMAT),
        "model": np.asarray(model),
        "factors": np.asarray(factors),
        "relations": np.asarray(graph.relations, dtype=object),
        "types": np.asarray(graph.types, dtype=object),
        # one raw id per node, in node order: the identity of every weight row
        "node_type": np.asarray([graph.type_of(i) for i in range(graph.n_nodes)], dtype=object),
        "node_id": np.asarray([str(graph.raw_id(i)) for i in range(graph.n_nodes)], dtype=object),
    }
    payload.update({f"w_{name}": np.asarray(value) for name, value in tensors.items()})
    np.savez_compressed(path, **payload)


class Checkpoint:
    """A trained embedding, already rebound to a graph's id space."""

    def __init__(self, model, factors, tensors, relations, missing):
        self.model = model            # "transd", ...
        self.factors = factors
        self.tensors = tensors        # name -> array, rows in *this* graph's ids
        self.relations = relations    # relation code in this graph -> row in the weights
        self.missing = missing        # nodes of this graph the checkpoint never saw

    def relation_row(self, code):
        """The weight row for a relation code, or None when this graph has a
        relation the trained model never saw."""
        return self.relations.get(code)


def load(path, graph):
    """Read a checkpoint and rebind it to `graph`, by name rather than position.

    Nodes the checkpoint does not cover keep a zero row: no evidence, no signal,
    the same convention an unobserved item gets in matrix factorization. They are
    listed in `.missing` so a caller can tell "unknown" from "uninteresting".
    """
    with np.load(path, allow_pickle=True) as data:
        stored = {key: data[key] for key in data.files}

    if int(stored["format"]) != FORMAT:
        raise ValueError(f"{path}: checkpoint format {int(stored['format'])}, expected {FORMAT}")

    factors = int(stored["factors"])
    row_of_node = _node_rows(stored, graph)
    covered = row_of_node >= 0

    tensors = {}
    for key, value in stored.items():
        if not key.startswith("w_"):
            continue
        name = key[2:]
        if name.startswith("relation"):
            tensors[name] = value          # indexed by relation row, remapped below
            continue
        rebound = np.zeros((graph.n_nodes, factors), dtype=value.dtype)
        rebound[covered] = value[row_of_node[covered]]
        tensors[name] = rebound

    trained = {str(name): row for row, name in enumerate(stored["relations"].tolist())}
    relations = {code: trained[name]
                 for code, name in enumerate(graph.relations) if name in trained}

    missing = [int(i) for i in np.flatnonzero(~covered)]
    return Checkpoint(str(stored["model"]), factors, tensors, relations, missing)


def _node_rows(stored, graph):
    """For each node of `graph`, the checkpoint row holding its weights (-1 if
    the checkpoint never saw it). Matching is on (type, raw id)."""
    row_of_identity = {}
    for row, (type_, raw) in enumerate(zip(stored["node_type"].tolist(),
                                           stored["node_id"].tolist())):
        row_of_identity[(str(type_), str(raw))] = row

    rows = np.full(graph.n_nodes, -1, dtype=np.int64)
    for index in range(graph.n_nodes):
        row = row_of_identity.get((graph.type_of(index), str(graph.raw_id(index))))
        if row is not None:
            rows[index] = row
    return rows
