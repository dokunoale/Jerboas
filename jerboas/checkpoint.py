"""Storing a trained embedding, and binding it back to a graph.

Mostly this module is about *identity*, not bytes. A node's integer id is an
artifact of how its graph was loaded: blocks are laid out in order of first
appearance, so adding one movie, or listing the attribute files differently,
shifts every id after it. A checkpoint keyed by position would still load against
the changed graph -- the shapes match -- and would score the wrong entities
without raising. So every weight row records whose it is, and binding resolves
those names against whatever ids the caller's graph is using now. The same goes
for relations, by name.

The file is a compressed .npz, and it is *inert*: every array is a native numpy
dtype, strings included, so it loads with `allow_pickle=False`. A pickled .npz is
executable code wearing a data extension; opening one from an untrusted source
would run it. Nothing here can.

    format        int      this layout's version
    model         str      which model wrote it: "transd", "transe", ...
    factors       int      embedding width

    relations     U[R]     relation name per code, in the trained model's order
    node_type     U[N]     the type of each weight row
    node_id       U[N]     the raw id of each weight row

    meta_*        scalar   provenance: when, for how long, on what

    w_<table>     f32      one per table the model declares
"""

from datetime import datetime, timezone

import numpy as np

FORMAT = 2
META = "meta_"
WEIGHT = "w_"
RELATION = "relation"


def provenance(graph, **details):
    """The record of a training run: enough to tell two checkpoints apart six
    months later, when only the files are left."""
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_nodes": graph.n_nodes,
        "graph_edges": int(len(graph.out_indices)),
        "graph_relations": len(graph.relations),
        **details,
    }


def save(path, model, tables, factors, graph, arrays, meta=None):
    """Write weights plus the identity needed to rebind them."""
    missing = {table for table, _space in tables} - set(arrays)
    if missing:
        raise ValueError(f"{model} checkpoint is missing tables: {sorted(missing)}")

    payload = {
        "format": np.asarray(FORMAT),
        "model": np.asarray(model),
        "factors": np.asarray(factors),
        "relations": np.asarray(list(graph.relations), dtype="U"),
        "node_type": np.asarray([graph.type_of(i) for i in range(graph.n_nodes)], dtype="U"),
        "node_id": np.asarray([str(graph.raw_id(i)) for i in range(graph.n_nodes)], dtype="U"),
    }
    for key, value in (meta or {}).items():
        payload[META + key] = np.asarray(value)
    for table, _space in tables:
        payload[WEIGHT + table] = np.asarray(arrays[table], dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


class Stored:
    """A checkpoint's contents, already rebound to a graph's id space."""

    def __init__(self, factors, tensors, missing_nodes, missing_relations, meta):
        self.factors = factors
        self.tensors = tensors                    # table -> array, in *this* graph's ids
        self.missing_nodes = missing_nodes        # nodes the checkpoint never saw
        self.missing_relations = missing_relations
        self.meta = meta


def load(path, graph, model, tables):
    """Read a checkpoint written by `model` and rebind it to `graph`, by name.

    Rows the checkpoint does not cover stay zero -- no evidence, no signal, the
    convention an unobserved item already gets in matrix factorization -- and are
    reported so a caller can tell "unknown" from "uninteresting".
    """
    with np.load(path, allow_pickle=False) as data:
        stored = {key: data[key] for key in data.files}

    version = int(stored["format"])
    if version != FORMAT:
        raise ValueError(f"{path}: checkpoint format {version}, expected {FORMAT}")
    written_by = str(stored["model"])
    if written_by != model:
        raise ValueError(f"{path}: written by {written_by!r}, not {model!r}")

    factors = int(stored["factors"])
    node_rows, missing_nodes = _node_rows(stored, graph)
    relation_rows, missing_relations = _relation_rows(stored, graph)

    tensors = {}
    for table, space in tables:
        weights = stored[WEIGHT + table]
        rows, size = ((relation_rows, len(graph.relations)) if space == RELATION
                      else (node_rows, graph.n_nodes))
        rebound = np.zeros((size, factors), dtype=weights.dtype)
        covered = rows >= 0
        rebound[covered] = weights[rows[covered]]
        tensors[table] = rebound

    meta = {key[len(META):]: stored[key].item() for key in stored if key.startswith(META)}
    return Stored(factors, tensors, missing_nodes, missing_relations, meta)


def _node_rows(stored, graph):
    """For each node of `graph`, the checkpoint row holding its weights, matched
    on (type, raw id); -1 where the checkpoint has none."""
    trained = {(str(type_), str(raw)): row for row, (type_, raw)
               in enumerate(zip(stored["node_type"].tolist(), stored["node_id"].tolist()))}
    rows = np.full(graph.n_nodes, -1, dtype=np.int64)
    for index in range(graph.n_nodes):
        row = trained.get((graph.type_of(index), str(graph.raw_id(index))))
        if row is not None:
            rows[index] = row
    return rows, [int(i) for i in np.flatnonzero(rows < 0)]


def _relation_rows(stored, graph):
    """The same, for relations, matched on name."""
    trained = {str(name): row for row, name in enumerate(stored["relations"].tolist())}
    rows = np.full(len(graph.relations), -1, dtype=np.int64)
    missing = []
    for code, name in enumerate(graph.relations):
        if name in trained:
            rows[code] = trained[name]
        else:
            missing.append(code)
    return rows, missing
