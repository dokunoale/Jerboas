"""How a trained embedding is stored, and how it is bound back to a graph.

Torch-free and numpy-only on purpose: this is the seam between training
(jerboas.models, which needs torch) and serving (strategies.Embedding, which does
not). A base install can load a checkpoint someone else fitted and rank with it.

The file is a compressed .npz, and it is *inert*: every array is a native numpy
dtype, strings included, so it loads with `allow_pickle=False`. That matters
because a pickled .npz is executable code wearing a data extension -- opening one
from an untrusted source would run it. Nothing here can.

    format        int      this layout's version
    model         str      a key into kge.REGISTRY: "transd", "transe", ...
    factors       int      embedding width

    relations     U[R]     relation name per code, in the trained model's order
    node_type     U[N]     the type of each weight row
    node_id       U[N]     the raw id of each weight row

    meta_*        scalar   provenance: when, for how long, on what

    w_<table>     f32      one per table the model declares

Why identity is stored
----------------------
A node's integer id is an artifact of *how its graph was loaded*: blocks are laid
out in order of first appearance, so adding one movie, or listing the attribute
files differently, shifts every id after it. A checkpoint keyed by position would
still load against the changed graph -- the shapes match -- and would score the
wrong entities without raising. So each weight row records whose it is, and
binding resolves those names against whatever ids the graph is using now.

The same applies to relations, by name. After `load`, every table is indexed in
the *caller's* space: `tensors["entity"][node]` and `tensors["relation"][code]`
both take ids from the graph that was passed in, not from the one trained on.
"""

from datetime import datetime, timezone

import numpy as np

from .kge import NODE, REGISTRY

FORMAT = 2
META = "meta_"
WEIGHT = "w_"


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


def save(path, model, factors, graph, weights, meta=None):
    """Write weights plus the identity needed to rebind them.

    `weights` maps a table name the model declares to its array."""
    missing = set(model.names) - set(weights)
    if missing:
        raise ValueError(f"{model.name} checkpoint is missing tables: {sorted(missing)}")

    payload = {
        "format": np.asarray(FORMAT),
        "model": np.asarray(model.name),
        "factors": np.asarray(factors),
        "relations": np.asarray(list(graph.relations), dtype="U"),
        "node_type": np.asarray([graph.type_of(i) for i in range(graph.n_nodes)], dtype="U"),
        "node_id": np.asarray([str(graph.raw_id(i)) for i in range(graph.n_nodes)], dtype="U"),
    }
    for key, value in (meta or {}).items():
        payload[META + key] = np.asarray(value)
    for name in model.names:
        payload[WEIGHT + name] = np.asarray(weights[name], dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


class Checkpoint:
    """A trained embedding, already rebound to a graph's id space."""

    def __init__(self, model, factors, tensors, missing_nodes, missing_relations, meta):
        self.model = model                        # the kge.Model, not just its name
        self.factors = factors
        self.tensors = tensors                    # name -> array, in *this* graph's ids
        self.missing_nodes = missing_nodes        # nodes the checkpoint never saw
        self.missing_relations = missing_relations
        self.meta = meta

    @property
    def name(self):
        return self.model.name

    def knows(self, code):
        """Whether the model was trained on this graph's relation `code`."""
        return code is not None and code not in self.missing_relations

    def gather(self, table, index):
        return self.tensors[table][index]

    def score(self, head, relation, tail):
        """Plausibility of triples, using the model's own arithmetic."""
        return self.model.score(self.gather, head, relation, tail)


def load(path, graph):
    """Read a checkpoint and rebind it to `graph`, by name rather than position.

    Rows the checkpoint does not cover stay zero -- no evidence, no signal, the
    convention an unobserved item already gets in matrix factorization -- and are
    reported so a caller can tell "unknown" from "uninteresting".
    """
    with np.load(path, allow_pickle=False) as data:
        stored = {key: data[key] for key in data.files}

    version = int(stored["format"])
    if version != FORMAT:
        raise ValueError(f"{path}: checkpoint format {version}, expected {FORMAT}")

    name = str(stored["model"])
    if name not in REGISTRY:
        raise ValueError(f"{path}: unknown model {name!r}; known: {sorted(REGISTRY)}")
    model = REGISTRY[name]
    factors = int(stored["factors"])

    node_rows, missing_nodes = _node_rows(stored, graph)
    relation_rows, missing_relations = _relation_rows(stored, graph)

    tensors = {}
    for table in model.tables:
        weights = stored[WEIGHT + table.name]
        rows, size = ((node_rows, graph.n_nodes) if table.space == NODE
                      else (relation_rows, len(graph.relations)))
        rebound = np.zeros((size, factors), dtype=weights.dtype)
        covered = rows >= 0
        rebound[covered] = weights[rows[covered]]
        tensors[table.name] = rebound

    meta = {key[len(META):]: stored[key].item() for key in stored if key.startswith(META)}
    return Checkpoint(model, factors, tensors, missing_nodes, missing_relations, meta)


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
    missing = [int(i) for i in np.flatnonzero(rows < 0)]
    return rows, missing


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
