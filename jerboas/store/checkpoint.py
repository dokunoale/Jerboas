"""Storing a trained embedding, and binding it back to a graph.

Mostly this is about *identity*, not bytes. A checkpoint keyed by position would
load against a graph built from a different slice of the source -- the shapes
match -- and score the wrong entities without raising. So every weight row
records whose it is, and binding resolves that against the ids the caller's
graph is using now; relations likewise, by name.

*Whose it is* is a column the checkpoint names. By default the node's own id;
when ids are not durable and the identifier lives in an attribute -- the way
`spotify.song` keeps `uri` -- the checkpoint records that instead, falling back
to the id for a type that has no such column:

    model.save("checkpoints/spotify.transd.npz", alias="uri")
    TransD.load("checkpoints/spotify.transd.npz", graph)     # reads the alias back

The file is a compressed .npz, and it is *inert*: every array is a native numpy
dtype, strings included, so it loads with `allow_pickle=False`. A pickled .npz
is executable code wearing a data extension. Nothing here is.

    format        int      this layout's version
    model         str      which model wrote it: "transd", "transe", ...
    factors       int      embedding width

    relations     U[R]     relation name per code, in the trained model's order
    node_type     U[N]     the type of each weight row
    node_id       U[N]     the identity of each weight row, read through `alias`
    alias         str      the attribute node_id holds; "id" for the node's own

    meta_*        scalar   provenance: when, for how long, on what

    w_<table>     f32      one per table the model declares
"""

from datetime import datetime, timezone

import numpy as np
import polars as pl

FORMAT = 3
# format 2 is format 3 without `alias`, which is to say with alias "id" -- the
# only difference is a key that did not exist, so it is read rather than refused.
# The reverse is not allowed: a reader that ignored `alias` would rebind a
# uri-keyed checkpoint onto raw ids and score the wrong entities in silence,
# which is the one failure this module exists to prevent.
READABLE = (2, 3)
META = "meta_"
WEIGHT = "w_"
NODE = "node"                    # a table with one row per node
RELATION = "relation"            # a table with one row per relation
IDENTITY = "id"                  # the alias meaning "the node's own id"


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


def identities(graph, alias, nodes=None):
    """What each node is called, for the purpose of rebinding: the value of
    `alias` where the node's type has that column, and its own id where it does
    not -- so a graph of mixed types needs one alias, not one per type.

    A type block at a time rather than a node at a time: on the whole Spotify
    graph that is four gathers instead of four million Python calls."""
    nodes = (np.arange(graph.n_nodes) if nodes is None
             else np.asarray(nodes, dtype=np.int64))
    names = np.empty(len(nodes), dtype=object)
    tags = graph._type_tag_of[nodes]
    for tag, type_ in enumerate(graph.types):
        where = np.flatnonzero(tags == tag)
        if not len(where):
            continue
        local = nodes[where] - int(graph.start[tag])
        named = local.astype(str).astype(object)
        column = None if alias == IDENTITY else graph.column(type_, alias)
        if column is not None:
            values = column.values[local]
            held = (np.ones(len(local), dtype=bool) if column.present is None
                    else np.asarray(column.present[local]))
            if values.dtype == object:
                held &= np.array([value is not None for value in values], dtype=bool)
            named[held] = values[held].astype(str)
        names[where] = named
    return names


def _types(graph, nodes=None):
    types = np.asarray(graph.types, dtype="U")
    tags = graph._type_tag_of if nodes is None else graph._type_tag_of[nodes]
    return types[tags]


def save(path, model, tables, factors, graph, arrays, meta=None, alias=IDENTITY,
         nodes=None):
    """Write weights plus the identity needed to rebind them.

    `alias` names the attribute that identifies a node durably. It defaults to
    the node's own id, and is stored, so a reader does not have to be told.

    `nodes` stores only those nodes, and then a node table holds one row per
    node in `nodes`, in that order -- a factorization has weights for two type
    blocks and nothing for the rest, a row nobody stores loads as zero anyway,
    and building the whole (N, factors) table only to cut it back would be the
    largest array of the save."""
    missing = {table for table, _space in tables} - set(arrays)
    if missing:
        raise ValueError(f"{model} checkpoint is missing tables: {sorted(missing)}")

    payload = {
        "format": np.asarray(FORMAT),
        "model": np.asarray(model),
        "factors": np.asarray(factors),
        "alias": np.asarray(alias),
        "relations": np.asarray(list(graph.relations), dtype="U"),
        "node_type": _types(graph, nodes),
        "node_id": identities(graph, alias, nodes).astype("U"),
    }
    for key, value in (meta or {}).items():
        payload[META + key] = np.asarray(value)
    for table, space in tables:
        weights = np.asarray(arrays[table], dtype=np.float32)
        expected = (len(graph.relations) if space == RELATION
                    else graph.n_nodes if nodes is None else len(nodes))
        if len(weights) != expected:
            raise ValueError(f"{model} table {table!r} has {len(weights)} rows, "
                             f"expected {expected}")
        payload[WEIGHT + table] = weights
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


def load(path, graph, model, tables, alias=None):
    """Read a checkpoint written by `model` and rebind it to `graph`, by name.

    The alias comes off the file; passing one here overrides it, for the case
    where two graphs carry the same identifier under different column names.

    Rows the checkpoint does not cover stay zero -- no evidence, no signal, the
    convention an unobserved item already gets in matrix factorization -- and are
    reported so a caller can tell "unknown" from "uninteresting".
    """
    with np.load(path, allow_pickle=False) as data:
        stored = {key: data[key] for key in data.files}

    version = int(stored["format"])
    if version not in READABLE:
        raise ValueError(f"{path}: checkpoint format {version}, expected one of {READABLE}")
    if alias is None:
        alias = str(stored["alias"]) if "alias" in stored else IDENTITY
    written_by = str(stored["model"])
    if written_by != model:
        raise ValueError(f"{path}: written by {written_by!r}, not {model!r}")

    factors = int(stored["factors"])
    node_rows, missing_nodes = _node_rows(stored, graph, alias)
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


def _node_rows(stored, graph, alias):
    """For each node of `graph`, the checkpoint row holding its weights, matched
    on (type, alias); -1 where the checkpoint has none. One join rather than a
    dictionary probed once per node. Where the checkpoint names a node twice,
    the later row wins."""
    trained = pl.DataFrame({
        "type": stored["node_type"].astype(str), "name": stored["node_id"].astype(str),
        "row": np.arange(len(stored["node_id"]), dtype=np.int64),
    }).unique(["type", "name"], keep="last")
    here = pl.DataFrame({"type": _types(graph).astype(str),
                         "name": identities(graph, alias).astype(str)})
    matched = here.join(trained, on=["type", "name"], how="left", maintain_order="left")
    rows = matched["row"].fill_null(-1).to_numpy().astype(np.int64)
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
