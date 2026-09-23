"""A finalized graph on disk, mapped back instead of rebuilt.

Loading reads every edge once and runs Python once per distinct node: on the
whole Spotify graph that is 77 s, and it is the same 77 s every time a process
starts. What it produces is a handful of arrays that never change afterwards --
the two CSRs, the type blocks, the attribute columns -- so the second load can
map them from disk instead:

    g = jb.Graph(kg=..., edges=[...], attrs=[...], cache="data/spotify/.jbcache")

A directory of `.npy` files rather than one `.npz`, because `np.load` ignores
`mmap_mode` for an archive and reads the whole of it into memory; a `.npy`
opened with `mmap_mode="r"` is paged in as a query touches it, and shared
between processes that map the same file. Text columns are Arrow IPC, the one
format polars reads without parsing.

A cache is only valid for the files it was built from. The manifest records
each input's path, size and modification time, and a cache whose record does not
match what is on disk now is rebuilt rather than trusted -- a stale graph
answers every query, just wrongly.
"""

import json
import os
import shutil

import numpy as np
import polars as pl

from .columns import Column

FORMAT = 1

# the finalized arrays, by the attribute of Graph that holds them
_ARRAYS = ("start", "_type_tag_of",
           "out_indptr", "out_indices", "out_rels", "out_weights",
           "in_indptr", "in_indices", "in_rels", "in_weights")


def fingerprint(files, renumber):
    """What a cache has to have been built from to be this graph."""
    record = []
    for path in files:
        stat = os.stat(path)
        record.append([os.path.abspath(path), stat.st_size, stat.st_mtime_ns])
    return {"format": FORMAT, "renumber": bool(renumber), "files": record}


def load(directory, expected):
    """The saved state, or None when there is none or it is for other files."""
    manifest_path = os.path.join(directory, "manifest.json")
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    if manifest.get("fingerprint") != expected:
        return None

    def array(name):
        return np.load(os.path.join(directory, name), mmap_mode="r",
                       allow_pickle=False)

    state = {name: array(f"{name.strip('_')}.npy") for name in _ARRAYS}
    state["types"] = manifest["types"]
    state["relations"] = manifest["relations"]
    state["n_nodes"] = manifest["n_nodes"]

    texts = {}
    columns = {}
    for type_, entries in manifest["columns"].items():
        table = {}
        for entry in entries:
            name, kind = entry["name"], entry["kind"]
            if kind == "alias":
                table[name] = table[entry["of"]]
                continue
            if kind == "text":
                if type_ not in texts:
                    texts[type_] = pl.read_ipc(os.path.join(directory, entry["file"]),
                                               memory_map=True)
                values = texts[type_][entry["field"]].to_numpy()
                table[name] = Column(np.asarray(values, dtype=object))
                continue
            values = array(entry["file"])
            present = array(entry["present"]) if entry.get("present") else None
            table[name] = Column(values, present)
        columns[type_] = table
    state["columns"] = columns
    state["vectors"] = {type_: {name: array(file) for name, file in table.items()}
                        for type_, table in manifest["vectors"].items()}
    return state


def save(graph, directory, stamp):
    """Write a finalized graph where `load` will find it.

    Written beside the target and renamed into place, so a process that dies
    halfway leaves no cache rather than half of one."""
    parent = os.path.dirname(os.path.abspath(directory))
    os.makedirs(parent, exist_ok=True)
    staging = f"{os.path.abspath(directory)}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)

    def put(name, values):
        np.save(os.path.join(staging, name), np.ascontiguousarray(values),
                allow_pickle=False)
        return name

    for name in _ARRAYS:
        put(f"{name.strip('_')}.npy", getattr(graph, name))

    columns = {}
    for tag, (type_, table) in enumerate(graph.columns.items()):
        entries, texts, seen = [], {}, {}
        for position, (name, column) in enumerate(table.items()):
            if id(column) in seen:            # `label` is `id` when nothing was renumbered
                entries.append({"name": name, "kind": "alias", "of": seen[id(column)]})
                continue
            seen[id(column)] = name
            if column.values.dtype == object:
                field = f"c{position}"
                texts[field] = pl.Series(field, column.values.tolist(), dtype=pl.String)
                entries.append({"name": name, "kind": "text", "field": field,
                                "file": f"text.{tag}.arrow"})
                continue
            entry = {"name": name, "kind": "array",
                     "file": put(f"column.{tag}.{position}.npy", column.values)}
            if column.present is not None:
                entry["present"] = put(f"present.{tag}.{position}.npy", column.present)
            entries.append(entry)
        if texts:
            pl.DataFrame(list(texts.values())).write_ipc(
                os.path.join(staging, f"text.{tag}.arrow"), compression="uncompressed")
        columns[type_] = entries

    vectors = {}
    for tag, (type_, table) in enumerate(graph.vectors.items()):
        vectors[type_] = {name: put(f"vector.{tag}.{position}.npy", block)
                          for position, (name, block) in enumerate(table.items())}

    manifest = {"fingerprint": stamp, "types": list(graph.types),
                "relations": list(graph.relations), "n_nodes": int(graph.n_nodes),
                "columns": columns, "vectors": vectors}
    with open(os.path.join(staging, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=1)

    shutil.rmtree(directory, ignore_errors=True)
    os.replace(staging, directory)
