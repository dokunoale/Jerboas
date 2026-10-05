"""The graph cache: the second load maps what the first one built.

A cached graph has to be the graph -- every array, every column, every answer --
and a cache built from other files has to be refused rather than trusted.
"""

import os

import numpy as np
import pytest

from jerboas import Graph, v


def _write(path, rows):
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n")


@pytest.fixture
def files(tmp_path, graph_rows):
    paths = {name: tmp_path / f"test.{name}" for name in
             ("kg", "has_interact", "movie", "person", "genre")}
    _write(paths["kg"], graph_rows["kg"])
    _write(paths["has_interact"], [("source", "target", "score")] + graph_rows["has_interact"])
    for name in ("movie", "person", "genre"):
        _write(paths[name], graph_rows[name])
    return paths


def _graph(files, cache=None, renumber=False):
    return Graph(kg=str(files["kg"]), edges=[str(files["has_interact"])],
                 attrs=[str(files[name]) for name in ("movie", "person", "genre")],
                 readable={"movie": "title"}, renumber=renumber, cache=cache)


def _same(left, right):
    assert left.types == right.types and left.relations == right.relations
    assert left.n_nodes == right.n_nodes
    for name in ("start", "_type_tag_of", "out_indptr", "out_indices", "out_rels",
                 "out_weights", "in_indptr", "in_indices", "in_rels", "in_weights"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert left.columns.keys() == right.columns.keys()
    for type_, table in left.columns.items():
        assert list(table) == list(right.columns[type_])
        for name, column in table.items():
            other = right.columns[type_][name]
            assert column.values.dtype == other.values.dtype
            assert column.values.tolist() == other.values.tolist()
            assert (column.present is None) == (other.present is None)


@pytest.mark.parametrize("renumber", [False, True])
def test_a_cached_graph_is_the_graph(files, tmp_path, renumber):
    built = _graph(files, renumber=renumber)
    first = _graph(files, cache=tmp_path / "cache", renumber=renumber)
    again = _graph(files, cache=tmp_path / "cache", renumber=renumber)
    _same(built, first)
    _same(built, again)
    query = lambda g: (g.nodes(user="user").hop(rec="has_interact")
                       .filter(v.rec.year >= 1994).labels("rec").pl)
    assert query(again).equals(query(built))


def test_the_second_load_maps_the_arrays(files, tmp_path):
    _graph(files, cache=tmp_path / "cache")
    again = _graph(files, cache=tmp_path / "cache")
    assert isinstance(again.out_indices, np.memmap)
    assert again._keys is None                      # nothing was parsed


def test_the_label_is_still_the_id_when_nothing_was_renumbered(files, tmp_path):
    _graph(files, cache=tmp_path / "cache")
    again = _graph(files, cache=tmp_path / "cache")
    assert again.columns["movie"]["label"] is again.columns["movie"]["id"]


def test_a_changed_file_is_read_again(files, tmp_path):
    _graph(files, cache=tmp_path / "cache")
    rows = files["has_interact"].read_text().splitlines()
    files["has_interact"].write_text("\n".join(rows[:-1]) + "\n")   # one edge fewer
    stat = os.stat(files["has_interact"])
    os.utime(files["has_interact"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    again = _graph(files, cache=tmp_path / "cache")
    assert not isinstance(again.out_indices, np.memmap)       # read, not mapped
    assert len(again.out_indices) == len(_graph(files).out_indices)
    rewritten = _graph(files, cache=tmp_path / "cache")
    assert isinstance(rewritten.out_indices, np.memmap)       # and the cache is current
    _same(again, rewritten)


def test_the_same_data_mounted_elsewhere_finds_its_cache(files, tmp_path):
    """The host writes it, a container reading the same files under another
    root maps it: the cache names files relative to itself."""
    _graph(files, cache=tmp_path / "cache")
    moved = tmp_path.parent / (tmp_path.name + "-mounted")
    os.rename(tmp_path, moved)
    try:
        elsewhere = {name: moved / path.name for name, path in files.items()}
        again = _graph(elsewhere, cache=moved / "cache")
        assert isinstance(again.out_indices, np.memmap)
    finally:
        os.rename(moved, tmp_path)
