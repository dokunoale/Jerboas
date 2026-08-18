"""Shared fixtures: a small, deterministic synthetic graph (not the real
MovieLens data), built through the real file loader so the loader itself is
exercised, not bypassed.

Node keys are `<type>.<id>` with a numeric id, the shape the loader assigns a
contiguous block of integers to.

Layout:
  movie.0  directed_by person.0  has_genre genre.0   title="Alpha"  year=1994
  movie.1  directed_by person.0  has_genre genre.0   title="Beta"   year=1994
  movie.2  directed_by person.1  has_genre genre.1   title="Gamma"  year=1999
  user.0 watched movie.0 (5), movie.1 (1)
  user.1 watched movie.1 (4), movie.2 (2)
  user.2 watched movie.0 (3), movie.2 (5)

The interactions carry a score, so the loader's weighted path is the one under
test: every edge is kept, and what a rating of 1 means is the query's business.
"""

import pytest

from jerboas import Graph

KG_ROWS = [
    ("movie.0", "directed_by", "person.0"),
    ("movie.1", "directed_by", "person.0"),
    ("movie.2", "directed_by", "person.1"),
    ("movie.0", "has_genre", "genre.0"),
    ("movie.1", "has_genre", "genre.0"),
    ("movie.2", "has_genre", "genre.1"),
]

EDGE_HEADER = ("source", "target", "score")

INTERACT_ROWS = [
    ("user.0", "movie.0", "5"),
    ("user.0", "movie.1", "1"),
    ("user.1", "movie.1", "4"),
    ("user.1", "movie.2", "2"),
    ("user.2", "movie.0", "3"),
    ("user.2", "movie.2", "5"),
]

MOVIE_ATTRS = [
    ("id", "title", "year"),
    ("0", "Alpha", "1994"),
    ("1", "Beta", "1994"),
    ("2", "Gamma", "1999"),
]

PERSON_ATTRS = [
    ("id", "name"),
    ("0", "Xavier Director"),
    ("1", "Yara Director"),
]

GENRE_ATTRS = [
    ("id", "name"),
    ("0", "Comedy"),
    ("1", "Drama"),
]


def _write_tsv(path, rows):
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n")


@pytest.fixture
def graph_rows():
    """The fixture's raw rows, for tests that need to build a second graph from
    the same data in a different order. A fixture rather than a module import:
    `tests` is a package name other installed projects also use."""
    return {"kg": KG_ROWS, "has_interact": INTERACT_ROWS, "movie": MOVIE_ATTRS,
            "person": PERSON_ATTRS, "genre": GENRE_ATTRS}


@pytest.fixture
def small_graph(tmp_path):
    kg_path = tmp_path / "test.kg"
    interact_path = tmp_path / "test.has_interact"
    movie_path = tmp_path / "test.movie"
    person_path = tmp_path / "test.person"
    genre_path = tmp_path / "test.genre"

    _write_tsv(kg_path, KG_ROWS)
    _write_tsv(interact_path, [EDGE_HEADER] + INTERACT_ROWS)
    _write_tsv(movie_path, MOVIE_ATTRS)
    _write_tsv(person_path, PERSON_ATTRS)
    _write_tsv(genre_path, GENRE_ATTRS)

    return Graph(
        kg=str(kg_path),
        edges=[str(interact_path)],
        attrs=[str(movie_path), str(person_path), str(genre_path)],
    )
