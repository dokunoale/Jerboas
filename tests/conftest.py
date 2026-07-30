"""Shared fixtures: a small, deterministic synthetic graph (not the real
MovieLens data), built through the real file loader so the loader itself is
exercised, not bypassed.

Node keys are `<type>.<id>` with a numeric id, the shape the loader assigns a
contiguous block of integers to.

Layout:
  movie.1  directed_by person.1  has_genre genre.1   title="Alpha"  year=1994
  movie.2  directed_by person.1  has_genre genre.1   title="Beta"   year=1994
  movie.3  directed_by person.2  has_genre genre.2   title="Gamma"  year=1999
  user.1 watched movie.1, movie.2
  user.2 watched movie.2, movie.3
  user.3 watched movie.1, movie.3
"""

import pytest

from jerboas import Graph

KG_ROWS = [
    ("movie.1", "directed_by", "person.1"),
    ("movie.2", "directed_by", "person.1"),
    ("movie.3", "directed_by", "person.2"),
    ("movie.1", "has_genre", "genre.1"),
    ("movie.2", "has_genre", "genre.1"),
    ("movie.3", "has_genre", "genre.2"),
]

UI_ROWS = [
    ("user.1", "movie.1"),
    ("user.1", "movie.2"),
    ("user.2", "movie.2"),
    ("user.2", "movie.3"),
    ("user.3", "movie.1"),
    ("user.3", "movie.3"),
]

MOVIE_ATTRS = [
    ("id", "title", "year"),
    ("1", "Alpha", "1994"),
    ("2", "Beta", "1994"),
    ("3", "Gamma", "1999"),
]

PERSON_ATTRS = [
    ("id", "name"),
    ("1", "Xavier Director"),
    ("2", "Yara Director"),
]

GENRE_ATTRS = [
    ("id", "name"),
    ("1", "Comedy"),
    ("2", "Drama"),
]


def _write_tsv(path, rows):
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n")


@pytest.fixture
def graph_rows():
    """The fixture's raw rows, for tests that need to build a second graph from
    the same data in a different order. A fixture rather than a module import:
    `tests` is a package name other installed projects also use."""
    return {"kg": KG_ROWS, "ui": UI_ROWS, "movie": MOVIE_ATTRS,
            "person": PERSON_ATTRS, "genre": GENRE_ATTRS}


@pytest.fixture
def small_graph(tmp_path):
    kg_path = tmp_path / "test.kg"
    ui_path = tmp_path / "test.ui"
    movie_path = tmp_path / "test.movie"
    person_path = tmp_path / "test.person"
    genre_path = tmp_path / "test.genre"

    _write_tsv(kg_path, KG_ROWS)
    _write_tsv(ui_path, UI_ROWS)
    _write_tsv(movie_path, MOVIE_ATTRS)
    _write_tsv(person_path, PERSON_ATTRS)
    _write_tsv(genre_path, GENRE_ATTRS)

    return Graph(
        kg=str(kg_path),
        ui=str(ui_path),
        attrs=[str(movie_path), str(person_path), str(genre_path)],
    )
