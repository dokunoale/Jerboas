"""Behavioural coverage for the surface: passive refs, Polars-style predicates,
Like (fuzzy + soft-where), path/OR, ranking, grouping, strategies -- plus the
values a query returns (Key, Rel) and the directed-relation model."""

import numpy as np
import pytest

from jerboas import Node, Edge, Path, Score, Like, Has, Key, Rel, Sum, Count, Max
from jerboas import Connectivity, DiffusedMatrixFactorization, MatrixFactorization, PageRank, Weight
from jerboas.strategies import Ascending, Descending


def names(rows):
    """Rows of Keys as their printable form, for comparing against literals."""
    if rows and isinstance(rows[0], tuple):
        return [tuple(str(v) for v in row) for row in rows]
    return [str(row) for row in rows]


# --- the values a query returns ---------------------------------------------

def test_select_returns_keys(small_graph):
    rows = sorted(small_graph.select(Node("movie")))
    assert all(isinstance(k, Key) for k in rows)
    assert names(rows) == ["movie.0", "movie.1", "movie.2"]


def test_key_exposes_type_id_label_attrs(small_graph):
    movie = Node("movie")
    key = list(small_graph.select(movie).where(movie.title == "Alpha"))[0]
    assert key.type == "movie"
    assert key.id == 0                       # the position
    assert key.label == 0                    # no renumbering: the identity is the position
    assert key.attrs["year"] == 1994         # typed at load, not a string


# --- an alias is what makes one query serve several types -------------------

READABLE = {"movie": "title", "person": "name", "genre": "name"}


def test_an_alias_renames_a_column_for_one_variable(small_graph):
    movie = Node("movie").alias(label="title")
    assert names(list(small_graph.select(movie).where(movie.label == "Alpha"))) == ["movie.0"]


def test_alias_returns_the_same_variable(small_graph):
    """It annotates rather than clones: a clone would be a second pattern
    variable, and the query would quietly stop meaning what it says."""
    movie = Node("movie")
    assert movie.alias(label="title") is movie


def test_an_alias_projects(small_graph):
    assert sorted(small_graph.select(Node("movie").alias(label="title").label)) \
        == ["Alpha", "Beta", "Gamma"]


def test_an_alias_resolves_per_type_on_an_untyped_node(small_graph):
    """One question across types whose columns are spelled differently -- what
    the loader used to guess at, said out loud instead."""
    anything = Node().alias(label=READABLE)
    assert names(list(small_graph.select(anything).where(anything.label == "Comedy"))) \
        == ["genre.0"]
    anything = Node().alias(label=READABLE)
    assert names(list(small_graph.select(anything).where(anything.label == "Alpha"))) \
        == ["movie.0"]


def test_a_type_the_alias_does_not_mention_keeps_the_plain_name(small_graph):
    """`user` has no readable column, so `label` stays `label` there -- and
    matches nothing rather than falling back to something invented."""
    anything = Node().alias(label=READABLE)
    rows = names(list(small_graph.select(anything).where(anything.label == "Xavier Director")))
    assert rows == ["person.0"]


def test_an_alias_drives_fuzzy_seed_resolution(small_graph):
    # the shape the coldstart use case has: one query, any type
    for needle, expected in (("Alph", "movie.0"), ("Xavier", "person.0")):
        node = Node().alias(label=READABLE)
        rows = names(list(small_graph.select(node).where(Like(node.label.is_in([needle])))))
        assert expected in rows


def test_key_indexes_arrays_directly(small_graph):
    # a Key is usable wherever its integer is: that is what lets a strategy
    # write embeddings[key] with no conversion
    key = list(small_graph.select(Node("movie")))[0]
    assert small_graph.type_of(key) == "movie"


# --- selection & projection -------------------------------------------------

def test_select_attribute(small_graph):
    assert sorted(small_graph.select(Node("movie").title)) == ["Alpha", "Beta", "Gamma"]


def test_select_id_column(small_graph):
    assert sorted(small_graph.select(Node("movie").id)) == [0, 1, 2]


def test_numeric_attribute_compares_as_a_number(small_graph):
    # the loader types the column, so all three spellings agree -- they did not
    # when every stored value was text and only some operators coerced
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.year == 1994))) == ["movie.0", "movie.1"]
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.year == "1994"))) == ["movie.0", "movie.1"]
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.year >= 1999))) == ["movie.2"]


def test_kwargs_sugar_matches_the_operator_form(small_graph):
    assert names(list(small_graph.select(Node("movie", year=1994)))) == ["movie.0", "movie.1"]


# --- relations & predicates -------------------------------------------------

def test_relation_via_eq(small_graph):
    movie, person = Node("movie"), Node("person")
    rows = set(names(list(small_graph.select(movie, person).where(movie.directed_by == person))))
    assert ("movie.0", "person.0") in rows and ("movie.2", "person.1") in rows


def test_polars_predicates(small_graph):
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.title.contains("lph")))) == ["movie.0"]
    movie = Node("movie")
    rows = names(list(small_graph.select(movie).where(movie.title.is_in(["Alpha", "Gamma"]))))
    assert set(rows) == {"movie.0", "movie.2"}


def test_id_equality_seeds(small_graph):
    user = Node("user")
    assert names(list(small_graph.select(user).where(user.id == 0))) == ["user.0"]


# --- direction: one relation, two ways --------------------------------------

def test_inverse_walks_the_relation_backwards(small_graph):
    person, movie = Node("person"), Node("movie")
    rows = names(list(small_graph.select(movie).where(person.directed_by.inverse == movie,
                                                      person.id == 0)))
    assert set(rows) == {"movie.0", "movie.1"}


def test_forward_and_inverse_are_not_interchangeable(small_graph):
    # person -directed_by-> movie does not exist; the edge runs the other way
    person, movie = Node("person"), Node("movie")
    assert list(small_graph.select(movie).where(person.directed_by == movie)) == []


def test_wildcard_edge_traverses_both_directions(small_graph):
    # the two-hop bridge movie -> genre -> movie only closes if the second step
    # may run against the stored direction; this is what rv=True used to fake
    left, right, path = Node("movie"), Node("movie"), Path()
    rows = names(list(small_graph.select(right).where(path == [left, Edge(), Node("genre"), Edge(), right],
                                                      left.id == 0)))
    assert set(rows) == {"movie.0", "movie.1"}


def test_relation_names_carry_no_direction_suffix(small_graph):
    assert set(small_graph.relations) == {"directed_by", "has_genre", "has_interact"}


# --- paths & OR -------------------------------------------------------------

def test_path_materializes(small_graph):
    movie, path = Node("movie"), Path()
    rows = list(small_graph.select(path).where(path == [movie, Edge(), Node()]))
    assert all(len(r) == 3 for r in rows)
    assert any(names([r]) == [("movie.0", "directed_by", "person.0")] for r in rows)


def test_path_relation_is_a_directed_rel(small_graph):
    movie, path = Node("movie"), Path()
    rows = list(small_graph.select(path).where(path == [movie, Edge("has_genre"), Node("genre")]))
    relation = rows[0][1]
    assert isinstance(relation, Rel)
    assert relation.name == "has_genre" and relation.reverse is False


def test_reverse_traversal_marks_the_rel(small_graph):
    genre, path = Node("genre"), Path()
    rows = list(small_graph.select(path).where(path == [genre, Edge("has_genre").inverse, Node("movie")]))
    assert rows and all(r[1] == Rel("has_genre", reverse=True) for r in rows)


def test_or_unions_branches(small_graph):
    movie, path = Node("movie"), Path()
    a = path == [movie, Edge("has_genre"), Node()]
    b = path == [movie, Edge("directed_by"), Node()]
    assert set(names(list(small_graph.select(movie).where(a | b)))) == {
        "movie.0", "movie.1", "movie.2"}


# --- anti-join & hidden nodes ----------------------------------------------

def test_anti_join_selected(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(user, movie).where(
        user.id == 0, ~Has(user, "has_interact", movie))))
    assert set(rows) == {("user.0", "movie.2")}


def test_anti_join_hidden_node(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(movie).where(
        user.id == 0, ~Has(user, "has_interact", movie))))
    assert set(rows) == {"movie.2"}


def test_hidden_node_dedupes(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = list(small_graph.select(movie).where(~Has(user, "has_interact", movie)))
    assert len(rows) == len(set(rows))


# --- degree (one Expr, three roles) ----------------------------------------

def test_degree_projection(small_graph):
    movie = Node("movie")
    rows = dict(small_graph.select(movie, movie.has_interact.inverse.count()))
    assert {str(k): v for k, v in rows.items()} == {"movie.0": 2, "movie.1": 2, "movie.2": 2}


def test_degree_predicate(small_graph):
    person = Node("person")
    assert names(list(small_graph.select(person).where(
        person.directed_by.inverse.count() >= 2))) == ["person.0"]
    person = Node("person")
    assert names(list(small_graph.select(person).where(
        person.directed_by.inverse.count() < 2))) == ["person.1"]


def test_degree_as_rank(small_graph):
    person = Node("person")
    assert names(list(small_graph.select(person)
                      .rank(person.directed_by.inverse.count()).top(1))) == ["person.0"]


def test_unknown_relation_has_zero_degree(small_graph):
    movie = Node("movie")
    assert list(small_graph.select(movie).where(movie.no_such_relation.count() >= 1)) == []


# --- ranking, score, grouping ----------------------------------------------

def test_top_limits(small_graph):
    movie = Node("movie")
    assert len(list(small_graph.select(movie).rank(Ascending(movie.title)).top(2))) == 2


def test_score_projection(small_graph):
    movie = Node("movie")
    scores = [s for _, s in small_graph.select(movie, Score())
              .rank(Ascending(movie.title)).top(3)]
    assert scores == sorted(scores, reverse=True)


def test_combined_score_stays_in_unit_range(small_graph):
    # signals are averaged, not summed, so stacking two does not push the score
    # past 1 and leave the caller dividing by the number of strategies
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(Ascending(movie.title), PageRank(to={"person.0"})).top(3))
    assert all(0.0 <= s <= 1.0 for _, s in rows)


def test_groupby(small_graph):
    person, movie = Node("person"), Node("movie")
    result = list(
        small_graph.select(person, movie)
        .where(movie.directed_by == person)
        .groupby(person).rank(Ascending(movie.title)).top(1)
        .rank(Ascending(person.name)).top(10)
    )
    by_person = {str(p): str(m) for p, m in result}
    assert by_person["person.0"] in ("movie.0", "movie.1")
    assert by_person["person.1"] == "movie.2"


# --- Like: fuzzy membership, soft-where -------------------------------------

def test_like_fuzzy_admission(small_graph):
    # "Xavier" matches "Xavier Director"; "Nobody" matches nothing
    person = Node("person")
    rows = names(list(small_graph.select(person).where(
        Like(person.name.is_in(["Xavier", "Nobody"])))))
    assert set(rows) == {"person.0"}


def test_like_soft_where_widens_and_weights(small_graph):
    # crisp title.is_in(["Alpha"]) alone -> only Alpha; softened -> admits near
    # matches and attaches a graded score, with the exact match ranked first
    movie = Node("movie")
    rows = list(small_graph.select(movie.title, Score()).where(Like(movie.title.is_in(["Alph"]))))
    assert "Alpha" in [t for t, _ in rows]
    assert rows[0][0] == "Alpha"                     # exact/substring match ranks top


def test_like_admits_the_best_match_not_a_region(small_graph):
    """A set of strings is a search box: the value meant, not everything nearby.
    Containment counts as a perfect match, and a typo still lands."""
    for needles, expected in ((["Xavier"], ["person.0"]),            # a fragment
                              (["Xavier Directr"], ["person.0"]),    # a typo
                              (["Xavier Director"], ["person.0"]),   # exact
                              (["Nobody At All"], []),               # below the cutoff
                              ([""], [])):                           # nothing asked
        person = Node("person")
        rows = names(list(small_graph.select(person)
                          .where(Like(person.name.is_in(needles)))))
        assert rows == expected, needles


def test_like_admits_k_matches_per_needle(small_graph):
    """k is how many candidates a needle is allowed to mean; both people here
    are Directors, so one needle reaches both only when asked to.

    At k=1 it is "Yara Director" rather than "Xavier Director": both contain the
    needle, and the one that adds least around it is the one that was meant."""
    for k, expected in ((1, ["person.1"]), (2, ["person.0", "person.1"])):
        person = Node("person")
        rows = names(list(small_graph.select(person)
                          .where(Like(person.name.is_in(["Director"]), k=k))))
        assert sorted(rows) == expected, k


def test_a_blank_needle_does_not_poison_the_others(small_graph):
    person = Node("person")
    rows = names(list(small_graph.select(person)
                      .where(Like(person.name.is_in(["", "Xavier"])))))
    assert rows == ["person.0"]


def test_like_on_a_numeric_set_is_not_string_matched(small_graph):
    movie = Node("movie")
    rows = names(list(small_graph.select(movie).where(Like(movie.year.is_in([1994])))))
    assert sorted(rows) == ["movie.0", "movie.1"]


def test_admission_and_weight_use_one_measure(small_graph):
    """Like's contract is that the crisp support is the region where membership
    exceeds eps; the two now share `closeness` rather than each having its own."""
    like = Like(Node("person").label.is_in(["Xavier"]))
    assert like.closeness("xavier", "xavier director") == 1.0
    assert like.closeness("", "xavier director") == 0.0
    assert 0.5 < like.closeness("xavier directr", "xavier director") < 1.0


def test_like_is_condition_and_strategy():
    from jerboas import Condition, Strategy
    like = Like(Node("movie").year < 3)
    assert isinstance(like, Condition) and isinstance(like, Strategy)


# --- strategies -------------------------------------------------------------

def test_connectivity_runs(small_graph):
    movie = Node("movie")
    result = list(small_graph.select(movie, Score()).rank(Connectivity(to={"person.0"})).top(3))
    assert len(result) == 3


def test_connectivity_favours_the_seed_neighbourhood(small_graph):
    movie = Node("movie")
    ranked = names(list(small_graph.select(movie).rank(Connectivity(to={"person.0"})).top(3)))
    assert set(ranked[:2]) == {"movie.0", "movie.1"}


def test_diffused_mf_to_seed(small_graph):
    movie = Node("movie")
    result = list(small_graph.select(movie, Score())
                  .rank(DiffusedMatrixFactorization(to={"person.0"})).top(3))
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_matrix_factorization_runs(small_graph):
    user, movie = Node("user"), Node("movie")
    result = list(
        small_graph.select(user, movie, Score())
        .where(Has(user, "has_interact", movie))
        .rank(MatrixFactorization()).top(3)
    )
    assert len(result) == 3


def test_pagerank_global_is_a_distribution(small_graph):
    pr = PageRank()
    pr.fit(small_graph)
    assert abs(pr._ranks.sum() - 1.0) < 1e-6          # stationary distribution sums to 1
    assert (pr._ranks >= 0).all()


def test_pagerank_personalized_favors_seed_neighborhood(small_graph):
    # restart at person.0 (directed movie.0 and movie.1): those two should
    # outrank movie.2, which person.0 never directed
    movie = Node("movie")
    ranked = names(list(small_graph.select(movie).rank(PageRank(to={"person.0"})).top(3)))
    assert set(ranked[:2]) == {"movie.0", "movie.1"}
    assert ranked[2] == "movie.2"


def test_pagerank_ranks_and_limits(small_graph):
    movie = Node("movie")
    result = list(small_graph.select(movie, Score()).rank(PageRank(to={"person.0"})).top(2))
    scores = [s for _, s in result]
    assert len(result) == 2 and scores == sorted(scores, reverse=True)


# --- a constraint must reach the selected pattern ---------------------------

def test_disconnected_variable_is_refused(small_graph):
    """The failure this exists to turn from a wrong answer into an error: two
    identical-looking Nodes are two variables, so the constraint below lands on
    one nobody selected and every movie comes back."""
    with pytest.raises(ValueError, match=r'Node\("movie"\).*never joined'):
        list(small_graph.select(Node("movie")).where(Node("movie").title == "Alpha"))


def test_the_same_object_is_the_fix(small_graph):
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.title == "Alpha"))) == ["movie.0"]


def test_an_edge_is_the_other_fix(small_graph):
    # a second variable is fine as soon as something joins it to the pattern
    movie, person = Node("movie"), Node("person")
    rows = names(list(small_graph.select(movie).where(
        movie.directed_by == person, person.name == "Yara Director")))
    assert rows == ["movie.2"]


def test_anti_join_endpoint_counts_as_joined(small_graph):
    # `user` is neither projected nor positively related, but the anti-join
    # constrains the pair, so it is connected and the query stands
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(movie).where(
        user.id == 0, ~Has(user, "has_interact", movie))))
    assert rows == ["movie.2"]


def test_unrelated_projected_nodes_are_still_a_cross_product(small_graph):
    # both are selected, so the caller asked for this and it is not refused
    movie, genre = Node("movie"), Node("genre")
    rows = list(small_graph.select(movie, genre))
    assert len(rows) == 3 * 2


# --- engines ----------------------------------------------------------------

def test_greedy_beam_uses_the_guide(small_graph):
    from jerboas import Greedy
    user, movie = Node("user"), Node("movie")
    rows = list(small_graph.select(user, movie)
                .where(Has(user, "has_interact", movie))
                .rank(MatrixFactorization())
                .using(Greedy(k=2)))
    assert rows and all(str(u).startswith("user.") and str(m).startswith("movie.")
                        for u, m in rows)


def test_greedy_falls_back_without_a_guide(small_graph):
    from jerboas import Greedy
    movie = Node("movie")
    rows = names(list(small_graph.select(movie).using(Greedy())))
    assert set(rows) == {"movie.0", "movie.1", "movie.2"}


# --- the graph itself -------------------------------------------------------

def test_type_blocks_are_contiguous(small_graph):
    low, high = small_graph.block("movie")
    assert high - low == 3
    assert {small_graph.type_of(i) for i in range(low, high)} == {"movie"}


def test_lookup_accepts_source_keys_and_pairs(small_graph):
    assert small_graph.lookup("movie.0") == small_graph.lookup(("movie", 0))
    assert small_graph.lookup("movie.998") is None


# --- an id is a position ----------------------------------------------------

def _numbered(tmp_path, name, ids):
    from jerboas import Graph
    path = tmp_path / f"{name}.knows"
    path.write_text("source\ttarget\n"
                    + "".join(f"person.{a}\tperson.{b}\n" for a, b in ids))
    return Graph(edges=[str(path)])


def test_an_id_is_the_index(tmp_path):
    """`person.2` is the third node of its block, whatever the file's order --
    so resolving a name is arithmetic, with nothing to build or keep in step."""
    graph = _numbered(tmp_path, "dense", [("2", "0"), ("0", "1")])
    assert [graph.lookup(f"person.{i}") for i in range(3)] == [0, 1, 2]
    assert [str(graph.key(i)) for i in range(3)] == ["person.0", "person.1", "person.2"]


def test_the_layout_does_not_depend_on_the_file_order(tmp_path):
    """The same graph written two ways loads to the same integers. That is what
    makes an id something a checkpoint or a URL can refer to."""
    forwards = _numbered(tmp_path, "fw", [("0", "1"), ("1", "2")])
    backwards = _numbered(tmp_path, "bw", [("1", "2"), ("0", "1")])
    assert ([str(forwards.key(i)) for i in range(forwards.n_nodes)]
            == [str(backwards.key(i)) for i in range(backwards.n_nodes)])


def test_lookup_answers_none_for_a_node_that_is_not_there(tmp_path):
    """Asking about a node is a question -- unlike loading a malformed file,
    which is a bug in whatever wrote it."""
    graph = _numbered(tmp_path, "edges", [("0", "1"), ("1", "2")])
    for missing in ("person.3", "person.-1", "person.two", "person.", "ghost.0"):
        assert graph.lookup(missing) is None, missing


def test_a_gap_in_the_ids_is_refused(tmp_path):
    """A skipped id leaves a node with no edges and no attributes, which cannot
    be told apart from one whose data went missing."""
    with pytest.raises(ValueError, match=r"person\.1 is never mentioned"):
        _numbered(tmp_path, "gap", [("0", "2")])


def test_an_id_named_twice_is_refused(tmp_path):
    """'person.7' and 'person.007' are one place and two nodes; whichever loaded
    first would simply be gone."""
    with pytest.raises(ValueError, match=r"person\.1 is named twice"):
        _numbered(tmp_path, "twice", [("0", "1"), ("01", "0")])


def test_a_text_id_is_refused(tmp_path):
    """The format is not negotiated per dataset: an id that is not a position
    fails at load rather than costing every query a translation table."""
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        _numbered(tmp_path, "text", [("ada", "grace")])


# --- renumber: doing at load what a builder would have done ------------------

def _loose(tmp_path, name, ids):
    from jerboas import Graph
    path = tmp_path / f"{name}.knows"
    path.write_text("source\ttarget\n"
                    + "".join(f"person.{a}\tperson.{b}\n" for a, b in ids))
    return Graph(edges=[str(path)], renumber=True)


def test_renumber_accepts_what_the_format_would_refuse(tmp_path):
    """The escape hatch is explicit and produces a conforming graph: positions
    from zero, and the ids the source used kept as `label`."""
    graph = _loose(tmp_path, "text", [("grace", "ada"), ("ada", "alan")])
    assert [str(graph.key(i)) for i in range(graph.n_nodes)] \
        == ["person.0", "person.1", "person.2"]
    assert [graph.key(i).label for i in range(3)] == ["ada", "alan", "grace"]


def test_renumber_sorts_numbers_as_numbers(tmp_path):
    """`movie.2` before `movie.10`: a source that numbered its nodes keeps the
    order it meant, which lexicographic sorting would scramble."""
    graph = _loose(tmp_path, "sparse", [("10", "2"), ("2", "300")])
    assert [graph.key(i).label for i in range(3)] == [2, 10, 300]


def test_renumber_does_not_depend_on_the_file_order(tmp_path):
    """Sorted rather than first-seen, so the same node set numbers the same way
    however the file lists it -- and however much of it is read at a time."""
    one = _loose(tmp_path, "one", [("b", "a"), ("a", "c")])
    other = _loose(tmp_path, "other", [("c", "a"), ("a", "b")])
    assert ([one.key(i).label for i in range(one.n_nodes)]
            == [other.key(i).label for i in range(other.n_nodes)])


def test_without_renumber_the_label_is_the_position(tmp_path):
    """Nothing is invented for a conforming graph: `label` and `id` are the
    same array under two names."""
    graph = _numbered(tmp_path, "plain", [("0", "1"), ("1", "2")])
    assert [graph.key(i).label for i in range(3)] == [0, 1, 2]
    assert graph.column("person", "label") is graph.column("person", "id")


# --- edge weights -----------------------------------------------------------
#
# The fixture's interactions carry a rating: user.0 rated movie.0 a 5 and
# movie.1 a 1, user.1 rated movie.1 a 4 and movie.2 a 2, user.2 rated movie.0 a
# 3 and movie.2 a 5. Everything in the kg is unscored, and therefore weighs 1.

def test_an_unscored_edge_weighs_one(small_graph):
    movie = small_graph.lookup("movie.0")
    person = small_graph.lookup("person.0")
    assert small_graph.weight_of(movie, person) == 1.0


def test_the_score_column_is_loaded_as_the_weight(small_graph):
    user = small_graph.lookup("user.0")
    assert small_graph.weight_of(user, small_graph.lookup("movie.0")) == 5.0
    assert small_graph.weight_of(user, small_graph.lookup("movie.1")) == 1.0


def test_score_predicate_narrows_the_traversal(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    watched = Edge("has_interact")
    rows = names(list(small_graph.select(rec).where(
        path == [user, watched, rec], user.id == 0, watched.score >= 3)))
    assert rows == ["movie.0"]                    # the 1-star edge is not walked


def test_the_score_kwarg_says_the_same_thing(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    rows = names(list(small_graph.select(rec).where(
        path == [user, Edge("has_interact", score=(3, None)), rec], user.id == 0)))
    assert rows == ["movie.0"]


def test_a_band_intersects_both_bounds(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    watched = Edge("has_interact")
    rows = names(list(small_graph.select(rec).where(
        path == [user, watched, rec], user.id == 1,
        watched.score >= 2, watched.score <= 4)))
    assert set(rows) == {"movie.1", "movie.2"}    # rated 4 and 2


def test_a_negated_score_predicate_keeps_the_rest(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    watched = Edge("has_interact")
    rows = names(list(small_graph.select(rec).where(
        path == [user, watched, rec], user.id == 0, ~(watched.score >= 3))))
    assert rows == ["movie.1"]


def test_score_survives_inverse(small_graph):
    rec, user, path = Node("movie"), Node("user"), Path()
    watched = Edge("has_interact", score=(4, None))
    rows = names(list(small_graph.select(rec).where(
        path == [rec, watched.inverse, user], user.id == 0)))
    assert rows == ["movie.0"]


def test_an_unconstrained_edge_walks_everything(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    rows = names(list(small_graph.select(rec).where(
        path == [user, Edge("has_interact"), rec], user.id == 0)))
    assert set(rows) == {"movie.0", "movie.1"}


def test_has_takes_the_score_kwarg(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(user, movie).where(
        Has(user, "has_interact", movie, score=(4, None)))))
    assert set(rows) == {("user.0", "movie.0"), ("user.1", "movie.1"), ("user.2", "movie.2")}


def test_anti_join_asks_for_the_absence_of_that_edge(small_graph):
    # ~Has(..., score>=4) is "no highly rated edge", not "an edge that is not
    # highly rated": user.0 rated movie.1 a 1, so the pair still qualifies
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(movie).where(
        user.id == 0, ~Has(user, "has_interact", movie, score=(4, None)))))
    assert set(rows) == {"movie.1", "movie.2"}


def test_a_constrained_edge_that_is_never_walked_is_refused(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    with pytest.raises(ValueError, match="never appears in a pattern"):
        list(small_graph.select(rec).where(
            path == [user, Edge("has_interact"), rec], Edge("has_interact").score >= 3))


def test_norm_rescales_inside_one_relation(small_graph):
    user = small_graph.lookup("user.0")
    # ratings run 1..5 across the relation, so a 5 is 1.0 and a 1 is 0.0
    assert small_graph.weight_of(user, small_graph.lookup("movie.0"), normalized=True) == 1.0
    assert small_graph.weight_of(user, small_graph.lookup("movie.1"), normalized=True) == 0.0
    # a relation whose weights are all equal reads as 1.0, not as 0.0
    movie = small_graph.lookup("movie.0")
    assert small_graph.weight_of(movie, small_graph.lookup("genre.0"), normalized=True) == 1.0


def test_norm_predicate_reads_the_rescaled_weight(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    watched = Edge("has_interact")
    rows = names(list(small_graph.select(rec).where(
        path == [user, watched, rec], user.id == 2, watched.score.norm() > 0.5)))
    assert rows == ["movie.2"]                    # rated 5 -> 1.0, against movie.0's 3 -> 0.5


def test_weight_ranks_by_the_stored_score(small_graph):
    user, rec, path = Node("user"), Node("movie"), Path()
    ranked = names(list(small_graph.select(rec).where(
        path == [rec, Edge("has_interact").inverse, user], user.id == 0).rank(Weight())))
    assert ranked == ["movie.0", "movie.1"]       # rated 5 before rated 1


def test_weight_without_a_path_falls_back_to_incident_weight(small_graph):
    rec = Node("movie")
    ranked = names(list(small_graph.select(rec).rank(Weight("has_interact"))))
    # normalized: movie.0 is 1.0 + 0.5, movie.2 is 1.0 + 0.25, movie.1 is 0.75 + 0.0
    assert ranked == ["movie.0", "movie.2", "movie.1"]


def test_weight_aggregates_a_walk(small_graph):
    rec, path = Node("movie"), Path()
    ranked = list(small_graph.select(rec, Score()).where(
        path == [rec, Edge(), Node("user"), Edge(), Node("movie")]).rank(Weight(how="min")))
    assert ranked and [s for _, s in ranked] == sorted([s for _, s in ranked], reverse=True)


def test_a_graph_without_scores_weighs_one_everywhere(tmp_path):
    from jerboas import Graph
    path = tmp_path / "plain.knows"
    path.write_text("source\ttarget\nperson.0\tperson.1\n")
    graph = Graph(edges=[str(path)])
    assert list(graph.out_weights) == [1.0]
    assert list(graph.weights(normalized=True)[0]) == [1.0]


def test_weighted_pagerank_follows_the_ratings(small_graph):
    movie = Node("movie")
    plain = names(list(small_graph.select(movie).rank(PageRank(to={"user.0"}))))
    weighted = names(list(small_graph.select(movie).rank(PageRank(to={"user.0"}, weighted=True))))
    # user.0 rated movie.0 a 5 and movie.1 a 1, which only the weighted walk knows
    assert weighted[0] == "movie.0"
    assert plain != weighted or plain[0] == "movie.0"


def test_weighted_matrix_factorization_reads_the_rating(small_graph):
    user, movie = Node("user"), Node("movie")
    ranked = names(list(small_graph.select(movie).where(
        user.id == 0, Has(user, "has_interact", movie))
        .rank(MatrixFactorization(weighted=True))))
    assert ranked[0] == "movie.0"                 # rated 5, against movie.1's 1


# --- the loader -------------------------------------------------------------
#
# Edge files are read a chunk at a time and split by column rather than line by
# line, so the two things that can go wrong are a chunk cutting a line in half
# and a file that is not a rectangle.

def test_chunk_boundaries_are_invisible(tmp_path, monkeypatch):
    from jerboas import Graph
    from jerboas import graph as graph_module

    rows = [(f"user.{i}", f"movie.{i % 7}", str(1 + i % 5)) for i in range(200)]
    path = tmp_path / "chunky.has_interact"
    path.write_text("source\ttarget\tscore\n"
                    + "\n".join("\t".join(row) for row in rows) + "\n")

    whole = Graph(edges=[str(path)])
    monkeypatch.setattr(graph_module, "CHUNK", 7)      # narrower than one line
    split = Graph(edges=[str(path)])

    assert split.n_nodes == whole.n_nodes
    assert split.relations == whole.relations
    assert np.array_equal(split.out_indices, whole.out_indices)
    assert np.array_equal(split.out_rels, whole.out_rels)
    assert np.array_equal(split.out_weights, whole.out_weights)
    # same nodes, and the same id for each: a chunk is not allowed to be visible
    assert [str(split.key(i)) for i in range(split.n_nodes)] == \
           [str(whole.key(i)) for i in range(whole.n_nodes)]


def test_a_ragged_edge_file_falls_back(tmp_path):
    from jerboas import Graph

    path = tmp_path / "mixed.rated"
    path.write_text("source\ttarget\tscore\n"
                    "user.0\tmovie.0\t5\n"
                    "user.1\tmovie.1\n"                 # no score: weighs 1
                    "user.3\n"                          # no target: not an edge
                    "\n"
                    "user.2\tmovie.2\t2\n")
    graph = Graph(edges=[str(path)])

    pairs = {(str(graph.key(s)), str(graph.key(t)))
             for s, t in zip(np.repeat(np.arange(graph.n_nodes), np.diff(graph.out_indptr)),
                             graph.out_indices)}
    assert pairs == {("user.0", "movie.0"), ("user.1", "movie.1"), ("user.2", "movie.2")}
    assert sorted(graph.out_weights.tolist()) == [1.0, 2.0, 5.0]


def test_the_kg_may_carry_a_score(tmp_path):
    from jerboas import Graph

    path = tmp_path / "weighted.kg"
    path.write_text("movie.0\tsimilar_to\tmovie.1\t0.9\n"
                    "movie.0\thas_genre\tgenre.0\n")     # unscored: weighs 1
    graph = Graph(kg=str(path))
    assert graph.weight_of(graph.lookup("movie.0"), graph.lookup("movie.1")) == 0.9
    assert graph.weight_of(graph.lookup("movie.0"), graph.lookup("genre.0")) == 1.0


# --- aggregates over what the pattern matched --------------------------------
#
# `Degree` counts a node's edges in the graph; an aggregate counts what *this*
# query matched. The graph below is the shape that needs it: two seed films,
# tags of varying strength, and candidates that share either one strong tag or
# several weak ones.

def _tagged(tmp_path, rows):
    from jerboas import Graph
    path = tmp_path / "t.has_tag"
    path.write_text("source\ttarget\tscore\n"
                    + "\n".join("\t".join(row) for row in rows) + "\n")
    return Graph(edges=[str(path)])


def _pattern(graph, seeds):
    seed, tag, rec, path = Node("movie"), Node("tag"), Node("movie"), Path()
    shared = Edge("has_tag")
    carried = shared.inverse            # named once: `.inverse` makes a new marker
    where = (path == [seed, shared, tag, carried, rec],
             seed.is_in(seeds), ~rec.is_in(seeds))
    return seed, tag, rec, path, carried, where


SHARED_TAGS = [
    ("movie.0", "tag.0", "0.9"), ("movie.0", "tag.1", "0.8"),
    ("movie.1", "tag.1", "0.7"), ("movie.1", "tag.2", "0.6"),
    # three weak tags in common
    ("movie.2", "tag.0", "0.5"), ("movie.2", "tag.1", "0.4"), ("movie.2", "tag.2", "0.3"),
    # one very strong one
    ("movie.3", "tag.0", "1.0"),
]


def test_sum_ranks_by_the_whole_pattern(tmp_path):
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, _tag, rec, _path, carried, where = _pattern(graph, {"movie.0", "movie.1"})
    ranked = names(list(graph.select(rec).where(*where).rank(Sum(carried.score))))
    assert ranked == ["movie.2", "movie.3"]         # 0.5+0.4+0.3 beats 1.0


def test_max_ranks_by_the_strongest_single_match(tmp_path):
    """The same query, the other question -- and the answers differ, which is
    why the aggregate has to be said out loud."""
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, _tag, rec, _path, carried, where = _pattern(graph, {"movie.0", "movie.1"})
    ranked = names(list(graph.select(rec).where(*where).rank(Max(carried.score))))
    assert ranked == ["movie.3", "movie.2"]


def test_count_counts_distinct_bindings(tmp_path):
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, tag, rec, _path, _carried, where = _pattern(graph, {"movie.0", "movie.1"})
    ranked = names(list(graph.select(rec).where(*where).rank(Count(tag))))
    assert ranked == ["movie.2", "movie.3"]         # three tags against one


def test_an_aggregate_collapses_the_rows(tmp_path):
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, _tag, rec, _path, carried, where = _pattern(graph, {"movie.0", "movie.1"})
    rows = list(graph.select(rec).where(*where).rank(Sum(carried.score)))
    assert len(rows) == len(set(rows)) == 2         # not one row per shared tag


def test_a_selected_path_still_explains_one_row_per_result(tmp_path):
    """Selecting the walk must not split the group: a Path is evidence of a
    match, not part of what was matched."""
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, _tag, rec, path, carried, where = _pattern(graph, {"movie.0", "movie.1"})
    rows = list(graph.select(rec, Score(), path).where(*where).rank(Sum(carried.score)))
    assert [str(film) for film, _score, _walk in rows] == ["movie.2", "movie.3"]
    assert all(len(walk) == 5 for _film, _score, walk in rows)


def test_one_match_reached_twice_is_still_one_match(tmp_path):
    """movie.7 carries a single tag that both seeds point at, so it is found
    twice; movie.8 carries two tags worth less together than that one. Counting
    the rows rather than the matches would reverse them."""
    graph = _tagged(tmp_path, [
        ("movie.0", "tag.0", "0.9"), ("movie.1", "tag.0", "0.9"),
        ("movie.3", "tag.0", "1.0"),
        ("movie.0", "tag.1", "0.9"), ("movie.2", "tag.1", "0.6"),
        ("movie.1", "tag.2", "0.9"), ("movie.2", "tag.2", "0.6"),
    ])
    _seed, _tag, rec, _path, carried, where = _pattern(graph, {"movie.0", "movie.1"})
    ranked = names(list(graph.select(rec).where(*where).rank(Sum(carried.score))))
    assert ranked == ["movie.2", "movie.3"]         # 1.2 against 1.0, not against 2.0


def test_only_count_takes_a_node(tmp_path):
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, _tag, rec, _path, _carried, where = _pattern(graph, {"movie.0", "movie.1"})
    with pytest.raises(TypeError, match="Count is the one aggregate"):
        list(graph.select(rec).where(*where).rank(Sum(rec)))


def test_an_aggregate_over_an_unmatched_edge_is_refused(tmp_path):
    graph = _tagged(tmp_path, SHARED_TAGS)
    _seed, _tag, rec, _path, _carried, where = _pattern(graph, {"movie.0", "movie.1"})
    with pytest.raises(ValueError, match="never appears in a pattern"):
        list(graph.select(rec).where(*where).rank(Sum(Edge("has_tag").score)))


# --- how the query meets the language ---------------------------------------

def test_and_or_not_are_refused_rather_than_silently_halved(small_graph):
    """`(a) and (b)` evaluates bool(a), finds it true and returns b -- half the
    constraint, no error. numpy, pandas and polars all refuse; so does this."""
    movie = Node("movie")
    with pytest.raises(TypeError, match="no truth value"):
        (movie.year >= 1990) and (movie.year <= 1995)
    with pytest.raises(TypeError, match="no truth value"):
        not (movie.year >= 1990)
    with pytest.raises(TypeError, match="no truth value"):
        bool(movie.title)


def test_a_chained_comparison_is_refused(small_graph):
    """Python expands it into an `and`, which would keep only the second half."""
    movie = Node("movie")
    with pytest.raises(TypeError, match="chained comparisons"):
        1990 <= movie.year <= 1995


def test_where_refuses_what_is_not_a_condition(small_graph):
    with pytest.raises(TypeError, match="takes Conditions"):
        small_graph.select(Node("movie")).where(True)


def test_a_key_refuses_to_be_pickled(small_graph):
    """It points into a Graph, and the default would serialise the graph with
    it -- megabytes for one node, gigabytes on a real dataset."""
    import pickle
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(small_graph["movie.0"])


def test_the_graph_is_a_container(small_graph):
    assert len(small_graph) == small_graph.n_nodes
    assert "movie.0" in small_graph and "movie.99" not in small_graph
    assert small_graph["movie.0"].attrs["title"] == "Alpha"
    with pytest.raises(KeyError):
        small_graph["movie.99"]
    assert {key.type for key in small_graph} == {"movie", "person", "genre", "user"}


def test_a_query_is_a_sequence(small_graph):
    from collections.abc import Sequence
    query = small_graph.select(Node("movie"))
    assert isinstance(query, Sequence)
    assert small_graph["movie.1"] in query
    assert query.index(small_graph["movie.2"]) == 2
    assert [str(k) for k in reversed(query)] == ["movie.2", "movie.1", "movie.0"]


def test_slicing_is_top(small_graph):
    """The two spellings look alike, so they must not cost differently: `q[:2]`
    limits the search rather than ranking everything and throwing it away."""
    query = small_graph.select(Node("movie"))
    assert len(query[:2]) == 2
    assert len(query) == 2, "the slice did not narrow the query itself"


def test_columns_names_the_projections(small_graph):
    movie = Node("movie")
    query = small_graph.select(movie, movie.title, Score())
    assert query.columns == ["movie", "title", "score"]


def test_a_query_hands_itself_to_numpy(small_graph):
    years = np.asarray(small_graph.select(Node("movie").year))
    assert sorted(years.tolist()) == [1994, 1994, 1999]


# --- direction is said out loud ---------------------------------------------

def test_ascending_and_descending_are_opposites(small_graph):
    movie = Node("movie")
    up = names(list(small_graph.select(movie).rank(Ascending(movie.title))))
    movie = Node("movie")
    down = names(list(small_graph.select(movie).rank(Descending(movie.title))))
    assert up == ["movie.0", "movie.1", "movie.2"]      # Alpha, Beta, Gamma
    assert down == list(reversed(up))


def test_a_bare_expression_ranks_the_way_a_degree_does(small_graph):
    """`rank(expr)` means more-is-better, which is what a degree already
    assumed -- so Descending is the explicit spelling of the default."""
    movie = Node("movie")
    bare = names(list(small_graph.select(movie).rank(movie.year)))
    movie = Node("movie")
    explicit = names(list(small_graph.select(movie).rank(Descending(movie.year))))
    assert bare == explicit == ["movie.2", "movie.0", "movie.1"]


def test_ascending_reverses_a_degree_too(small_graph):
    person = Node("person")
    fewest = names(list(small_graph.select(person)
                        .rank(Ascending(person.directed_by.inverse.count()))))
    assert fewest == ["person.1", "person.0"]           # one film, then two


def test_an_exact_match_beats_a_longer_container(small_graph, tmp_path):
    """"Alien" is contained in Alien, Aliens, Alien 3 and Alien: Resurrection.
    Treating every containment as equally perfect let the tie fall to whichever
    loaded first -- a coin toss wearing the shape of a search result."""
    from jerboas import Graph
    path = tmp_path / "a.movie"
    path.write_text("id\ttitle\n0\tAliens\n1\tAlien: Resurrection\n2\tAlien\n3\tAlien 3\n")
    graph = Graph(attrs=[str(path)], edges=[])
    movie = Node("movie")
    rows = names(list(graph.select(movie).where(Like(movie.title.is_in(["Alien"])))))
    assert rows == ["movie.2"]
