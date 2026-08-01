"""Behavioural coverage for the surface: passive refs, Polars-style predicates,
Like (fuzzy + soft-where), path/OR, ranking, grouping, strategies -- plus the
values a query returns (Key, Rel) and the directed-relation model."""

import pytest

from jerboas import Node, Edge, Path, Score, Like, Has, Key, Rel
from jerboas import Connectivity, DiffusedMatrixFactorization, MatrixFactorization, PageRank
from jerboas.strategies import Alphabetical


def names(rows):
    """Rows of Keys as their printable form, for comparing against literals."""
    if rows and isinstance(rows[0], tuple):
        return [tuple(str(v) for v in row) for row in rows]
    return [str(row) for row in rows]


# --- the values a query returns ---------------------------------------------

def test_select_returns_keys(small_graph):
    rows = sorted(small_graph.select(Node("movie")))
    assert all(isinstance(k, Key) for k in rows)
    assert names(rows) == ["movie.1", "movie.2", "movie.3"]


def test_key_exposes_type_id_label_attrs(small_graph):
    movie = Node("movie")
    key = list(small_graph.select(movie).where(movie.title == "Alpha"))[0]
    assert key.type == "movie"
    assert key.id == 1                       # numeric ids stay numbers
    assert key.label == "Alpha"              # the type's label column
    assert key.attrs["year"] == 1994         # typed at load, not a string


def test_label_prefers_name_over_other_text_columns(small_graph):
    person = Node("person")
    key = list(small_graph.select(person).where(person.id == 1))[0]
    assert key.label == "Xavier Director"


def test_label_is_queryable_not_just_readable(small_graph):
    # the same spelling reaches `title` on a movie and `name` on a person, so a
    # caller filtering by readable name needs to know neither
    movie, person, genre = Node("movie"), Node("person"), Node("genre")
    assert names(list(small_graph.select(movie).where(movie.label == "Alpha"))) == ["movie.1"]
    assert names(list(small_graph.select(person)
                      .where(person.label == "Xavier Director"))) == ["person.1"]
    assert names(list(small_graph.select(genre).where(genre.label == "Drama"))) == ["genre.2"]


def test_label_projects(small_graph):
    assert sorted(small_graph.select(Node("movie").label)) == ["Alpha", "Beta", "Gamma"]
    assert sorted(small_graph.select(Node("genre").label)) == ["Comedy", "Drama"]


def test_label_resolves_per_type_on_an_untyped_node(small_graph):
    anything = Node()
    rows = names(list(small_graph.select(anything).where(anything.label == "Comedy")))
    assert rows == ["genre.1"]


def test_label_matches_what_the_key_reports(small_graph):
    # querying a key's own label finds exactly that key back
    for key in list(small_graph.select(Node("movie"))):
        movie = Node("movie")
        found = list(small_graph.select(movie).where(movie.label == key.label))
        assert [str(k) for k in found] == [str(key)]


def test_label_drives_fuzzy_seed_resolution(small_graph):
    # the shape example.py uses: one query, any type
    for type_, needle, expected in (("movie", "Alph", "movie.1"),
                                    ("person", "Xavier", "person.1")):
        node = Node(type_)
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
    assert sorted(small_graph.select(Node("movie").id)) == [1, 2, 3]


def test_numeric_attribute_compares_as_a_number(small_graph):
    # the loader types the column, so all three spellings agree -- they did not
    # when every stored value was text and only some operators coerced
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.year == 1994))) == ["movie.1", "movie.2"]
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.year == "1994"))) == ["movie.1", "movie.2"]
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.year >= 1999))) == ["movie.3"]


def test_kwargs_sugar_matches_the_operator_form(small_graph):
    assert names(list(small_graph.select(Node("movie", year=1994)))) == ["movie.1", "movie.2"]


# --- relations & predicates -------------------------------------------------

def test_relation_via_eq(small_graph):
    movie, person = Node("movie"), Node("person")
    rows = set(names(list(small_graph.select(movie, person).where(movie.directed_by == person))))
    assert ("movie.1", "person.1") in rows and ("movie.3", "person.2") in rows


def test_polars_predicates(small_graph):
    movie = Node("movie")
    assert names(list(small_graph.select(movie).where(movie.title.contains("lph")))) == ["movie.1"]
    movie = Node("movie")
    rows = names(list(small_graph.select(movie).where(movie.title.is_in(["Alpha", "Gamma"]))))
    assert set(rows) == {"movie.1", "movie.3"}


def test_id_equality_seeds(small_graph):
    user = Node("user")
    assert names(list(small_graph.select(user).where(user.id == 1))) == ["user.1"]


# --- direction: one relation, two ways --------------------------------------

def test_inverse_walks_the_relation_backwards(small_graph):
    person, movie = Node("person"), Node("movie")
    rows = names(list(small_graph.select(movie).where(person.directed_by.inverse == movie,
                                                      person.id == 1)))
    assert set(rows) == {"movie.1", "movie.2"}


def test_forward_and_inverse_are_not_interchangeable(small_graph):
    # person -directed_by-> movie does not exist; the edge runs the other way
    person, movie = Node("person"), Node("movie")
    assert list(small_graph.select(movie).where(person.directed_by == movie)) == []


def test_wildcard_edge_traverses_both_directions(small_graph):
    # the two-hop bridge movie -> genre -> movie only closes if the second step
    # may run against the stored direction; this is what rv=True used to fake
    left, right, path = Node("movie"), Node("movie"), Path()
    rows = names(list(small_graph.select(right).where(path == [left, Edge(), Node("genre"), Edge(), right],
                                                      left.id == 1)))
    assert set(rows) == {"movie.1", "movie.2"}


def test_relation_names_carry_no_direction_suffix(small_graph):
    assert set(small_graph.relations) == {"directed_by", "has_genre", "has_interact"}


# --- paths & OR -------------------------------------------------------------

def test_path_materializes(small_graph):
    movie, path = Node("movie"), Path()
    rows = list(small_graph.select(path).where(path == [movie, Edge(), Node()]))
    assert all(len(r) == 3 for r in rows)
    assert any(names([r]) == [("movie.1", "directed_by", "person.1")] for r in rows)


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
        "movie.1", "movie.2", "movie.3"}


# --- anti-join & hidden nodes ----------------------------------------------

def test_anti_join_selected(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(user, movie).where(
        user.id == 1, ~Has(user, "has_interact", movie))))
    assert set(rows) == {("user.1", "movie.3")}


def test_anti_join_hidden_node(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(movie).where(
        user.id == 1, ~Has(user, "has_interact", movie))))
    assert set(rows) == {"movie.3"}


def test_hidden_node_dedupes(small_graph):
    user, movie = Node("user"), Node("movie")
    rows = list(small_graph.select(movie).where(~Has(user, "has_interact", movie)))
    assert len(rows) == len(set(rows))


# --- degree (one Expr, three roles) ----------------------------------------

def test_degree_projection(small_graph):
    movie = Node("movie")
    rows = dict(small_graph.select(movie, movie.has_interact.inverse.count()))
    assert {str(k): v for k, v in rows.items()} == {"movie.1": 2, "movie.2": 2, "movie.3": 2}


def test_degree_predicate(small_graph):
    person = Node("person")
    assert names(list(small_graph.select(person).where(
        person.directed_by.inverse.count() >= 2))) == ["person.1"]
    person = Node("person")
    assert names(list(small_graph.select(person).where(
        person.directed_by.inverse.count() < 2))) == ["person.2"]


def test_degree_as_rank(small_graph):
    person = Node("person")
    assert names(list(small_graph.select(person)
                      .rank(person.directed_by.inverse.count()).top(1))) == ["person.1"]


def test_unknown_relation_has_zero_degree(small_graph):
    movie = Node("movie")
    assert list(small_graph.select(movie).where(movie.no_such_relation.count() >= 1)) == []


# --- ranking, score, grouping ----------------------------------------------

def test_top_limits(small_graph):
    assert len(list(small_graph.select(Node("movie")).rank(Alphabetical()).top(2))) == 2


def test_score_projection(small_graph):
    movie = Node("movie")
    scores = [s for _, s in small_graph.select(movie, Score()).rank(Alphabetical()).top(3)]
    assert scores == sorted(scores, reverse=True)


def test_combined_score_stays_in_unit_range(small_graph):
    # signals are averaged, not summed, so stacking two does not push the score
    # past 1 and leave the caller dividing by the number of strategies
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(Alphabetical(), PageRank(to={"person.1"})).top(3))
    assert all(0.0 <= s <= 1.0 for _, s in rows)


def test_groupby(small_graph):
    person, movie = Node("person"), Node("movie")
    result = list(
        small_graph.select(person, movie)
        .where(movie.directed_by == person)
        .groupby(person).rank(Alphabetical()).top(1).rank(Alphabetical()).top(10)
    )
    by_person = {str(p): str(m) for p, m in result}
    assert by_person["person.1"] in ("movie.1", "movie.2")
    assert by_person["person.2"] == "movie.3"


# --- Like: fuzzy membership, soft-where -------------------------------------

def test_like_fuzzy_admission(small_graph):
    # "Xavier" matches "Xavier Director"; "Nobody" matches nothing
    person = Node("person")
    rows = names(list(small_graph.select(person).where(
        Like(person.name.is_in(["Xavier", "Nobody"])))))
    assert set(rows) == {"person.1"}


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
    for needles, expected in ((["Xavier"], ["person.1"]),            # a fragment
                              (["Xavier Directr"], ["person.1"]),    # a typo
                              (["Xavier Director"], ["person.1"]),   # exact
                              (["Nobody At All"], []),               # below the cutoff
                              ([""], [])):                           # nothing asked
        person = Node("person")
        rows = names(list(small_graph.select(person)
                          .where(Like(person.label.is_in(needles)))))
        assert rows == expected, needles


def test_like_admits_k_matches_per_needle(small_graph):
    """k is how many candidates a needle is allowed to mean; both people here
    are Directors, so one needle reaches both only when asked to."""
    for k, expected in ((1, ["person.1"]), (2, ["person.1", "person.2"])):
        person = Node("person")
        rows = names(list(small_graph.select(person)
                          .where(Like(person.label.is_in(["Director"]), k=k))))
        assert sorted(rows) == expected, k


def test_a_blank_needle_does_not_poison_the_others(small_graph):
    person = Node("person")
    rows = names(list(small_graph.select(person)
                      .where(Like(person.label.is_in(["", "Xavier"])))))
    assert rows == ["person.1"]


def test_like_on_a_numeric_set_is_not_string_matched(small_graph):
    movie = Node("movie")
    rows = names(list(small_graph.select(movie).where(Like(movie.year.is_in([1994])))))
    assert sorted(rows) == ["movie.1", "movie.2"]


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
    result = list(small_graph.select(movie, Score()).rank(Connectivity(to={"person.1"})).top(3))
    assert len(result) == 3


def test_connectivity_favours_the_seed_neighbourhood(small_graph):
    movie = Node("movie")
    ranked = names(list(small_graph.select(movie).rank(Connectivity(to={"person.1"})).top(3)))
    assert set(ranked[:2]) == {"movie.1", "movie.2"}


def test_diffused_mf_to_seed(small_graph):
    movie = Node("movie")
    result = list(small_graph.select(movie, Score())
                  .rank(DiffusedMatrixFactorization(to={"person.1"})).top(3))
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
    # restart at person.1 (directed movie.1 and movie.2): those two should
    # outrank movie.3, which person.1 never directed
    movie = Node("movie")
    ranked = names(list(small_graph.select(movie).rank(PageRank(to={"person.1"})).top(3)))
    assert set(ranked[:2]) == {"movie.1", "movie.2"}
    assert ranked[2] == "movie.3"


def test_pagerank_ranks_and_limits(small_graph):
    movie = Node("movie")
    result = list(small_graph.select(movie, Score()).rank(PageRank(to={"person.1"})).top(2))
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
    assert names(list(small_graph.select(movie).where(movie.title == "Alpha"))) == ["movie.1"]


def test_an_edge_is_the_other_fix(small_graph):
    # a second variable is fine as soon as something joins it to the pattern
    movie, person = Node("movie"), Node("person")
    rows = names(list(small_graph.select(movie).where(
        movie.directed_by == person, person.name == "Yara Director")))
    assert rows == ["movie.3"]


def test_anti_join_endpoint_counts_as_joined(small_graph):
    # `user` is neither projected nor positively related, but the anti-join
    # constrains the pair, so it is connected and the query stands
    user, movie = Node("user"), Node("movie")
    rows = names(list(small_graph.select(movie).where(
        user.id == 1, ~Has(user, "has_interact", movie))))
    assert rows == ["movie.3"]


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
    assert set(rows) == {"movie.1", "movie.2", "movie.3"}


# --- the graph itself -------------------------------------------------------

def test_type_blocks_are_contiguous(small_graph):
    low, high = small_graph.block("movie")
    assert high - low == 3
    assert {small_graph.type_of(i) for i in range(low, high)} == {"movie"}


def test_lookup_accepts_source_keys_and_pairs(small_graph):
    assert small_graph.lookup("movie.1") == small_graph.lookup(("movie", 1))
    assert small_graph.lookup("movie.999") is None
