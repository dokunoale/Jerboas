"""What the library promises, exercised through the real loader and the real
frames. The fixture graph is in conftest.py.

The shape of a test here is the shape of the API: build a frame, hop, filter,
and read a column. Nothing is mocked, and the polars frame underneath is the one
a caller gets.
"""

import numpy as np
import polars as pl
import pytest

from jerboas import (Connectivity, DiffusedMatrixFactorization, Graph, Key,
                     MatrixFactorization, PageRank, Weight, col, concat, v)


def names(frame, column=None):
    """One column as source keys, for asserting on identity rather than ids."""
    return [str(key) for key in frame.keys(column)]


# --- a frame is a dataframe, and knows its graph -----------------------------

def test_a_graph_prints_as_an_object_and_a_frame_as_a_table(small_graph):
    assert repr(small_graph).startswith("<Graph:")
    assert "movie" in repr(small_graph)
    assert "shape: (3, 1)" in repr(small_graph.nodes("movie"))


def test_nodes_is_one_column_of_one_type(small_graph):
    frame = small_graph.nodes("movie")
    assert frame.columns == ["movie"]
    assert len(frame) == 3
    assert names(frame) == ["movie.0", "movie.1", "movie.2"]


def test_a_variable_is_the_column_name(small_graph):
    assert small_graph.nodes(rec="movie").columns == ["rec"]


def test_two_variables_at_once_are_refused(small_graph):
    """A second column would be a cross product, and a cross product is never
    what was meant: the second column comes from a hop or a join."""
    with pytest.raises(TypeError, match="one column at a time"):
        small_graph.nodes(a="movie", b="person")


def test_nodes_with_no_type_carries_every_node(small_graph):
    frame = small_graph.nodes()
    assert len(frame) == small_graph.n_nodes
    assert set(frame.pl["type"].to_list()) == set(small_graph.types)


def test_nodes_from_keys(small_graph):
    frame = small_graph.nodes(seed=["movie.0", "movie.2"])
    assert names(frame) == ["movie.0", "movie.2"]


def test_an_unknown_type_says_so(small_graph):
    with pytest.raises(ValueError, match="no type 'film'"):
        small_graph.nodes("film")


def test_attrs_reads_stored_columns(small_graph):
    frame = small_graph.nodes("movie").attrs(movie=["title", "year"])
    assert frame.pl["movie.title"].to_list() == ["Alpha", "Beta", "Gamma"]
    assert frame.pl["movie.year"].to_list() == [1994, 1994, 1999]


def test_attrs_expands_a_whole_type(small_graph):
    frame = small_graph.nodes("movie").attrs("movie")
    assert {"movie.title", "movie.year", "movie.id"} <= set(frame.columns)


def test_a_numeric_attribute_stays_numeric(small_graph):
    """Typed at load, so the comparison is one array operation and `year == 1994`
    and `year == '1994'` cannot disagree."""
    frame = small_graph.nodes("movie").attrs(movie="year")
    assert frame.pl.schema["movie.year"] == pl.Int64
    assert len(frame.filter(v.movie.year >= 1999)) == 1


def test_labels_read_the_graphs_readable_map(small_graph):
    frame = small_graph.nodes("movie").labels("movie")
    assert frame.pl["movie.label"].to_list() == ["Alpha", "Beta", "Gamma"]


def test_labels_resolve_per_type_on_an_untyped_column(small_graph):
    """One column holding movies and people reads each type's own name column --
    which is what a query over a mixed seed set needs."""
    frame = small_graph.nodes(seed=["movie.0", "person.0"]).labels("seed")
    assert frame.pl["seed.label"].to_list() == ["Alpha", "Xavier Director"]


def test_a_type_the_map_does_not_mention_falls_back_to_its_identity(small_graph):
    """Never a guess at which attribute happens to be a name."""
    frame = small_graph.nodes(seed=["user.0"]).labels("seed")
    assert frame.pl["seed.label"].to_list() == [0]


def test_a_missing_attribute_says_what_the_type_has(small_graph):
    with pytest.raises(ValueError, match="movie has no attribute 'plot'"):
        small_graph.nodes("movie").attrs(movie="plot")


def test_keys_carry_their_meaning(small_graph):
    key = small_graph.nodes("movie").keys()[0]
    assert isinstance(key, Key)
    assert (key.type, key.id, key.label) == ("movie", 0, 0)
    assert key.attrs["title"] == "Alpha"
    assert str(key) == "movie.0"


def test_a_key_indexes_an_array_directly(small_graph):
    table = np.arange(small_graph.n_nodes)
    assert table[small_graph.nodes("movie").keys()[2]] == small_graph.lookup("movie.2")


# --- hop: one traversal ------------------------------------------------------

def test_hop_follows_a_named_relation_forwards(small_graph):
    frame = small_graph.nodes("movie").hop(person="directed_by")
    assert set(zip(names(frame, "movie"), names(frame, "person"))) == {
        ("movie.0", "person.0"), ("movie.1", "person.0"), ("movie.2", "person.1")}


def test_hop_reverse_walks_the_same_relation_backwards(small_graph):
    """There is no `directed_by_r`: one relation, and the direction belongs to
    the traversal."""
    frame = small_graph.nodes("person").hop(movie="~directed_by")
    assert sorted(names(frame, "movie")) == ["movie.0", "movie.1", "movie.2"]


def test_forward_and_backward_are_not_interchangeable(small_graph):
    assert len(small_graph.nodes("person").hop(movie="directed_by")) == 0


def test_the_wildcard_walks_both_directions(small_graph):
    """Which is what closes a bridge pattern without the store holding every
    edge twice."""
    frame = small_graph.nodes(seed=["movie.0"]).hop(other=())
    assert set(names(frame, "other")) == {"person.0", "genre.0", "user.0", "user.2"}


def test_a_hop_names_the_relation_it_walked(small_graph):
    frame = small_graph.nodes(seed=["movie.0"]).hop(other=())
    walked = dict(zip(names(frame, "other"),
                      frame.with_columns(r=v.other.via).pl["r"].to_list()))
    assert walked["person.0"] == "directed_by"
    assert walked["user.0"] == "~has_interact"          # walked against the store


def test_a_named_hop_knows_it_walked_one_relation(small_graph):
    """Every row would carry the same name, so nothing is stored -- but asking
    still answers, because provenance is an attribute and not a column."""
    frame = small_graph.nodes("movie").hop(person="directed_by")
    assert frame.columns == ["movie", "person"]
    assert frame.hidden == []


def test_hop_filters_the_target_by_type(small_graph):
    frame = small_graph.nodes(seed=["movie.0"]).hop(user=()).filter(v.user.type == "user")
    assert sorted(names(frame, "user")) == ["user.0", "user.2"]


def test_a_hop_leaves_from_the_rightmost_column_of_nodes(small_graph):
    two = (small_graph.nodes("movie").hop(genre="has_genre")
           .hop(sibling="~has_genre"))
    assert two.columns == ["movie", "genre", "sibling"]
    # to leave from another, select it and join the result back -- which is
    # what working on a dataframe is for
    from_movie = (small_graph.nodes("movie").select("movie")
                  .hop(person="directed_by"))
    assert set(names(from_movie, "person")) == {"person.0", "person.1"}


def test_a_node_with_no_such_edge_drops_out(small_graph):
    assert len(small_graph.nodes("genre").hop(person="directed_by")) == 0


def test_an_unknown_relation_matches_nothing(small_graph):
    assert len(small_graph.nodes("movie").hop(person="produced_by")) == 0


def test_a_hop_needs_at_least_one_named_step(small_graph):
    with pytest.raises(ValueError, match="last step of a hop must be named"):
        small_graph.nodes("movie").hop()


def test_a_filter_admits_before_the_rows_are_built(small_graph):
    """The same rows a filter afterwards would leave, without building the ones
    it would have thrown away -- which is now the only spelling there is."""
    ahead = (small_graph.nodes("person").hop(rec="~directed_by")
             .filter(v.rec.year >= 1999))
    eager = small_graph.nodes("person").hop(rec="~directed_by")
    eager.pl
    assert names(ahead, "rec") == names(eager.filter(v.rec.year >= 1999), "rec")


def test_membership_takes_any_way_of_naming_nodes(small_graph):
    keys = [small_graph["movie.0"], small_graph["movie.1"]]
    frame = (small_graph.nodes("person").hop(rec="~directed_by")
             .filter(v.rec.is_in(keys)))
    assert sorted(names(frame, "rec")) == ["movie.0", "movie.1"]


def test_type_is_a_condition_like_any_other(small_graph):
    seen = small_graph.nodes(seed=["movie.0"])
    frame = (small_graph.nodes("user").hop(rec=())
             .filter(v.rec.type == "movie", v.rec.is_in(seen)))
    assert set(names(frame, "rec")) == {"movie.0"}


def test_top_over_is_the_best_per_group(small_graph):
    """One user's best film, not the best film overall -- a window function
    rather than one query per group."""
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .top(1, by=v.rec.score, over="user"))
    assert len(frame) == 3                                    # one row per user
    assert frame.with_columns(s=v.rec.score).pl["s"].to_list() == [1.0, 0.75, 1.0]


def test_top_over_is_a_beam_when_it_sits_between_two_hops(small_graph):
    """Keep the k most promising partial walks and expand only those. That was
    an engine once."""
    frame = (small_graph.nodes(user="user").hop(mid="has_interact")
             .top(1, by=v.mid.score, over="user")             # the beam
             .hop(genre="has_genre"))
    assert len(frame) == 3
    assert set(names(frame, "mid")) == {"movie.0", "movie.1", "movie.2"}


def test_a_hop_cannot_land_on_a_column_already_there(small_graph):
    """Suffixing it to `genre_2` would leave a filter written against the
    obvious name silently reading the other step."""
    walked = small_graph.nodes("movie").hop(genre="has_genre")
    with pytest.raises(ValueError, match="its own name"):
        walked.hop(genre="~has_genre")


def test_a_hop_cannot_overwrite_the_column_it_left_from(small_graph):
    walked = small_graph.nodes("movie").hop(genre="has_genre")
    with pytest.raises(ValueError, match="its own name"):
        walked.hop(movie="~has_genre")


# --- a name is resolved late, by the frame that has the graph ----------------

def test_an_attribute_is_read_on_demand(small_graph):
    """No `.attrs()` first: the predicate reaches the frame unresolved, and the
    frame knows `year` is a column of a movie."""
    assert names(small_graph.nodes("movie").filter(v.movie.year >= 1999)) == ["movie.2"]


def test_a_filter_filters_rows_and_not_columns(small_graph):
    """What it read to decide is not left behind: the frame's shape does not
    change under a filter, and `.attrs()` is how a column stays."""
    frame = small_graph.nodes("movie").filter(v.movie.year >= 1994)
    assert frame.columns == ["movie"]
    kept = small_graph.nodes("movie").attrs(movie="year").filter(v.movie.year >= 1994)
    assert kept.columns == ["movie", "movie.year"]


def test_a_relation_answers_how_many(small_graph):
    """`movie.has_genre.count()` -- the arity the old spelling had, resolved
    because the frame can tell a relation from a column."""
    assert names(small_graph.nodes("movie").filter(v.movie.has_genre.count() >= 1)) \
        == ["movie.0", "movie.1", "movie.2"]
    assert len(small_graph.nodes("movie").filter(v.movie.has_genre.count() >= 2)) == 0


def test_a_relation_answers_whether_any(small_graph):
    """And existence, without walking the edge into rows."""
    who = small_graph.nodes(seed=["person.0"])
    assert names(small_graph.nodes("movie").filter(v.movie.directed_by.is_in(who))) \
        == ["movie.0", "movie.1"]
    assert names(small_graph.nodes("movie").filter(~v.movie.directed_by.is_in(who))) \
        == ["movie.2"]


def test_a_relation_has_no_value_of_its_own(small_graph):
    with pytest.raises(TypeError, match="ask how many with .count"):
        small_graph.nodes("movie").filter(v.movie.has_genre >= 2)


def test_names_resolve_column_then_attribute_then_relation(tmp_path):
    """The order is the whole rule, and a tie is refused rather than guessed."""
    edges = tmp_path / "a.knows"
    edges.write_text("source\ttarget\nperson.0\tperson.1\nperson.1\tperson.0\n")
    attrs = tmp_path / "a.person"
    attrs.write_text("id\tknows\n0\t7\n1\t3\n")           # an attribute named like the relation
    graph = Graph(edges=[str(edges)], attrs=[str(attrs)])

    with pytest.raises(ValueError, match="both an attribute .* and a relation"):
        graph.nodes("person").filter(v.person.knows >= 1)
    assert len(graph.nodes("person").filter(v.person.attr.knows >= 5)) == 1
    assert len(graph.nodes("person").filter(v.person.rel.knows.count() >= 1)) == 2


def test_a_name_that_is_neither_says_so(small_graph):
    with pytest.raises(ValueError, match="no attribute 'plot'.*no relation"):
        small_graph.nodes("movie").filter(v.movie.plot == "x")


def test_expressions_combine_and_refuse_to_be_bool(small_graph):
    frame = small_graph.nodes("movie").filter(
        (v.movie.year == 1994) & (v.movie.has_genre.count() >= 1))
    assert names(frame) == ["movie.0", "movie.1"]
    with pytest.raises(TypeError, match="no truth value"):
        bool(v.movie.year >= 1990)


def test_raw_polars_still_passes_through(small_graph):
    """`.expr` is the way out: the column by that exact name, unresolved."""
    frame = small_graph.nodes("movie").attrs(movie="year")
    assert len(frame.filter(pl.col("movie.year") >= 1999)) == 1
    assert len(frame.filter(v.movie.year.expr >= 1999)) == 1


def test_signals_still_combine_by_arithmetic(small_graph):
    frame = (small_graph.nodes("movie")
             .with_columns(pr=PageRank().on("movie").norm())
             .with_columns(score=0.5 * v.pr + 0.5 * v.pr))
    assert frame.pl["score"].to_list() == frame.pl["pr"].to_list()


# --- a filter after a hop lands on the rows that were not built --------------

def test_a_pushed_filter_gives_what_an_eager_one_would(small_graph):
    """The whole point: same answer, applied before the rows exist."""
    def hop():
        return small_graph.nodes("user").hop(rec="has_interact").filter(v.rec.type == "movie")

    eager = hop()
    eager.pl                                        # force the rows into being
    assert (names(hop().filter(v.rec.year >= 1999), "rec")
            == names(eager.filter(v.rec.year >= 1999), "rec"))


def test_a_predicate_about_the_old_frame_is_not_pushed(small_graph):
    """It cannot be: the rows it speaks about are the ones being built."""
    frame = (small_graph.nodes(user="user").attrs(user="id")
             .hop(rec="has_interact").filter(v.rec.type == "movie")
             .filter(v.rec.year >= 1994, v.user.id == 0))
    assert sorted(names(frame, "rec")) == ["movie.0", "movie.1"]


def test_a_mixed_predicate_keeps_both_halves(small_graph):
    """One pushable, one not, in the same call."""
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .filter(v.rec.type == "movie")
             .filter((v.rec.year >= 1994) & (v.rec.score >= 0.75)))
    assert len(frame) == 3


def test_the_steps_confidence_is_the_columns(small_graph):
    """The weight of the edge that revealed a node is that node column's
    confidence -- not a column of its own with a name to remember."""
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .filter(v.rec.score >= 0.75))
    assert sorted(frame.with_columns(s=v.rec.score).pl["s"].to_list()) == [0.75, 1.0, 1.0]


# --- several steps in one hop ------------------------------------------------

def test_a_hop_takes_several_steps(small_graph):
    """One step out of a film reaches its people, genres and viewers; two reach
    the other films they connect to."""
    frame = small_graph.nodes(seed=["movie.0"]).hop((), rec=())
    reached = names(frame.filter(v.rec.type == "movie"), "rec")
    assert "movie.1" in reached                      # same director, same genre
    assert frame.columns == ["seed", "rec"]          # the middle was not named


def test_the_last_step_must_be_named(small_graph):
    """Where the walk ends is what the frame holds, so it needs a column."""
    with pytest.raises(ValueError, match="last step of a hop must be named"):
        small_graph.nodes("movie").hop("has_genre")


def test_an_unnamed_step_folds_the_routes_through_it(tmp_path):
    """Nothing names the intermediate, so nothing tells two routes through it
    apart -- and carrying both is what makes a bridge query explode."""
    path = tmp_path / "d.knows"
    path.write_text("source\ttarget\n"
                    "person.0\tperson.1\nperson.0\tperson.2\n"       # two routes...
                    "person.1\tperson.3\nperson.2\tperson.3\n")      # ...to the same node
    graph = Graph(edges=[str(path)])
    seed = graph.nodes(seed=["person.0"])
    folded = seed.hop("knows", rec="knows")
    kept = seed.hop(mid="knows", rec="knows")
    assert len(kept) == 2 and len(folded) == 1
    assert names(folded, "rec") == ["person.3"]


def test_naming_a_step_keeps_it(small_graph):
    frame = small_graph.nodes(seed=["movie.0"]).hop(via_1=(), rec=())
    assert frame.columns == ["seed", "via_1", "rec"]
    assert frame.vars["via_1"] is None


def test_a_step_may_name_several_relations(small_graph):
    """A collection is "any of these"; `~` reads one backwards."""
    both = small_graph.nodes(seed=["movie.0"]).hop(other=("directed_by", "has_genre"))
    assert sorted(names(both, "other")) == ["genre.0", "person.0"]
    back = small_graph.nodes("person").hop(movie="~directed_by")
    assert sorted(names(back, "movie")) == ["movie.0", "movie.1", "movie.2"]


def test_the_empty_collection_is_the_wildcard(small_graph):
    """No constraint on the relation is the empty set of constraints."""
    frame = small_graph.nodes(seed=["movie.0"]).hop(other=())
    assert set(names(frame, "other")) == {"person.0", "genre.0", "user.0", "user.2"}


def test_lengths_are_unioned_by_concat(small_graph):
    """Two lengths are two frames. The column one branch lacks comes back null,
    which is exactly what "reached the other way" means."""
    seeds = small_graph.nodes(seed=["movie.0"])
    direct = seeds.hop(rec=()).with_columns(hops=pl.lit(1))
    bridge = seeds.hop((), rec=()).with_columns(hops=pl.lit(2))
    both = concat(direct, bridge).filter(v.rec.type == "movie")
    assert set(both.pl["hops"].to_list()) == {2}     # a film reaches no film in one


# --- degree and existence: what a relation says without walking it -----------

def test_a_relation_counts_the_graphs_own_edges(small_graph):
    frame = (small_graph.nodes("person")
             .with_columns(n=v.person.directed_by.count()))
    counts = dict(zip(names(frame), frame.pl["n"].to_list()))
    assert counts == {"person.0": 2, "person.1": 1}


def test_a_relation_count_is_a_fact_about_the_graph(small_graph):
    """Which is what tells it apart from group_by(...).agg(count): filtering the
    frame first does not change it."""
    whole = small_graph.nodes("person").with_columns(n=v.person.directed_by.count())
    part = (small_graph.nodes("person")
            .filter(v.person == small_graph.lookup("person.0"))
            .with_columns(n=v.person.directed_by.count()))
    assert part.pl["n"].to_list() == [whole.pl["n"].to_list()[0]]


def test_a_relation_the_graph_never_saw_is_refused(small_graph):
    with pytest.raises(ValueError, match="no relation by that name"):
        small_graph.nodes("movie").with_columns(n=v.movie.produced_by.count())


def test_a_relation_says_whether_any_edge_exists(small_graph):
    frame = small_graph.nodes("movie").filter(v.movie.has_interact.is_in(
        small_graph.nodes("user")))
    assert sorted(names(frame)) == ["movie.0", "movie.1", "movie.2"]


def test_existence_narrows_without_walking_the_frame(small_graph):
    who = small_graph.nodes(seed=["person.0"])
    frame = small_graph.nodes("movie").filter(v.movie.directed_by.is_in(who))
    assert names(frame) == ["movie.0", "movie.1"]
    assert frame.columns == ["movie"]              # nothing was expanded


def test_the_negation_is_the_absence_of_that_edge(small_graph):
    who = small_graph.nodes(seed=["person.0"])
    assert names(small_graph.nodes("movie").filter(~v.movie.directed_by.is_in(who))) \
        == ["movie.2"]


def test_existence_agrees_with_the_join_it_replaces(small_graph):
    """"the films this user has not seen", the short way and the long way."""
    who = small_graph.nodes(seed=["user.0"])
    watched = who.hop(movie="has_interact").select("movie")
    by_expr = small_graph.nodes("movie").filter(~v.movie.has_interact.is_in(who))
    by_join = small_graph.nodes("movie").join(watched, on="movie", how="anti")
    assert names(by_expr) == names(by_join) == ["movie.2"]


# --- filtering: a predicate is an expression ---------------------------------

def test_filter_on_an_attribute(small_graph):
    frame = small_graph.nodes("movie").attrs(movie="year").filter(v.movie.year >= 1999)
    assert names(frame) == ["movie.2"]


def test_col_and_v_are_the_same_column(small_graph):
    frame = small_graph.nodes("movie").attrs(movie="title")
    assert (len(frame.filter(col("movie.title") == "Alpha"))
            == len(frame.filter(v.movie.title == "Alpha")) == 1)


def test_two_spellings_of_a_variable_are_one_variable(small_graph):
    """The whole identity system: a variable is a name, so saying it twice says
    the same thing -- where two `Node("movie")` used to be two variables."""
    assert str(v.rec) == str(col("rec")) == "rec"
    assert hash(v.rec) == hash(col("rec"))


def test_a_method_name_is_still_reachable_as_a_column(small_graph):
    """`v.person.name` is the person's name, not the string "person". A method
    defined on the class would have shadowed every column called `name` --
    silently, which is the worst way to be wrong."""
    assert str(v.person.name) == "person.name"
    assert str(v.movie.count) == "movie.count"
    frame = small_graph.nodes("person").filter(v.person.name.contains("Xavier"))
    assert names(frame) == ["person.0"]


def test_calling_a_name_is_how_a_method_is_reached(small_graph):
    """The two readings never compete: one is the column, the other is the
    question asked of its parent."""
    counted = (small_graph.nodes(user="user").hop(rec="has_interact")
               .group_by(v.user).agg(n=v.rec.count()))
    assert sorted(counted.pl["n"].to_list()) == [2, 2, 2]
    with pytest.raises(AttributeError, match="no method 'nonesuch'"):
        v.movie.nonesuch()


def test_a_keyword_filter_compares_against_a_value(small_graph):
    """polars reads a bare string as a column name wherever a verb takes one, so
    a value that happens to be a string must not be read that way."""
    frame = small_graph.nodes("movie").attrs(movie="title")
    assert names(frame.filter(**{"movie.title": "Alpha"})) == ["movie.0"]


def test_boolean_operators_combine(small_graph):
    frame = small_graph.nodes("movie").attrs(movie=["title", "year"])
    both = frame.filter((v.movie.year == 1994) & (v.movie.title == "Alpha"))
    either = frame.filter((v.movie.year == 1999) | (v.movie.title == "Alpha"))
    assert names(both) == ["movie.0"]
    assert names(either) == ["movie.0", "movie.2"]


def test_is_in_takes_keys_and_frames(small_graph):
    seeds = small_graph.nodes(seed=["movie.0", "movie.1"])
    by_frame = small_graph.nodes("movie").filter(v.movie.is_in(seeds))
    by_keys = small_graph.nodes("movie").filter(v.movie.is_in(seeds.keys()))
    assert names(by_frame) == names(by_keys) == ["movie.0", "movie.1"]


def test_exclusion_is_a_negated_membership(small_graph):
    seen = small_graph.nodes(seed=["movie.0"])
    assert names(small_graph.nodes("movie").filter(~v.movie.is_in(seen))) \
        == ["movie.1", "movie.2"]


def test_an_anti_join_drops_the_pairs_that_exist(small_graph):
    """"films this user has not watched" is a join with how="anti", vectorized,
    where it used to be a Python loop over every result row."""
    watched = small_graph.nodes(user="user").filter(v.user == small_graph.lookup("user.0")) \
                         .hop(movie="has_interact")
    unseen = small_graph.nodes("movie").join(watched.select("movie"), on="movie", how="anti")
    assert names(unseen) == ["movie.2"]


def test_a_source_key_resolves_inside_an_expression(small_graph):
    """The predicate reaches the frame unresolved, and the frame has the graph
    -- so a name the graph can look up is a name that works here."""
    assert names(small_graph.nodes("movie").filter(v.movie.is_in(["movie.0"]))) \
        == ["movie.0"]


# --- group_by: an aggregate is written out -----------------------------------

TAGS = [
    ("movie.0", "tag.0", "0.9"), ("movie.0", "tag.1", "0.85"),
    ("movie.1", "tag.1", "0.8"), ("movie.1", "tag.2", "0.75"),
    # three tags in common, none of them the strongest
    ("movie.2", "tag.0", "0.8"), ("movie.2", "tag.1", "0.75"), ("movie.2", "tag.2", "0.7"),
    # one that is
    ("movie.3", "tag.0", "1.0"),
    # and something faint, so the relation's scale is not two values wide: a
    # confidence is min-maxed within its own relation, and what the ends of that
    # range are is a fact about the data
    ("movie.4", "tag.3", "0.1"),
]


@pytest.fixture
def tagged(tmp_path):
    path = tmp_path / "t.has_tag"
    path.write_text("source\ttarget\tscore\n"
                    + "\n".join("\t".join(row) for row in TAGS) + "\n")
    return Graph(edges=[str(path)])


def _shared(graph, seeds):
    """Candidates reached from the seeds through a shared tag."""
    return (graph.nodes(seed=seeds)
            .hop(tag="has_tag")
            .hop(rec="~has_tag")
            .filter(~v.rec.is_in(graph.ids_of(seeds))))


def test_sum_ranks_by_the_whole_pattern(tagged):
    """Not "does it share a tag?" but "how much of the list is it?" -- one tag
    is a coincidence, three is a taste."""
    ranked = _shared(tagged, ["movie.0", "movie.1"]) \
        .group_by(v.rec).agg(score=v.rec.score.sum()).top(2)
    assert names(ranked, "rec") == ["movie.2", "movie.3"]      # four matches beat one


def test_max_ranks_by_the_strongest_single_match(tagged):
    """The other question, answered differently -- which is why the aggregate
    has to be said out loud."""
    ranked = _shared(tagged, ["movie.0", "movie.1"]) \
        .group_by(v.rec).agg(score=v.rec.score.max()).top(2)
    assert names(ranked, "rec") == ["movie.3", "movie.2"]


def test_counting_the_matches(tagged):
    counted = _shared(tagged, ["movie.0", "movie.1"]) \
        .group_by(v.rec).agg(score=pl.len(), tags=v.tag.n_unique()).top(1)
    assert counted.pl["score"].to_list() == [4]               # four (seed, tag) matches
    assert counted.pl["tags"].to_list() == [3]                # over three distinct tags


def test_grouping_collapses_the_rows(tagged):
    frame = _shared(tagged, ["movie.0", "movie.1"])
    assert len(frame) > len(frame.group_by(v.rec).agg(score=v.rec.score.sum()))


def test_the_evidence_survives_the_grouping(tagged):
    """A group can keep one walk to explain itself with, which is what selecting
    a Path used to be for -- except here it does not multiply the rows."""
    explained = (_shared(tagged, ["movie.0", "movie.1"])
                 .group_by(v.rec).agg(score=v.rec.score.sum(),
                                      via=v.tag.first(), seed=v.seed.first()))
    assert set(explained.columns) == {"rec", "score", "via", "seed"}
    assert len(explained) == 2


# --- like: graded membership -------------------------------------------------

def test_like_finds_the_whole_from_a_fragment(small_graph):
    found = small_graph.nodes("person").filter(v.person.label.like("xavier"))
    assert names(found) == ["person.0"]


def test_like_survives_a_typo(small_graph):
    found = small_graph.nodes("person").filter(v.person.label.like("Xavir Diretor"))
    assert names(found) == ["person.0"]


def test_like_admits_k_matches_per_needle(small_graph):
    found = small_graph.nodes("person").filter(v.person.label.like("Director", k=2))
    assert len(found) == 2


def test_like_keeps_the_measure_as_the_columns_confidence(small_graph):
    """Admission and weight are one measure, and the measure stays -- as the
    confidence of the column it judged, not as a column with a name to learn."""
    found = (small_graph.nodes("person").filter(v.person.label.like("xavier"))
             .with_columns(sim=v.person.label.score))
    assert found.pl["sim"].to_list() == [1.0]
    typo = (small_graph.nodes("movie").filter(v.movie.label.like("Alpa"))
            .with_columns(sim=v.movie.label.score))
    assert 0.6 < typo.pl["sim"].to_list()[0] < 1.0            # close, not contained


def test_a_blank_needle_admits_nothing(small_graph):
    """It is contained in everything, which would make a blank search the
    broadest one possible instead of the narrowest."""
    found = small_graph.nodes("movie").filter(v.movie.label.like(["", "Alpha"]))
    assert names(found) == ["movie.0"]


def test_like_reads_the_column_it_is_given(small_graph):
    """`v.person.label` names a column the frame does not have yet, so it is
    read rather than refused -- and taken off again, being only a means."""
    found = small_graph.nodes("person").filter(v.person.label.like("xavier"))
    assert names(found) == ["person.0"]
    assert found.columns == ["person"]


def test_an_exact_match_beats_a_longer_container(tmp_path):
    """"alien" is inside Alien, Aliens and Alien 3; the one that adds least is
    the one that was meant."""
    path = tmp_path / "f.movie"
    path.write_text("id\ttitle\n0\tAliens\n1\tAlien\n2\tAlien 3\n")
    edges = tmp_path / "f.has_genre"
    edges.write_text("source\ttarget\nmovie.0\tgenre.0\nmovie.1\tgenre.0\nmovie.2\tgenre.0\n")
    graph = Graph(edges=[str(edges)], attrs=[str(path)], readable={"movie": "title"})
    found = graph.nodes("movie").filter(v.movie.label.like("alien"))
    assert names(found) == ["movie.1"]


# --- concat: the disjunction a single pattern cannot express ------------------

def test_concat_unions_two_routes(small_graph):
    direct = small_graph.nodes(seed=["movie.0"]).hop(genre="has_genre")
    bridged = (small_graph.nodes(seed=["movie.0"]).hop(user="~has_interact")
               .hop(rec="has_interact"))
    both = concat(direct, bridged)
    assert set(both.columns) >= {"seed", "genre", "user", "rec"}
    assert len(both) == len(direct) + len(bridged)


def test_the_column_a_branch_lacks_is_null(small_graph):
    """Which is exactly what "reached the other way" means."""
    one = small_graph.nodes(seed=["movie.0"]).hop(genre="has_genre")
    other = small_graph.nodes(seed=["movie.0"]).hop(person="directed_by")
    both = concat(one, other)
    assert both.pl["genre"].null_count() == len(other)


# --- edge weights are a column -----------------------------------------------

def test_a_traversed_edge_is_the_new_columns_confidence(small_graph):
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .with_columns(s=v.rec.score))
    scores = dict(zip(zip(names(frame, "user"), names(frame, "rec")), frame.pl["s"]))
    assert scores[("user.0", "movie.0")] == 1.0               # a 5, the highest rating
    assert scores[("user.0", "movie.1")] == 0.0               # a 1, the lowest


def test_an_unscored_edge_leaves_nothing_in_doubt(small_graph):
    """Every row would say 1.0, so nothing is stored -- and asking still says
    1.0, because absent means certain."""
    frame = small_graph.nodes("movie").hop(person="directed_by")
    assert frame.hidden == []
    assert frame.with_columns(s=v.person.score).pl["s"].to_list() == [1.0, 1.0, 1.0]


def test_filtering_by_confidence_is_filtering(small_graph):
    """Where a score band used to be a compile-time mask over stored edges."""
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .filter(v.rec.score >= 0.5))
    assert len(frame) == 4                                    # the 5, 4, 3 and 5


def test_confidence_is_rescaled_within_one_relation(small_graph):
    """A 1-5 rating and a cosine similarity are both floats and mean nothing to
    each other, so the scale is per relation -- and it is the only scale there
    is, since a confidence that is not in [0, 1] is not one."""
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .with_columns(s=v.rec.score))
    assert min(frame.pl["s"].to_list()) == 0.0
    assert max(frame.pl["s"].to_list()) == 1.0


def test_a_graph_without_scores_is_certain_everywhere(tmp_path):
    path = tmp_path / "plain.knows"
    path.write_text("source\ttarget\nperson.0\tperson.1\nperson.1\tperson.2\n")
    graph = Graph(edges=[str(path)])
    frame = graph.nodes("person").hop(other="knows")
    assert frame.hidden == []
    assert frame.with_columns(s=v.other.score).pl["s"].to_list() == [1.0, 1.0]


def test_the_edges_frame_is_the_store(small_graph):
    edges = small_graph.edges("has_interact")
    assert len(edges) == 6
    assert set(edges.pl["relation"].to_list()) == {"has_interact"}
    assert sorted(edges.pl["score"].to_list()) == [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]


def test_the_edges_frame_carries_every_relation(small_graph):
    assert set(small_graph.edges().pl["relation"].to_list()) == set(small_graph.relations)


def test_the_kg_may_carry_a_score(tmp_path):
    path = tmp_path / "weighted.kg"
    path.write_text("movie.0\tsimilar_to\tmovie.1\t0.9\n"
                    "movie.0\thas_genre\tgenre.0\n")     # unscored: weighs 1
    graph = Graph(kg=str(path))
    scores = dict(zip(graph.edges().pl["relation"].to_list(),
                      graph.edges().pl["score"].to_list()))
    assert scores == {"similar_to": 0.9, "has_genre": 1.0}


# --- strategies: the scores a column cannot hold on its own ------------------

def test_pagerank_is_a_distribution(small_graph):
    frame = small_graph.nodes().with_columns(pr=PageRank().on("node"))
    assert pytest.approx(sum(frame.pl["pr"].to_list()), rel=1e-6) == 1.0


def test_personalized_pagerank_favours_the_seed_neighbourhood(small_graph):
    seeds = small_graph.nodes(seed=["genre.0"])
    ranked = (small_graph.nodes("movie")
              .with_columns(score=PageRank(to=seeds).on("movie")).top(3))
    assert names(ranked)[0] in ("movie.0", "movie.1")        # both are in genre.0
    assert names(ranked)[-1] == "movie.2"


def test_connectivity_counts_the_seeds_that_reach(small_graph):
    seeds = ["person.0"]
    frame = small_graph.nodes("movie").with_columns(c=Connectivity(to=seeds).on("movie"))
    counts = dict(zip(names(frame), frame.pl["c"].to_list()))
    assert counts["movie.0"] == counts["movie.1"] == 1.0
    assert counts["movie.2"] == 0.0


def test_weight_ranks_by_the_weight_the_data_put_on_a_node(small_graph):
    frame = (small_graph.nodes("movie")
             .with_columns(w=Weight("has_interact", normalized=False).on("movie")))
    weights = dict(zip(names(frame), frame.pl["w"].to_list()))
    assert weights["movie.0"] == 8.0                          # rated 5 and 3
    assert weights["movie.1"] == 5.0                          # rated 1 and 4


def test_matrix_factorization_needs_to_be_told_whose_taste(small_graph):
    """It used to fall back to the first user the search walked through, which
    is an arbitrary person's ranking wearing the shape of an answer."""
    with pytest.raises(ValueError, match="whose taste to apply"):
        small_graph.nodes("movie").with_columns(
            mf=MatrixFactorization(factors=2, iterations=2).on("movie"))


def test_matrix_factorization_reads_the_user_column(small_graph):
    frame = (small_graph.nodes(user="user").hop(rec="has_interact")
             .with_columns(mf=MatrixFactorization(factors=2, iterations=3,
                                                  item_type="movie").on("rec", "user")))
    assert all(value > 0 for value in frame.pl["mf"].to_list())


def test_matrix_factorization_takes_one_user_for_the_whole_frame(small_graph):
    frame = small_graph.nodes("movie").with_columns(
        mf=MatrixFactorization(factors=2, iterations=3, user="user.0").on("movie"))
    assert len(frame.pl["mf"].to_list()) == 3


def test_weighted_matrix_factorization_reads_the_rating(small_graph):
    """user.0 rated movie.0 a 5 and movie.1 a 1: explicit feedback separates
    them, implicit cannot."""
    def fitted(weighted):
        return small_graph.nodes("movie").with_columns(
            mf=MatrixFactorization(factors=2, iterations=8, weighted=weighted,
                                   user="user.0").on("movie")).pl["mf"].to_list()
    weighted = fitted(True)
    assert weighted[0] > weighted[1]


def test_diffused_mf_scores_against_a_seed_set(small_graph):
    seeds = small_graph.nodes(seed=["genre.0"])
    frame = small_graph.nodes("movie").with_columns(
        d=DiffusedMatrixFactorization(factors=2, iterations=3, to=seeds).on("movie"))
    assert len(frame.pl["d"].to_list()) == 3


def test_diffused_mf_scores_each_row_against_its_own_seed(small_graph):
    frame = (small_graph.nodes(seed=["genre.0"]).hop(rec="~has_genre")
             .with_columns(d=DiffusedMatrixFactorization(factors=2, iterations=3)
                           .on("rec", "seed")))
    assert len(frame) == 2


# --- signals combine by arithmetic, in the open ------------------------------

def test_norm_puts_a_signal_in_the_unit_range(small_graph):
    frame = small_graph.nodes("movie").with_columns(pr=PageRank().on("movie").norm())
    values = frame.pl["pr"].to_list()
    assert min(values) == 0.0 and max(values) == 1.0


def test_a_flat_signal_does_not_veto_the_others(small_graph):
    """A signal that says the same about everything normalizes to 1.0, not 0.0:
    saying nothing is not the same as saying no."""
    frame = small_graph.nodes("movie").with_columns(
        w=Weight("directed_by", normalized=False).on("movie").norm())
    assert frame.pl["w"].to_list() == [1.0, 1.0, 1.0]


def test_a_relation_the_graph_never_saw_weighs_nothing(small_graph):
    """Rather than everything: an unknown name used to fall through to "all
    relations", so a typo silently ranked by the whole graph."""
    frame = small_graph.nodes("movie").with_columns(
        w=Weight("produced_by", normalized=False).on("movie"))
    assert frame.pl["w"].to_list() == [0.0, 0.0, 0.0]


def test_two_signals_combine_by_weights_that_are_written_down(small_graph):
    seeds = small_graph.nodes(seed=["genre.0"])
    frame = (small_graph.nodes("movie")
             .with_columns(pr=PageRank(to=seeds).on("movie").norm(),
                           w=Weight("has_interact").on("movie").norm())
             .with_columns(score=0.7 * v.pr + 0.3 * v.w))
    combined = frame.pl["score"].to_list()
    assert all(0.0 <= value <= 1.0 for value in combined)
    assert combined == [pytest.approx(0.7 * pr + 0.3 * w) for pr, w
                        in zip(frame.pl["pr"].to_list(), frame.pl["w"].to_list())]


def test_top_sorts_by_the_score_column_when_there_is_one(small_graph):
    frame = small_graph.nodes("movie").with_columns(score=PageRank().on("movie"))
    assert frame.top(1).pl["score"].to_list() == [max(frame.pl["score"].to_list())]


def test_on_needs_a_column(small_graph):
    with pytest.raises(TypeError, match="needs the column being scored"):
        PageRank().on()


# --- interop -----------------------------------------------------------------

def test_a_frame_hands_itself_to_polars_and_numpy(small_graph):
    frame = small_graph.nodes("movie").attrs(movie="title")
    assert isinstance(frame.pl, pl.DataFrame)
    assert isinstance(frame.to_polars(), pl.DataFrame)
    assert np.asarray(frame).shape == (3, 2)


def test_a_frame_hands_itself_to_pandas(small_graph):
    pytest.importorskip("pyarrow")                  # polars converts through it
    frame = small_graph.nodes("movie").attrs(movie="title")
    assert list(frame.to_pandas()["movie.title"]) == ["Alpha", "Beta", "Gamma"]


def test_a_frame_is_iterable_and_sized(small_graph):
    frame = small_graph.nodes("movie")
    assert len(frame) == 3
    assert len(list(frame)) == 3
    assert frame.shape == (3, 1)


def test_the_escape_hatch_is_polars_itself(small_graph):
    """Anything the frame does not forward is one attribute away."""
    frame = small_graph.nodes("movie").attrs(movie="year")
    assert frame.pl.group_by("movie.year").len().height == 2


def test_forwarded_verbs_keep_the_graph(small_graph):
    frame = small_graph.nodes("movie").attrs(movie="year").sort("movie.year",
                                                                descending=True)
    assert names(frame) == ["movie.2", "movie.0", "movie.1"]
    assert frame.graph is small_graph


def test_renaming_follows_the_variable(small_graph):
    frame = small_graph.nodes("movie").rename({"movie": "rec"})
    assert frame.vars == {"rec": "movie"}
    assert names(frame, "rec") == ["movie.0", "movie.1", "movie.2"]


def test_selecting_away_a_node_column_forgets_the_variable(small_graph):
    frame = small_graph.nodes("movie").attrs(movie="title").select("movie.title")
    assert frame.vars == {}


# --- the graph itself --------------------------------------------------------

def test_type_blocks_are_contiguous(small_graph):
    low, high = small_graph.block("movie")
    assert high - low == 3
    assert {small_graph.type_of(i) for i in range(low, high)} == {"movie"}


def test_lookup_accepts_source_keys_and_pairs(small_graph):
    assert small_graph.lookup("movie.0") == small_graph.lookup(("movie", 0))
    assert small_graph.lookup("movie.998") is None


def test_the_graph_is_a_container(small_graph):
    assert len(small_graph) == small_graph.n_nodes
    assert "movie.0" in small_graph and "movie.99" not in small_graph
    assert str(small_graph["movie.1"]) == "movie.1"
    with pytest.raises(KeyError):
        small_graph["movie.99"]


def test_a_key_refuses_to_be_pickled(small_graph):
    import pickle
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(small_graph["movie.0"])


# --- an id is a position -----------------------------------------------------

def _numbered(tmp_path, name, ids):
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
    graph = _numbered(tmp_path, "edges", [("0", "1"), ("1", "2")])
    for missing in ("person.3", "person.-1", "person.two", "person.", "ghost.0"):
        assert graph.lookup(missing) is None, missing


def test_a_gap_in_the_ids_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"person\.1 is never mentioned"):
        _numbered(tmp_path, "gap", [("0", "2")])


def test_an_id_named_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"person\.1 is named twice"):
        _numbered(tmp_path, "twice", [("0", "1"), ("01", "0")])


def test_a_text_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        _numbered(tmp_path, "text", [("ada", "grace")])


# --- renumber: doing at load what a builder would have done ------------------

def _loose(tmp_path, name, ids):
    path = tmp_path / f"{name}.knows"
    path.write_text("source\ttarget\n"
                    + "".join(f"person.{a}\tperson.{b}\n" for a, b in ids))
    return Graph(edges=[str(path)], renumber=True)


def test_renumber_accepts_what_the_format_would_refuse(tmp_path):
    graph = _loose(tmp_path, "text", [("grace", "ada"), ("ada", "alan")])
    assert [str(graph.key(i)) for i in range(graph.n_nodes)] \
        == ["person.0", "person.1", "person.2"]
    assert [graph.key(i).label for i in range(3)] == ["ada", "alan", "grace"]


def test_renumber_sorts_numbers_as_numbers(tmp_path):
    graph = _loose(tmp_path, "sparse", [("10", "2"), ("2", "300")])
    assert [graph.key(i).label for i in range(3)] == [2, 10, 300]


def test_renumber_does_not_depend_on_the_file_order(tmp_path):
    one = _loose(tmp_path, "one", [("b", "a"), ("a", "c")])
    other = _loose(tmp_path, "other", [("c", "a"), ("a", "b")])
    assert ([one.key(i).label for i in range(one.n_nodes)]
            == [other.key(i).label for i in range(other.n_nodes)])


def test_without_renumber_the_label_is_the_position(tmp_path):
    graph = _numbered(tmp_path, "plain", [("0", "1"), ("1", "2")])
    assert [graph.key(i).label for i in range(3)] == [0, 1, 2]
    assert graph.column("person", "label") is graph.column("person", "id")


# --- the loader --------------------------------------------------------------

def test_chunk_boundaries_are_invisible(tmp_path, monkeypatch):
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
    assert [str(split.key(i)) for i in range(split.n_nodes)] == \
           [str(whole.key(i)) for i in range(whole.n_nodes)]


def test_a_ragged_edge_file_falls_back(tmp_path):
    path = tmp_path / "mixed.rated"
    path.write_text("source\ttarget\tscore\n"
                    "user.0\tmovie.0\t5\n"
                    "user.1\tmovie.1\n"                 # no score: weighs 1
                    "user.3\n"                          # no target: not an edge
                    "\n"
                    "user.2\tmovie.2\t2\n")
    graph = Graph(edges=[str(path)])
    edges = graph.edges()
    pairs = {(str(graph.key(s)), str(graph.key(t)))
             for s, t in zip(edges.pl["source"].to_list(), edges.pl["target"].to_list())}
    assert pairs == {("user.0", "movie.0"), ("user.1", "movie.1"), ("user.2", "movie.2")}
    assert sorted(graph.out_weights.tolist()) == [1.0, 2.0, 5.0]


# --- traversal is a gather ---------------------------------------------------

def test_ranges_concatenates_slices():
    from jerboas.traverse import ranges
    starts = np.array([10, 0, 5])
    counts = np.array([3, 0, 2])
    assert ranges(starts, counts).tolist() == [10, 11, 12, 5, 6]


def test_a_hop_expands_every_source_row(small_graph):
    """One result row per edge, and each keeps the row it came from."""
    frame = small_graph.nodes("user").hop(movie="has_interact")
    assert len(frame) == 6
    assert sorted(names(frame, "user")) == ["user.0", "user.0", "user.1",
                                            "user.1", "user.2", "user.2"]
