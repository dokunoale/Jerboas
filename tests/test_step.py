"""A budgeted step: at most n edges per row, the best or a draw, fused into the
store so it costs what it keeps. The fixture graph is in conftest.py."""

import polars as pl
import pytest

import jerboas as jb
from jerboas import step, v


def pairs(frame, *columns):
    return set(zip(*(frame.keys(column) for column in columns)))


def as_str(found):
    return {tuple(str(one) for one in pair) for pair in found}


def test_a_budget_wider_than_every_node_is_the_whole_step(small_graph):
    users = small_graph.nodes(user="user")
    full = users.hop(movie="has_interact")
    assert pairs(users.hop(movie=step("has_interact").top(10)), "user", "movie") \
        == pairs(full, "user", "movie")
    assert pairs(users.hop(movie=step("has_interact")), "user", "movie") \
        == pairs(full, "user", "movie")


def test_top_keeps_the_heaviest_edge_of_each_row(small_graph):
    kept = small_graph.nodes(user="user").hop(movie=step("has_interact").top(1))
    assert as_str(pairs(kept, "user", "movie")) == {
        ("user.0", "movie.0"), ("user.1", "movie.1"), ("user.2", "movie.2")}
    # what the step measured is still the column's confidence
    assert kept.with_columns(s=v.movie.score).pl["s"].to_list() == [1.0, 0.75, 1.0]


def test_top_ranks_by_any_expression_of_the_step(small_graph):
    lightest = step("has_interact").top(1, by=-v.movie.score)
    kept = small_graph.nodes(user="user").hop(movie=lightest)
    assert as_str(pairs(kept, "user", "movie")) == {
        ("user.0", "movie.1"), ("user.1", "movie.2"), ("user.2", "movie.0")}


def test_a_map_may_read_the_frame_the_walk_leaves_from(small_graph):
    # each user's own preference is a column of the frame, so the map is
    # dynamic: evaluated per row, on the candidates, not fused into the store
    users = small_graph.nodes(user=["user.0", "user.1"]).with_columns(
        pref=small_graph.nodes(m=["movie.1", "movie.2"]).pl["m"])
    preferred = step("has_interact").top(1, by=(v.movie == v.pref).cast(pl.Float64))
    assert as_str(pairs(users.hop(movie=preferred), "user", "movie")) == {
        ("user.0", "movie.1"), ("user.1", "movie.2")}


def test_a_map_reading_what_the_frame_does_not_have_is_refused(small_graph):
    users = small_graph.nodes(user="user")
    with pytest.raises(ValueError, match="the frame it leaves from"):
        users.hop(movie=step("has_interact").top(1, by=v.nobody.score))


def test_selectors_chain_into_a_cascade(small_graph):
    # the two heaviest, then the lightest of those
    cascade = step("has_interact").top(2).top(1, by=-v.movie.score)
    kept = small_graph.nodes(user="user").hop(movie=cascade)
    assert as_str(pairs(kept, "user", "movie")) == {
        ("user.0", "movie.1"), ("user.1", "movie.2"), ("user.2", "movie.0")}


def test_probability_is_each_edges_share_of_its_row(small_graph):
    users = small_graph.nodes(user=["user.0"])
    even = users.hop(movie=step("has_interact").probability(by=1))
    assert even.with_columns(p=v.movie.score).pl["p"].to_list() == [0.5, 0.5]
    # by weight: a 5 and a 1 are 1.0 and 0.0 on the relation's own scale
    weighed = users.hop(movie=step("has_interact").probability())
    assert sorted(weighed.with_columns(p=v.movie.score).pl["p"].to_list()) == [0.0, 1.0]
    # over what was kept, as a decoder renormalizes its top k
    kept = users.hop(movie=step("has_interact").top(1).probability(by=1))
    assert kept.with_columns(p=v.movie.score).pl["p"].to_list() == [1.0]


def test_a_budget_over_several_relations_keeps_n_of_all_of_them(small_graph):
    movies = small_graph.nodes(movie="movie")
    both = step("directed_by", "has_genre")
    assert pairs(movies.hop(x=both.top(10)), "movie", "x") == pairs(movies.hop(x=both), "movie", "x")
    assert len(movies.hop(x=both.top(1))) == 3
    # any relation, either way: both directions at once
    assert len(movies.hop(x=step().top(1))) == 3


def test_a_sample_is_a_subset_and_repeats_with_its_seed(small_graph):
    users = small_graph.nodes(user="user")
    full = pairs(users.hop(movie="has_interact"), "user", "movie")
    drawn = step("has_interact").sample(1, by=1, seed=3)
    first = pairs(users.hop(movie=drawn), "user", "movie")
    assert first <= full
    assert len(first) == 3                        # one per user
    assert pairs(users.hop(movie=drawn), "user", "movie") == first


def test_a_sample_by_weight_never_draws_an_edge_weighing_nothing(small_graph):
    # user.0 rated movie.1 a 1, the relation's minimum: its weight is 0
    drawn = step("has_interact").sample(20, seed=0)
    kept = small_graph.nodes(user=["user.0"]).hop(movie=drawn)
    assert as_str(pairs(kept, "user", "movie")) == {("user.0", "movie.0")}


def test_a_budgeted_step_plans_like_any_other(small_graph):
    walk = lambda: (small_graph.nodes(user="user")
                    .hop(movie=step("has_interact").top(1))
                    .hop(fan=step("~has_interact").top(1, by=-(v.fan == v.user).cast(pl.Float64))))
    eager = pairs(walk(), "user", "movie", "fan")
    with jb.optimize():
        planned = pairs(walk(), "user", "movie", "fan")
    assert planned == eager


def test_a_draw_belongs_to_the_node_not_to_its_row(small_graph):
    drawn = step("~has_interact").sample(1, by=1, seed=7)
    movies = small_graph.nodes(movie=["movie.0", "movie.1", "movie.0"])
    fans = movies.hop(fan=drawn).keys("fan")
    assert str(fans[0]) == str(fans[2])
    alone = small_graph.nodes(movie=["movie.0"]).hop(fan=drawn).keys("fan")
    assert str(alone[0]) == str(fans[0])


def test_the_nucleus_keeps_the_fewest_edges_holding_a_share(small_graph):
    users = small_graph.nodes(user=["user.0", "user.2"])
    # user.0's weights are 1.0 and 0.0: the heavier holds all of the mass
    kept = users.hop(movie=step("has_interact").top_p(0.5))
    assert as_str(pairs(kept, "user", "movie")) == {
        ("user.0", "movie.0"), ("user.2", "movie.2")}
    # alike, half the mass is one of two, and all of it is both
    assert len(users.hop(movie=step("has_interact").top_p(0.5, by=1))) == 2
    assert len(users.hop(movie=step("has_interact").top_p(1.0, by=1))) == 4
    # the same, evaluated on the candidates rather than fused
    dynamic = step("has_interact").top_p(0.5, by=(v.user >= 0).cast(pl.Float64) * v.movie.score)
    assert pairs(users.hop(movie=dynamic), "user", "movie") == pairs(kept, "user", "movie")


def _walk_home(frame):
    # user.0 -> its two films, each 0.5 -> each film's two fans, each 0.25
    return (frame.hop(movie=step("has_interact").probability(by=1))
                 .hop(fan=step("~has_interact").probability(by=1)))


def test_a_probability_composes_along_the_walk(small_graph):
    walked = _walk_home(small_graph.nodes(user=["user.0"]))
    p = walked.with_columns(p=v.fan.score).pl["p"].to_list()
    assert p == [0.25] * 4 and sum(p) == 1.0
    ended = walked.group_by(v.fan, confidence="sum").len()
    mass = dict(zip((str(one) for one in ended.keys("fan")),
                    ended.with_columns(p=v.fan.score).pl["p"].to_list()))
    assert mass == {"user.0": 0.5, "user.1": 0.25, "user.2": 0.25}


def test_the_mass_of_a_planned_walk_is_folded_across_slices(small_graph):
    with jb.optimize(rows=1):
        ended = _walk_home(small_graph.nodes(user=["user.0"])) \
            .group_by(v.fan, confidence="sum").len()
    mass = dict(zip((str(one) for one in ended.keys("fan")),
                    ended.with_columns(p=v.fan.score).pl["p"].to_list()))
    assert mass == {"user.0": 0.5, "user.1": 0.25, "user.2": 0.25}


def test_a_probability_step_must_be_named(small_graph):
    with pytest.raises(ValueError, match="must be named"):
        small_graph.nodes(user="user").hop(step("has_interact").probability(),
                                           fan="~has_interact")
