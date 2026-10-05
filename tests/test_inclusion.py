"""The chance a drawn edge had of being drawn, carried as a shadow column
(`v.x.inclusion`), and what it turns a count into: an estimate of the exact
walk's count rather than a truncated one, with a standard error.

The fixture graph is in conftest.py: every user watched exactly two films, so
`sample(1)` of a user's films keeps each with chance 1/2 and an estimated
count of the user's films is exactly 2, one kept row standing for both.
"""

import numpy as np
import polars as pl
import pytest

import jerboas as jb
from jerboas import step, v
from jerboas.query import traverse


def test_a_flat_draws_chance_is_the_budget_over_the_degree(small_graph):
    drawn = step("has_interact").sample(1, by=1, seed=0).inclusion()
    kept = small_graph.nodes(user="user").hop(movie=drawn)
    pi = kept.with_columns(pi=v.movie.inclusion).pl["pi"].to_list()
    assert pi == [0.5, 0.5, 0.5]


def test_nothing_cut_means_no_column(small_graph):
    # two edges, a budget of two: kept with certainty, and certainty allocates
    # nothing -- as a confidence of all 1.0 is not a column either
    drawn = step("has_interact").sample(2, by=1, seed=0).inclusion()
    kept = small_graph.nodes(user="user").hop(movie=drawn)
    assert not any("inclusion" in name for name in kept.hidden)
    assert kept.with_columns(pi=v.movie.inclusion).pl["pi"].to_list() == [1.0] * 6


def test_a_column_nothing_sampled_reads_as_certain(small_graph):
    kept = small_graph.nodes(user="user").hop(movie="has_interact")
    assert kept.with_columns(pi=v.movie.inclusion).pl["pi"].to_list() == [1.0] * 6


def test_inclusion_needs_a_draw(small_graph):
    with pytest.raises(ValueError, match="sample"):
        step("has_interact").top(1).inclusion()
    with pytest.raises(ValueError, match="sample"):
        step("has_interact").inclusion()


def test_an_inclusion_step_must_be_named(small_graph):
    with pytest.raises(ValueError, match="must be named"):
        small_graph.nodes(user="user").hop(
            step("has_interact").sample(1, by=1, seed=0).inclusion(),
            fan="~has_interact")


def test_each_column_carries_its_own_draws_chance(small_graph):
    # like a confidence, an inclusion is the column's own measurement: the
    # chance of the row is the product of its columns' chances, which is what
    # an estimate reads -- never one column's alone
    walked = (small_graph.nodes(user="user")
              .hop(movie=step("has_interact").sample(1, by=1, seed=0).inclusion())
              .hop(fan=step("~has_interact").sample(1, by=1, seed=0).inclusion()))
    assert walked.with_columns(pi=v.movie.inclusion).pl["pi"].unique().to_list() == [0.5]
    assert walked.with_columns(pi=v.fan.inclusion).pl["pi"].unique().to_list() == [0.5]


def test_a_count_composes_the_chances_of_every_step(small_graph):
    # one kept route user -> 1 of 2 films -> 1 of its 2 fans stands for the
    # whole two-step walk: 1 / (1/2 * 1/2) = 4 rows, the exact count
    drawn = (step("has_interact").sample(1, by=1, seed=0).inclusion(),
             step("~has_interact").sample(1, by=1, seed=0).inclusion())
    counted = (small_graph.nodes(user=["user.0"])
               .hop(movie=drawn[0]).hop(fan=drawn[1])
               .group_by(v.user, confidence=None).len("routes"))
    assert counted.pl["routes"].to_list() == [4.0]


def test_an_exact_step_after_a_draw_adds_no_chance_of_its_own(small_graph):
    # the fans' edges were all kept: no shadow on them, and the estimate still
    # accounts for the draw through the film's own -- the kept film's two fans
    # each stand for both of the user's films
    walked = (small_graph.nodes(user=["user.0"])
              .hop(movie=step("has_interact").sample(1, by=1, seed=0).inclusion())
              .hop(fan="~has_interact"))
    assert not any("inclusion" in name and name.endswith("fan") for name in walked.hidden)
    counted = walked.group_by(v.fan, confidence=None).len("routes")
    assert counted.pl["routes"].to_list() == [2.0, 2.0]


def test_a_count_over_a_sampled_walk_is_an_estimate(small_graph):
    # one kept row per user stands for both of the user's films: the estimate
    # is the exact count, and its standard error the count's confidence
    drawn = step("has_interact").sample(1, by=1, seed=0).inclusion()
    counted = (small_graph.nodes(user="user").hop(movie=drawn)
               .group_by(v.user, confidence=None).len("seen"))
    assert counted.pl["seen"].to_list() == [2.0, 2.0, 2.0]
    se = counted.with_columns(se=v.seen.score).pl["se"].to_list()
    assert se == [np.sqrt(2.0)] * 3


def test_a_count_over_an_unsampled_walk_is_a_count(small_graph):
    counted = (small_graph.nodes(user="user").hop(movie="has_interact")
               .group_by(v.user).len("seen"))
    assert counted.pl["seen"].to_list() == [2, 2, 2]
    assert counted.with_columns(se=v.seen.score).pl["se"].to_list() == [1.0] * 3


def test_the_estimate_is_unbiased_over_the_seed(small_graph):
    # every film has exactly two watchers; averaging the estimated count over
    # draws lands on it
    exact = {"movie.0": 2.0, "movie.1": 2.0, "movie.2": 2.0}
    totals = {name: 0.0 for name in exact}
    rounds = 2000
    for seed in range(rounds):
        drawn = step("has_interact").sample(1, by=1, seed=seed).inclusion()
        counted = (small_graph.nodes(user="user").hop(movie=drawn)
                   .group_by(v.movie, confidence=None).len("seen"))
        for name, estimate in zip((str(one) for one in counted.keys("movie")),
                                  counted.pl["seen"].to_list()):
            totals[name] += estimate / rounds
    for name, estimate in totals.items():
        assert abs(estimate - exact[name]) < 0.15


def test_a_weighted_draws_chance_follows_the_weights():
    # one draw is in exact proportion: w over the segment's mass
    weights = np.array([1.0, 1.0, 2.0])
    ranked = traverse.Ranked(np.array([3]), weights)
    for seed in range(50):
        _rows, index, pi = ranked.draw([0], 1, seed, [7], inclusion=True)
        assert pi.tolist() == [weights[index[0]] / 4.0]
    # an edge weighing nothing is never drawn, so no chance is ever 0
    weights[0] = 0.0
    ranked = traverse.Ranked(np.array([3]), weights)
    for seed in range(50):
        _rows, index, pi = ranked.draw([0], 2, seed, [7], inclusion=True)
        assert set(index.tolist()) == {1, 2}
        assert (pi > 0).all()


def test_a_cascade_composes_within_a_step():
    # 10 of 50, then 3 of those 10, drawn flat: 3 of 50, exactly
    ranked = traverse.Ranked(np.array([50]))
    for seed in range(20):
        _r, index, pi = ranked.draw([0], 10, seed, [7], inclusion=True)
        assert (pi == 10 / 50).all()
        ranked2 = traverse.Ranked(np.array([10]), np.ones(10))
        _r2, index2, pi2 = ranked2.draw([0], 3, seed, [7], inclusion=True)
        assert (pi2 == 3 / 10).all()


def test_a_draws_chance_composes_within_a_step(small_graph):
    # two draws in one step: the second samples what the first kept, so the
    # chance that rides on the kept edge is the first's, subset and composed
    cascaded = (step("has_interact").sample(1, by=1, seed=0)
                .sample(1, by=1, seed=1).inclusion())
    kept = small_graph.nodes(user="user").hop(movie=cascaded)
    pi = kept.with_columns(pi=v.movie.inclusion).pl["pi"].to_list()
    assert pi == [0.5, 0.5, 0.5] and len(kept) == 3


def test_a_planned_walk_estimates_like_an_eager_one(small_graph):
    def counted():
        drawn = step("has_interact").sample(1, by=1, seed=0).inclusion()
        return (small_graph.nodes(user="user").hop(movie=drawn)
                .group_by(v.user, confidence=None).len("seen"))

    eager = counted()
    with jb.optimize(rows=1):
        planned = counted()
    assert planned.pl["seen"].to_list() == eager.pl["seen"].to_list()
    assert (planned.with_columns(se=v.seen.score).pl["se"].to_list()
            == eager.with_columns(se=v.seen.score).pl["se"].to_list())
