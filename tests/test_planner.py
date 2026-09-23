"""The planner: a walk described, then taken in slices, from the cheaper end.

Every test here has the same shape, because the planner has one promise: the
frame a planned walk builds is the frame the eager walk would have built -- the
same rows, in the same order, with the same shadows. What changes is only how
much of it exists at once and which end it was walked from.
"""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import jerboas as jb
from jerboas import Graph, v
from jerboas.plan.optimize import Budget, row_bytes
from jerboas.query import traverse


@pytest.fixture
def tangle():
    """Two types, three relations, multi-edges and distinct weights: every way
    two walks could disagree about order has a chance to."""
    rng = np.random.default_rng(7)
    rows = []
    for relation, (head, tail) in {"r": ("a", "b"), "s": ("a", "a"),
                                   "t": ("b", "a")}.items():
        for _ in range(70):
            rows.append((f"{head}.{rng.integers(0, 12)}", f"{tail}.{rng.integers(0, 12)}",
                         relation))
        rows += rows[-6:]                             # parallel edges
    edges = pl.DataFrame(rows, schema=["source", "target", "relation"], orient="row")
    edges = edges.with_columns(score=pl.Series(rng.permutation(len(rows)).astype(float)))
    return Graph.from_frames(edges, renumber=True)


def _ids(graph, type_, locals_):
    return graph.ids_of([f"{type_}.{one}" for one in locals_])


# --- walking from the far end -------------------------------------------------

@pytest.mark.parametrize("relation,reverse", [
    ("r", False), ("r", True), ("s", None), ("t", True),
    (None, None), (None, False), (None, True)])
def test_reach_is_expand_kept_to_the_set(tangle, relation, reverse):
    """Same arrays, same order: the canonical store is what makes that true."""
    rng = np.random.default_rng(3)
    nodes = rng.integers(0, tangle.n_nodes, 50)            # repeats included
    targets = rng.choice(tangle.n_nodes, 9, replace=False)
    rows, landed, codes, weights = traverse.expand(tangle, nodes, relation, reverse,
                                                   normalized=True)
    keep = np.isin(landed, targets)
    found = traverse.Reach(tangle, targets, relation, reverse).gather(nodes, normalized=True)
    for expected, got in zip((rows[keep], landed[keep], codes[keep], weights[keep]), found):
        np.testing.assert_array_equal(expected, got)


def _count_reaches(monkeypatch):
    calls = []
    original = traverse.Reach.gather

    def gather(self, *args, **kwargs):
        calls.append(self.size)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(traverse.Reach, "gather", gather)
    return calls


@pytest.mark.parametrize("spec", ["r", ("r", "~t"), ()])
@pytest.mark.parametrize("rows", [None, 7])
def test_a_walk_that_must_land_in_a_small_set_starts_there(tangle, monkeypatch, spec, rows):
    wanted = _ids(tangle, "b", [1, 4]).tolist() + _ids(tangle, "a", [2]).tolist()

    def query():
        return tangle.nodes(x="a").hop(y=spec).filter(v.y.is_in(wanted))

    eager = query()
    calls = _count_reaches(monkeypatch)
    with jb.optimize(rows=rows):
        planned = query()
    assert planned.raw.equals(eager.raw)
    assert calls, "the planner should have walked from the set"


def test_a_set_bigger_than_the_frame_is_not_walked_from(tangle, monkeypatch):
    few = tangle.nodes(x="a").head(1)
    everything = tangle.nodes(y="b").ids("y").tolist()
    calls = _count_reaches(monkeypatch)
    with jb.optimize():
        planned = few.hop(y="r").filter(v.y.is_in(everything))
    assert planned.raw.equals(few.hop(y="r").filter(v.y.is_in(everything)).raw)
    assert not calls


def test_a_membership_inside_a_conjunction_still_says_where_to_land(tangle, monkeypatch):
    wanted = _ids(tangle, "b", [3]).tolist()

    def query():
        return tangle.nodes(x="a").hop(y="r").filter(
            v.y.is_in(wanted) & (v.y.score >= 0.1))

    eager = query()
    calls = _count_reaches(monkeypatch)
    with jb.optimize():
        planned = query()
    assert planned.raw.equals(eager.raw)
    assert calls


# --- stages ---------------------------------------------------------------------

def test_a_hop_after_a_planned_hop_extends_the_plan(tangle):
    """So the middle never exists whole, and a condition on the middle prunes
    it before the next step multiplies it."""
    middle = _ids(tangle, "b", [0, 2, 5, 7]).tolist()

    def query():
        return (tangle.nodes(x="a").hop(mid="r").filter(v.mid.is_in(middle))
                .hop(end="t"))

    eager = query()
    with jb.optimize(rows=3):
        planned = query()
        assert len(planned._plan.stages) == 2
        assert len(planned._plan.stages[0][1]) == 1        # decided at the middle
    assert planned.raw.equals(eager.raw)


def test_a_condition_joins_the_earliest_stage_that_can_decide_it(tangle):
    with jb.optimize(rows=3):
        walk = tangle.nodes(x="a").hop(mid="r").hop(end="t").filter(
            v.mid.score >= 0.5, v.end.score >= 0.5)
        first, second = walk._plan.stages
    assert len(first[1]) == 1 and len(second[1]) == 1


def test_a_planned_walk_names_the_relation_it_walked(small_graph):
    with jb.optimize(rows=1):
        frame = small_graph.nodes(user="user").hop(rec="has_interact")
    assert set(frame.with_columns(w=v.rec.via).pl["w"].to_list()) == {"has_interact"}


# --- conditions that need the whole answer ----------------------------------------

@pytest.mark.parametrize("conditions", [
    lambda: [v.rec.score >= v.rec.score.mean()],
    lambda: [v.rec.score >= v.rec.score.mean(), v.rec.year >= 1994],
    lambda: [pl.col("rec") == pl.col("rec").max()],
    lambda: [v.rec.label.like("Alpha", rule=jb.Fuzzy(k=1))],
])
def test_a_condition_that_reads_other_rows_sees_the_whole_answer(small_graph, conditions):
    def query():
        return small_graph.nodes(user="user").hop(rec="has_interact").filter(*conditions())

    eager = query()
    with jb.optimize(rows=1):
        planned = query()
    assert planned.raw.equals(eager.raw)


# --- streaming ------------------------------------------------------------------

def test_batches_are_the_answer_a_slice_at_a_time(tangle):
    with jb.optimize(rows=10):
        walk = tangle.nodes(x="a").hop(y="r")
    parts = list(walk.batches())
    assert len(parts) > 1
    assert pl.concat([one.raw for one in parts]).equals(walk.raw)


def test_a_frame_that_was_not_planned_is_one_batch(small_graph):
    frame = small_graph.nodes(user="user")
    assert [one is frame for one in frame.batches()] == [True]


@pytest.mark.parametrize("over", [None, "x"])
def test_top_is_kept_a_slice_at_a_time(tangle, over):
    def query():
        return tangle.nodes(x="a").hop(y="r")

    eager = query().top(3, by=v.y.score, over=over)
    with jb.optimize(rows=5):
        walk = query()
    best = walk.top(3, by=v.y.score, over=over)
    assert walk._plan is not None                   # never built whole
    assert best.raw.equals(eager.raw)


def test_unique_is_folded_a_slice_at_a_time(tangle):
    eager = tangle.nodes(x="a").hop(y="r").unique(["x", "y"])
    with jb.optimize(rows=5):
        walk = tangle.nodes(x="a").hop(y="r")
    folded = walk.unique(["x", "y"])
    assert walk._plan is not None
    assert folded.raw.equals(eager.raw)


@pytest.mark.parametrize("confidence", ["mean", "min", "max", "product", "first", None])
def test_an_aggregate_that_decomposes_is_folded_a_slice_at_a_time(tangle, confidence):
    def aggregate(frame):
        return frame.group_by(v.y, confidence=confidence).agg(
            total=v.y.score.sum(), n=v.x.count(), average=v.y.score.mean(),
            low=v.y.score.min(), high=v.y.score.max(),
            earliest=v.x.first(), latest=v.x.last())

    eager = aggregate(tangle.nodes(x="a").hop(y="r"))
    with jb.optimize(rows=5):
        walk = tangle.nodes(x="a").hop(y="r")
    folded = aggregate(walk)
    assert walk._plan is not None
    # equal up to the order a float sum was taken in
    assert_frame_equal(folded.raw, eager.raw)


def test_group_len_is_folded_a_slice_at_a_time(tangle):
    eager = tangle.nodes(x="a").hop(y="r").group_by(v.y).len()
    with jb.optimize(rows=5):
        walk = tangle.nodes(x="a").hop(y="r")
    assert_frame_equal(walk.group_by(v.y).len().raw, eager.raw)


def test_an_aggregate_that_does_not_decompose_is_taken_whole(tangle):
    eager = tangle.nodes(x="a").hop(y="r").group_by(v.y).agg(spread=v.y.score.std())
    with jb.optimize(rows=5):
        walk = tangle.nodes(x="a").hop(y="r")
    assert walk.group_by(v.y).agg(spread=v.y.score.std()).raw.equals(eager.raw)


def test_slices_that_disagree_about_a_confidence_are_reconciled():
    """The first user's edges both carry the top weight, so its slice has no
    confidence column at all; the second's does. Stacked, the first reads 1.0."""
    edges = pl.DataFrame({"source": ["user.0", "user.0", "user.1"],
                          "target": ["movie.0", "movie.1", "movie.0"],
                          "score": [2.0, 2.0, 1.0]})
    graph = Graph.from_frames({"saw": edges})
    eager = graph.nodes(u="user").hop(m="saw")
    with jb.optimize(rows=2):
        planned = graph.nodes(u="user").hop(m="saw")
        assert [one.hidden for one in planned.batches()] == [[], ["__jb_score__m"]]
    assert planned.raw.equals(eager.raw)


# --- the budget -------------------------------------------------------------------

def test_memory_becomes_rows_at_the_frame_s_width(small_graph):
    users = small_graph.nodes(user="user")
    per_row = row_bytes(users, [("has_interact", "rec")])
    assert Budget(memory=10 * per_row).rows(users, [("has_interact", "rec")]) == 10
    assert Budget(rows=4).rows(users) == 4


def test_a_wider_frame_gets_fewer_rows(small_graph):
    narrow = small_graph.nodes(rec="movie")
    wide = narrow.attrs(rec="title").attrs(rec="year")
    budget = Budget(memory=1 << 20)
    assert budget.rows(wide, [("has_genre", "g")]) < budget.rows(narrow, [("has_genre", "g")])


def test_the_budget_is_said_one_way():
    with pytest.raises(ValueError, match="not both"):
        with jb.optimize(rows=10, memory=10):
            pass


def test_no_budget_means_the_memory_that_is_free(small_graph):
    with jb.optimize():
        walk = small_graph.nodes(user="user").hop(rec="has_interact")
        assert walk._plan.budget.memory > 0
    assert walk.raw.equals(small_graph.nodes(user="user").hop(rec="has_interact").raw)


# --- a confidence nobody measured, in an aggregate ---------------------------------

def test_an_unmeasured_confidence_sums_to_the_count(small_graph):
    """Every edge of directed_by is 1.0 and none is stored: summed over a
    person, that is how many films, not one."""
    totals = (small_graph.nodes(m="movie").hop(p="directed_by")
              .group_by(v.p).agg(total=v.p.score.sum()))
    by_person = dict(zip((str(key) for key in totals.keys("p")), totals.pl["total"]))
    assert by_person == {"person.0": 2.0, "person.1": 1.0}


# --- found on the way: the eager walk, where comparing against it showed it wrong --

def test_a_search_pushed_into_a_hop_keeps_what_it_measured(small_graph):
    """Applied to the hop's arrays or to its rows, a `like` measures the same
    closeness, and it is the column's confidence either way."""
    found = (small_graph.nodes(user="user").hop(rec="has_interact")
             .filter(v.rec.label.like("Alpha", rule=jb.Fuzzy(k=1))))
    assert "__jb_score__rec.label" in found.hidden


def test_a_filter_pushed_into_a_hop_keeps_the_relation_it_walked(small_graph):
    walked = (small_graph.nodes(user="user").hop(rec="has_interact")
              .filter(v.rec.year >= 1994))
    assert set(walked.with_columns(w=v.rec.via).pl["w"].to_list()) == {"has_interact"}


# --- matrices over the store ---------------------------------------------------------

def _reference(graph, relation=None):
    """The matrix the old builder made: every stored edge as a (source, target)
    cell, summed."""
    import scipy.sparse as sp
    keep = (np.ones(len(graph.out_indices), dtype=bool) if relation is None
            else graph.out_rels == graph.relation_code(relation))
    sources = graph.sources()[keep]
    return sp.csr_matrix((np.ones(keep.sum()), (sources, graph.out_indices[keep])),
                         shape=(graph.n_nodes, graph.n_nodes))


@pytest.mark.parametrize("relation", [None, "r", "s", "t"])
def test_a_relation_matrix_is_the_store_read_as_one(tangle, relation):
    built = tangle.relation_matrix(relation)
    assert abs(built - _reference(tangle, relation)).max() == 0


def test_the_adjacency_is_both_directions_counting_multi_edges(tangle):
    reference = _reference(tangle)
    assert abs(tangle.adjacency() - (reference + reference.T)).max() == 0
    forwards, backwards = tangle.matrices()
    assert abs(backwards - forwards.T).max() == 0


def test_a_repeated_edge_is_one_interaction_to_a_factorization():
    once = pl.DataFrame({"source": ["u.0", "u.0", "u.1"], "target": ["i.0", "i.1", "i.1"]})
    twice = pl.concat([once, once.head(1)])
    scores = []
    for edges in (once, twice):
        graph = Graph.from_frames({"likes": edges})
        model = jb.MatrixFactorization(factors=2, iterations=3, item_type="i",
                                       user_type="u", relation="likes", user="u.0")
        scores.append(graph.nodes(item="i").with_columns(s=model.on("item")).pl["s"])
    assert scores[0].to_list() == scores[1].to_list()
