"""Embedding models: one class that trains, stores itself, and ranks.

Fitting needs torch, so everything here skips without the [torch] extra.
"""

import numpy as np
import pytest

from jerboas import Graph, Greedy, Has, Node, Score, Strategy
from jerboas.checkpoint import FORMAT

torch = pytest.importorskip("torch", reason="fitting needs the [torch] extra")

from jerboas.models import MODELS, train                       # noqa: E402
from jerboas.models.base import NODE, RELATION                 # noqa: E402
from jerboas.models.train import triples, type_bounds          # noqa: E402


@pytest.fixture(params=sorted(MODELS))
def fitted(request, small_graph, tmp_path):
    """Every model, trained briefly and written to a checkpoint.

    Parameterized rather than fixed to one, so adding a model puts it through
    training, round-trip, rebinding and ranking without anyone remembering to
    extend the suite."""
    model = train(MODELS[request.param](factors=8, seed=1), small_graph,
                  epochs=3, batch_size=8, device="cpu", report=None)
    path = str(tmp_path / "m.npz")
    model.save(path)
    return model, path


# --- a model is one class ----------------------------------------------------

def test_a_model_is_a_strategy():
    """No wrapper and no registry pairing a model with its maths: the class is
    the ranker, so rank(TransD.load(...)) needs nothing around it."""
    for name, model in MODELS.items():
        assert issubclass(model, Strategy)
        assert model.name == name
        assert model.supports_guidance


def test_tables_declare_their_index_space():
    for model in MODELS.values():
        assert model.tables, model.name
        names = [table for table, _space in model.tables]
        assert all(space in (NODE, RELATION) for _table, space in model.tables)
        assert len(set(names)) == len(names)


def test_score_and_plausibility_stay_distinct():
    """`score` is the one interface rank(...) speaks; a triple's plausibility is
    a different quantity and carries a different name."""
    for model in MODELS.values():
        assert model.score is Strategy.score or callable(model.score)
        assert model.plausibility is not model.score


# --- the triple store is the CSR --------------------------------------------

def test_triples_read_off_the_csr(small_graph):
    head, relation, tail = triples(small_graph)
    assert len(head) == len(relation) == len(tail) == len(small_graph.out_indices)
    for h, r, t in list(zip(head.tolist(), relation.tolist(), tail.tolist()))[:20]:
        assert small_graph.has_edge(h, t, r, reverse=False)


def test_type_bounds_cover_each_node_with_its_own_block(small_graph):
    low, high = type_bounds(small_graph)
    for index in range(small_graph.n_nodes):
        start, stop = small_graph.block(small_graph.type_of(index))
        assert (low[index], high[index]) == (start, stop)


def test_corruptions_stay_inside_the_type(small_graph):
    from jerboas.models.train import _corrupt
    low, high = type_bounds(small_graph)
    rng = np.random.default_rng(0)
    nodes = np.arange(small_graph.n_nodes)
    for _ in range(20):
        corrupted = _corrupt(rng, nodes, low, high)
        assert ([small_graph.type_of(c) for c in corrupted]
                == [small_graph.type_of(n) for n in nodes])


# --- the checkpoint file -----------------------------------------------------

def test_checkpoint_is_inert(fitted):
    """No pickled arrays, so opening one cannot execute anything. A .npz that
    needs allow_pickle is code wearing a data extension."""
    _model, path = fitted
    with np.load(path, allow_pickle=False) as data:       # would raise if pickled
        assert all(data[key].dtype != object for key in data.files)


def test_checkpoint_round_trip(small_graph, fitted):
    model, path = fitted
    restored = type(model).load(path, small_graph)
    assert restored.name == model.name and restored.factors == 8
    assert restored.missing_nodes == [] and restored.missing_relations == []
    for table, _space in model.tables:
        trained = model.weights[table].weight.detach().numpy()
        assert np.allclose(restored.arrays[table], trained, atol=1e-6), table


def test_provenance_is_readable_without_the_graph(small_graph, fitted):
    """Provenance exists for the moment when only the files are left, so it has
    to be legible straight out of the archive."""
    model, path = fitted
    with np.load(path, allow_pickle=False) as data:
        recorded = {k[len("meta_"):]: data[k].item()
                    for k in data.files if k.startswith("meta_")}
    for key in ("trained_at", "graph_nodes", "graph_edges", "epochs", "lr", "sampler"):
        assert key in recorded, recorded
    assert recorded["graph_nodes"] == small_graph.n_nodes and recorded["epochs"] == 3
    assert type(model).load(path, small_graph).meta == recorded


def test_rejects_a_future_format(small_graph, fitted, tmp_path):
    model, path = fitted
    bumped = _tweak(path, tmp_path / "future.npz", format=np.asarray(FORMAT + 1))
    with pytest.raises(ValueError, match="format"):
        type(model).load(bumped, small_graph)


def test_refuses_a_checkpoint_from_another_model(small_graph, fitted, tmp_path):
    model, path = fitted
    other = _tweak(path, tmp_path / "other.npz", model=np.asarray("somethingelse"))
    with pytest.raises(ValueError, match="written by"):
        type(model).load(other, small_graph)


# --- rebinding ---------------------------------------------------------------

def test_rebinds_nodes_by_name_not_position(small_graph, fitted, tmp_path, graph_rows):
    """The bug the format exists to prevent: a position-keyed checkpoint would
    load here without complaint and score the wrong entities."""
    model, path = fitted
    here = type(model).load(path, small_graph)
    shuffled = _rebuild(tmp_path, graph_rows)
    there = type(model).load(path, shuffled)
    assert there.missing_nodes == []

    keys = ["movie.1", "movie.3", "person.2", "genre.1", "user.1"]
    assert [k for k in keys if small_graph.lookup(k) != shuffled.lookup(k)], \
        "the two graphs agree on every id; this test would prove nothing"
    for key in keys:
        a, b = small_graph.lookup(key), shuffled.lookup(key)
        assert np.allclose(here.arrays["entity"][a], there.arrays["entity"][b]), key


def test_rebinds_relations_by_name(small_graph, fitted, tmp_path, graph_rows):
    """Relation codes come from load order too, so relation tables are rebound
    the same way -- after load, every table speaks the caller's ids."""
    model, path = fitted
    here = type(model).load(path, small_graph)
    shuffled = _rebuild(tmp_path, graph_rows)
    there = type(model).load(path, shuffled)

    assert list(small_graph.relations) != list(shuffled.relations), "codes did not move"
    for name in small_graph.relations:
        a, b = small_graph.relation_code(name), shuffled.relation_code(name)
        assert np.allclose(here.arrays["relation"][a], there.arrays["relation"][b]), name


def test_reports_nodes_it_never_saw(fitted, tmp_path):
    model, path = fitted
    extra = tmp_path / "c.kg"
    extra.write_text("movie.99\tdirected_by\tperson.1\n")
    bigger = Graph(kg=str(extra))
    rebound = type(model).load(path, bigger)
    unseen = bigger.lookup("movie.99")
    assert unseen in rebound.missing_nodes
    assert np.allclose(rebound.arrays["entity"][unseen], 0.0)


def test_reports_relations_it_never_saw(fitted, tmp_path):
    model, path = fitted
    extra = tmp_path / "d.kg"
    extra.write_text("movie.1\tinspired_by\tmovie.2\n")
    other = Graph(kg=str(extra))
    rebound = type(model).load(path, other)
    assert other.relation_code("inspired_by") in rebound.missing_relations


# --- the fitted and the loaded model agree -----------------------------------

def test_the_two_weight_forms_score_alike(small_graph, fitted):
    """plausibility() is one implementation; `get` is what differs, returning an
    nn.Embedding lookup while fitting and an array row once loaded. This is the
    guard on that seam."""
    model, path = fitted
    loaded = type(model).load(path, small_graph)
    code = small_graph.relation_code("directed_by")

    head, relation, tail = triples(small_graph)
    keep = relation == code
    head, tail = head[keep], tail[keep]

    with torch.no_grad():
        fitted_scores = model.plausibility(
            torch.as_tensor(head), torch.full((len(head),), int(code)),
            torch.as_tensor(tail)).numpy()
    loaded_scores = loaded.plausibility(head, np.full(len(head), code), tail)
    assert np.allclose(fitted_scores, loaded_scores, atol=1e-4)


# --- ranking -----------------------------------------------------------------

def test_ranks_without_a_wrapper(small_graph, fitted):
    model, path = fitted
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(type(model).load(path, small_graph, to={"user.1"})).top(3))
    scores = [s for _, s in rows]
    assert len(rows) == 3 and scores == sorted(scores, reverse=True)


def test_can_score_a_relation_backwards(small_graph, fitted):
    # directed_by runs movie -> person, so ranking movies for a person seed has
    # to read it the other way round
    model, path = fitted
    movie = Node("movie")
    forward = [s for _, s in small_graph.select(movie, Score()).rank(
        type(model).load(path, small_graph, to={"person.1"}, relation="directed_by"))]
    reverse = [s for _, s in small_graph.select(movie, Score()).rank(
        type(model).load(path, small_graph, to={"person.1"},
                         relation="directed_by", reverse=True))]
    assert forward != reverse


def test_guides_greedy(small_graph, fitted):
    model, path = fitted
    strategy = type(model).load(path, small_graph, to={"user.1"})
    user, movie = Node("user"), Node("movie")
    rows = list(small_graph.select(user, movie)
                .where(Has(user, "has_interact", movie))
                .rank(strategy).using(Greedy(k=2)))
    assert rows


def test_is_silent_about_an_unknown_relation(small_graph, fitted):
    model, path = fitted
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(type(model).load(path, small_graph, to={"user.1"},
                                       relation="no_such_relation")))
    assert all(s == 0.0 for _, s in rows)


# --- refusals ----------------------------------------------------------------

def test_training_needs_edges(tmp_path):
    from jerboas import TransD
    empty = tmp_path / "e.kg"
    empty.write_text("")
    with pytest.raises(ValueError, match="no edges"):
        train(TransD(factors=4), Graph(kg=str(empty)), epochs=1, report=None)


def test_saving_an_unbuilt_model_is_refused(tmp_path):
    from jerboas import TransD
    with pytest.raises(ValueError, match="not been built"):
        TransD(factors=4).save(str(tmp_path / "x.npz"))


# --- helpers -----------------------------------------------------------------

def _tweak(path, target, **changes):
    """A copy of a checkpoint with some fields replaced."""
    data = dict(np.load(path, allow_pickle=False))
    data.update(changes)
    np.savez_compressed(target, **data)
    return str(target)


def _rebuild(tmp_path, rows):
    """The same data read in reverse, so ids and relation codes land elsewhere."""
    def write(name, content):
        path = tmp_path / name
        path.write_text("\n".join("\t".join(r) for r in content) + "\n")
        return str(path)

    return Graph(
        kg=write("r.kg", list(reversed(rows["kg"]))),
        ui=write("r.ui", list(reversed(rows["ui"]))),
        attrs=[write("r.genre", rows["genre"]), write("r.person", rows["person"]),
               write("r.movie", rows["movie"])],
    )
