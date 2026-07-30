"""The train/serve seam: checkpoint portability, and the two scoring paths.

Training needs torch; serving does not. Everything here that needs torch is
skipped when it is absent, so the base install still runs a meaningful suite --
the checkpoint round-trip and the numpy scorer are exercised from hand-built
weights, no training involved.
"""

import numpy as np
import pytest

from jerboas import Node, Score, Embedding
from jerboas.checkpoint import save, load
from jerboas.strategies.embedding import SCORERS

torch = pytest.importorskip("torch", reason="training path needs the [torch] extra")

from jerboas.models import MODELS, TransD, TransE, train       # noqa: E402
from jerboas.models.train import triples, type_bounds          # noqa: E402


@pytest.fixture(params=sorted(MODELS))
def fitted(request, small_graph, tmp_path):
    """Every registered model, trained briefly and written to a checkpoint.

    Parameterized rather than fixed to TransD so that adding a model puts it
    through the whole seam -- training, round-trip, scorer agreement, ranking --
    without anyone remembering to extend the suite."""
    model = train(MODELS[request.param](factors=8, seed=1), small_graph,
                  epochs=3, batch_size=8, device="cpu", report=None)
    path = str(tmp_path / "m.npz")
    model.save(path)
    return model, path


# --- the train/serve seam must be covered on both sides ---------------------

def test_every_model_has_a_scorer():
    """The invariant that the orphaned TransE scorer violated: a model with no
    scorer trains into an unreadable checkpoint, and a scorer with no model is
    maths nothing compares against its torch original."""
    assert set(MODELS) == set(SCORERS), (
        f"models without a scorer: {set(MODELS) - set(SCORERS)}; "
        f"scorers without a model: {set(SCORERS) - set(MODELS)}"
    )


def test_model_names_match_their_registry_key():
    for key, model in MODELS.items():
        assert model.name == key


# --- the triple store is the CSR --------------------------------------------

def test_triples_read_off_the_csr(small_graph):
    head, relation, tail = triples(small_graph)
    assert len(head) == len(relation) == len(tail) == len(small_graph.out_indices)
    # every extracted triple is an edge the graph agrees exists
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
        types = [small_graph.type_of(c) for c in corrupted]
        assert types == [small_graph.type_of(n) for n in nodes]


# --- checkpoint --------------------------------------------------------------

def test_checkpoint_round_trip(small_graph, fitted):
    model, path = fitted
    restored = load(path, small_graph)
    assert restored.model == model.name and restored.factors == 8
    assert restored.missing == []
    for name in model.tables:
        trained = getattr(model, name).weight.detach().numpy()
        assert np.allclose(restored.tensors[name], trained, atol=1e-6), name


def test_checkpoint_rebinds_by_name_not_position(small_graph, fitted, tmp_path, graph_rows):
    """The point of storing identity: a graph whose ids came out in a different
    order still gets each node its own weights.

    This is the bug the format exists to prevent -- a position-keyed checkpoint
    would load here without complaint and score the wrong entities."""
    _model, path = fitted
    restored = load(path, small_graph)

    from jerboas import Graph

    def write(name, rows):
        p = tmp_path / name
        p.write_text("\n".join("\t".join(r) for r in rows) + "\n")
        return str(p)

    # same rows, read in an order that lays the type blocks out differently
    shuffled = Graph(
        kg=write("b.kg", list(reversed(graph_rows["kg"]))),
        ui=write("b.ui", list(reversed(graph_rows["ui"]))),
        attrs=[write("b.genre", graph_rows["genre"]),
               write("b.person", graph_rows["person"]),
               write("b.movie", graph_rows["movie"])],
    )
    rebound = load(path, shuffled)
    assert rebound.missing == []

    keys = ["movie.1", "movie.3", "person.2", "genre.1", "user.1"]
    moved = [k for k in keys if small_graph.lookup(k) != shuffled.lookup(k)]
    assert moved, "the two graphs happen to agree on every id; test proves nothing"

    for key in keys:
        a, b = small_graph.lookup(key), shuffled.lookup(key)
        assert np.allclose(restored.tensors["entity"][a], rebound.tensors["entity"][b]), key


def test_checkpoint_reports_nodes_it_never_saw(small_graph, fitted, tmp_path):
    _model, path = fitted
    from jerboas import Graph
    extra = tmp_path / "c.kg"
    extra.write_text("movie.99\tdirected_by\tperson.1\n")
    bigger = Graph(kg=str(extra))
    rebound = load(path, bigger)
    assert bigger.lookup("movie.99") in rebound.missing
    assert np.allclose(rebound.tensors["entity"][bigger.lookup("movie.99")], 0.0)


def test_rejects_a_future_format(small_graph, fitted, tmp_path):
    _model, path = fitted
    data = dict(np.load(path, allow_pickle=True))
    data["format"] = np.asarray(999)
    bumped = str(tmp_path / "future.npz")
    np.savez_compressed(bumped, **data)
    with pytest.raises(ValueError, match="format"):
        load(bumped, small_graph)


# --- the duplicated scoring functions must agree ----------------------------

def test_torch_and_numpy_score_agree(small_graph, fitted):
    """The one guard on keeping a torch scorer and a numpy scorer side by side."""
    model, path = fitted
    restored = load(path, small_graph)
    code = small_graph.relation_code("directed_by")
    row = restored.relation_row(code)

    head, relation, tail = triples(small_graph)
    keep = relation == code
    head, tail = head[keep], tail[keep]

    with torch.no_grad():
        expected = model.score(torch.as_tensor(head),
                               torch.full((len(head),), code),
                               torch.as_tensor(tail)).numpy()
    actual = SCORERS[model.name](restored.tensors, row, head, tail)
    assert np.allclose(expected, actual, atol=1e-4), f"{expected} != {actual}"


# --- the strategy ------------------------------------------------------------

def test_embedding_ranks(small_graph, fitted):
    _model, path = fitted
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(Embedding.load(path, small_graph,
                                     to={"user.1"}, relation="has_interact"))
                .top(3))
    scores = [s for _, s in rows]
    assert len(rows) == 3 and scores == sorted(scores, reverse=True)


def test_embedding_guides_greedy(small_graph, fitted):
    from jerboas import Greedy, Has
    _model, path = fitted
    strategy = Embedding.load(path, small_graph, to={"user.1"})
    assert strategy.supports_guidance
    user, movie = Node("user"), Node("movie")
    rows = list(small_graph.select(user, movie)
                .where(Has(user, "has_interact", movie))
                .rank(strategy).using(Greedy(k=2)))
    assert rows


def test_embedding_is_silent_about_an_unknown_relation(small_graph, fitted):
    _model, path = fitted
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(Embedding.load(path, small_graph, to={"user.1"},
                                     relation="no_such_relation")))
    assert all(s == 0.0 for _, s in rows)


def test_training_needs_edges(tmp_path):
    from jerboas import Graph
    empty = tmp_path / "e.kg"
    empty.write_text("")
    with pytest.raises(ValueError, match="no edges"):
        train(TransD(factors=4), Graph(kg=str(empty)), epochs=1, report=None)
