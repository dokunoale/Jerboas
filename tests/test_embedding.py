"""The train/serve seam: one definition of a model, one portable checkpoint.

Training needs torch; serving does not. Everything requiring torch is skipped
when it is absent, so a base install still exercises the format and the ranking
path.
"""

import numpy as np
import pytest

from jerboas import Node, Score, Embedding
from jerboas.checkpoint import FORMAT, load
from jerboas.kge import NODE, RELATION, REGISTRY

torch = pytest.importorskip("torch", reason="training needs the [torch] extra")

from jerboas.models import MODELS, train                      # noqa: E402
from jerboas.models.train import triples, type_bounds          # noqa: E402


@pytest.fixture(params=sorted(MODELS))
def fitted(request, small_graph, tmp_path):
    """Every registered model, trained briefly and written to a checkpoint.

    Parameterized rather than fixed to one model, so adding a model puts it
    through the whole seam -- training, round-trip, rebinding, ranking -- without
    anyone remembering to extend the suite."""
    model = train(MODELS[request.param](factors=8, seed=1), small_graph,
                  epochs=3, batch_size=8, device="cpu", report=None)
    path = str(tmp_path / "m.npz")
    model.save(path)
    return model, path


# --- one definition per model ------------------------------------------------

def test_a_model_is_its_spec():
    """There is no torch scorer and numpy scorer to keep in step: both sides
    reach the same kge.Model, so drift between them is not expressible."""
    for name, cls in MODELS.items():
        assert cls.spec is REGISTRY[name]
        assert cls.spec.name == name


def test_registry_and_trainers_cover_the_same_models():
    assert set(MODELS) == set(REGISTRY)


def test_tables_declare_their_index_space():
    for model in REGISTRY.values():
        assert model.tables, model.name
        assert all(t.space in (NODE, RELATION) for t in model.tables)
        assert len(set(model.names)) == len(model.names)      # no duplicate names


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
    restored = load(path, small_graph)
    assert restored.name == model.name and restored.factors == 8
    assert restored.missing_nodes == [] and restored.missing_relations == []
    for name in model.spec.names:
        trained = model.tables[name].weight.detach().numpy()
        assert np.allclose(restored.tensors[name], trained, atol=1e-6), name


def test_provenance_is_readable_without_the_graph(small_graph, fitted):
    """Provenance exists for the moment when only the files are left, so it has
    to be legible straight out of the archive."""
    _model, path = fitted
    with np.load(path, allow_pickle=False) as data:
        recorded = {k[len("meta_"):]: data[k].item()
                    for k in data.files if k.startswith("meta_")}
    for key in ("trained_at", "graph_nodes", "graph_edges", "epochs", "lr", "sampler"):
        assert key in recorded, recorded
    assert recorded["graph_nodes"] == small_graph.n_nodes and recorded["epochs"] == 3
    assert load(path, small_graph).meta == recorded


def test_rejects_a_future_format(small_graph, fitted, tmp_path):
    _model, path = fitted
    data = dict(np.load(path, allow_pickle=False))
    data["format"] = np.asarray(FORMAT + 1)
    bumped = str(tmp_path / "future.npz")
    np.savez_compressed(bumped, **data)
    with pytest.raises(ValueError, match="format"):
        load(bumped, small_graph)


def test_rejects_an_unknown_model(small_graph, fitted, tmp_path):
    _model, path = fitted
    data = dict(np.load(path, allow_pickle=False))
    data["model"] = np.asarray("transwhatever")
    odd = str(tmp_path / "odd.npz")
    np.savez_compressed(odd, **data)
    with pytest.raises(ValueError, match="unknown model"):
        load(odd, small_graph)


# --- rebinding ---------------------------------------------------------------

def test_rebinds_nodes_by_name_not_position(small_graph, fitted, tmp_path, graph_rows):
    """The bug the format exists to prevent: a position-keyed checkpoint would
    load here without complaint and score the wrong entities."""
    _model, path = fitted
    restored = load(path, small_graph)
    shuffled = _rebuild(tmp_path, graph_rows, reverse=True)
    rebound = load(path, shuffled)
    assert rebound.missing_nodes == []

    keys = ["movie.1", "movie.3", "person.2", "genre.1", "user.1"]
    assert [k for k in keys if small_graph.lookup(k) != shuffled.lookup(k)], \
        "the two graphs agree on every id; this test would prove nothing"
    for key in keys:
        a, b = small_graph.lookup(key), shuffled.lookup(key)
        assert np.allclose(restored.tensors["entity"][a], rebound.tensors["entity"][b]), key


def test_rebinds_relations_by_name(small_graph, fitted, tmp_path, graph_rows):
    """Relation codes come from load order too, so the relation tables are
    rebound the same way -- after load, every table speaks the caller's ids."""
    _model, path = fitted
    restored = load(path, small_graph)
    shuffled = _rebuild(tmp_path, graph_rows, reverse=True)
    rebound = load(path, shuffled)

    assert list(small_graph.relations) != list(shuffled.relations), "codes did not move"
    for name in small_graph.relations:
        a, b = small_graph.relation_code(name), shuffled.relation_code(name)
        assert np.allclose(restored.tensors["relation"][a],
                           rebound.tensors["relation"][b]), name


def test_reports_nodes_it_never_saw(small_graph, fitted, tmp_path):
    from jerboas import Graph
    _model, path = fitted
    extra = tmp_path / "c.kg"
    extra.write_text("movie.99\tdirected_by\tperson.1\n")
    bigger = Graph(kg=str(extra))
    rebound = load(path, bigger)
    unseen = bigger.lookup("movie.99")
    assert unseen in rebound.missing_nodes
    assert np.allclose(rebound.tensors["entity"][unseen], 0.0)


def test_reports_relations_it_never_saw(small_graph, fitted, tmp_path):
    from jerboas import Graph
    _model, path = fitted
    extra = tmp_path / "d.kg"
    extra.write_text("movie.1\tinspired_by\tmovie.2\n")
    other = Graph(kg=str(extra))
    rebound = load(path, other)
    assert other.relation_code("inspired_by") in rebound.missing_relations
    assert not rebound.knows(other.relation_code("inspired_by"))


# --- scoring is the same function on both sides ------------------------------

def test_torch_and_numpy_paths_agree(small_graph, fitted):
    """The arithmetic is shared, so this guards what is not: the gathering. A
    wrong table or a mis-resolved relation row would show up here."""
    model, path = fitted
    restored = load(path, small_graph)
    code = small_graph.relation_code("directed_by")

    head, relation, tail = triples(small_graph)
    keep = relation == code
    head, tail = head[keep], tail[keep]

    with torch.no_grad():
        expected = model.score(torch.as_tensor(head),
                               torch.full((len(head),), int(code)),
                               torch.as_tensor(tail)).numpy()
    actual = restored.score(head, np.full(len(head), code), tail)
    assert np.allclose(expected, actual, atol=1e-4), f"{expected} != {actual}"


# --- the strategy ------------------------------------------------------------

def test_embedding_ranks(small_graph, fitted):
    _model, path = fitted
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(Embedding.load(path, small_graph, to={"user.1"})).top(3))
    scores = [s for _, s in rows]
    assert len(rows) == 3 and scores == sorted(scores, reverse=True)


def test_embedding_can_score_a_relation_backwards(small_graph, fitted):
    # directed_by runs movie -> person, so ranking movies for a person seed
    # has to read it the other way round
    _model, path = fitted
    movie = Node("movie")
    forward = [s for _, s in small_graph.select(movie, Score()).rank(
        Embedding.load(path, small_graph, to={"person.1"}, relation="directed_by"))]
    reverse = [s for _, s in small_graph.select(movie, Score()).rank(
        Embedding.load(path, small_graph, to={"person.1"},
                       relation="directed_by", reverse=True))]
    assert forward != reverse


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
    from jerboas.models import TransD
    empty = tmp_path / "e.kg"
    empty.write_text("")
    with pytest.raises(ValueError, match="no edges"):
        train(TransD(factors=4), Graph(kg=str(empty)), epochs=1, report=None)


def test_saving_an_unbuilt_model_is_refused(tmp_path):
    from jerboas.models import TransD
    with pytest.raises(ValueError, match="not been built"):
        TransD(factors=4).save(str(tmp_path / "x.npz"))


# --- helpers -----------------------------------------------------------------

def _rebuild(tmp_path, rows, reverse=False):
    """The same data, read in an order that lays out ids and relation codes
    differently."""
    from jerboas import Graph

    def write(name, content):
        path = tmp_path / name
        path.write_text("\n".join("\t".join(r) for r in content) + "\n")
        return str(path)

    order = (lambda x: list(reversed(x))) if reverse else list
    return Graph(
        kg=write("r.kg", order(rows["kg"])),
        ui=write("r.ui", order(rows["ui"])),
        attrs=[write("r.genre", rows["genre"]), write("r.person", rows["person"]),
               write("r.movie", rows["movie"])],
    )
