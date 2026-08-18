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

    keys = ["movie.0", "movie.2", "person.1", "genre.0", "user.0"]
    assert [k for k in keys if small_graph.lookup(k) != shuffled.lookup(k)], \
        "the two graphs agree on every id; this test would prove nothing"
    for key in keys:
        a, b = small_graph.lookup(key), shuffled.lookup(key)
        assert np.allclose(here.arrays["entity"][a], there.arrays["entity"][b]), key


def _renumbered(tmp_path, name, uris):
    """A graph whose movies carry `uris`, in order, at positions 0..n-1.

    The point of an alias: an id is a position, so the same entity sits
    somewhere else in a build made from more of the source, and only an
    attribute survives that."""
    kg = tmp_path / f"{name}.kg"
    kg.write_text("".join(f"movie.{i}\tdirected_by\tperson.0\n"
                          for i in range(len(uris))))
    attrs = tmp_path / f"{name}.movie"
    attrs.write_text("id\turi\ttitle\n"
                     + "".join(f"{i}\t{uri}\t{uri[4:]}\n" for i, uri in enumerate(uris)))
    return Graph(kg=str(kg), attrs=[str(attrs)])


ALPHA, BETA = "urn:alpha", "urn:beta"


def test_an_alias_rebinds_where_the_ids_have_moved(tmp_path):
    """`alias="uri"` keys the checkpoint on an attribute, so a graph that put
    the same entity in a different place still gets its own weights."""
    here = _renumbered(tmp_path, "a", [ALPHA])
    model = train(MODELS["transd"](factors=4, seed=1), here,
                  epochs=1, batch_size=2, device="cpu", report=None)
    path = str(tmp_path / "alias.npz")
    model.save(path, alias="uri")

    there = _renumbered(tmp_path, "b", [BETA, ALPHA])        # alpha moved to 1
    alpha_here, alpha_there = here.lookup("movie.0"), there.lookup("movie.1")
    assert alpha_here != alpha_there, "alpha did not move; this would prove nothing"

    loaded, rebound = type(model).load(path, here), type(model).load(path, there)
    assert np.allclose(loaded.arrays["entity"][alpha_here],
                       rebound.arrays["entity"][alpha_there])
    assert alpha_there not in rebound.missing_nodes
    assert there.lookup("movie.0") in rebound.missing_nodes   # beta was never trained


def test_without_the_alias_the_moved_node_is_simply_unknown(tmp_path):
    """The failure has to be visible: keyed on the position, the entity that
    moved is reported missing rather than quietly given somebody else's row --
    which is what makes the alias the portable way to store a checkpoint."""
    here = _renumbered(tmp_path, "c", [ALPHA])
    model = train(MODELS["transd"](factors=4, seed=1), here,
                  epochs=1, batch_size=2, device="cpu", report=None)
    path = str(tmp_path / "plain.npz")
    model.save(path)                                  # alias defaults to the id

    there = _renumbered(tmp_path, "d", [BETA, ALPHA])
    rebound = type(model).load(path, there)
    moved = there.lookup("movie.1")
    assert moved in rebound.missing_nodes
    assert np.allclose(rebound.arrays["entity"][moved], 0.0)


def test_an_older_checkpoint_reads_as_keyed_on_the_id(small_graph, fitted, tmp_path):
    """Format 2 is format 3 without `alias`, so the files already on disk keep
    loading and keep meaning what they meant."""
    model, path = fitted
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "alias"}
    payload["format"] = np.asarray(2)
    older = str(tmp_path / "v2.npz")
    np.savez_compressed(older, **payload)

    new, old = type(model).load(path, small_graph), type(model).load(older, small_graph)
    assert old.missing_nodes == []
    assert np.allclose(new.arrays["entity"], old.arrays["entity"])


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
    """A fourth film, where the fixture had three: an id past the end of what
    was trained is the shape "the checkpoint has never met this node" takes once
    ids are positions."""
    model, path = fitted
    extra = tmp_path / "c.kg"
    extra.write_text("".join(f"movie.{i}\tdirected_by\tperson.0\n" for i in range(4)))
    bigger = Graph(kg=str(extra))
    rebound = type(model).load(path, bigger)
    unseen = bigger.lookup("movie.3")
    assert unseen in rebound.missing_nodes
    assert np.allclose(rebound.arrays["entity"][unseen], 0.0)


def test_reports_relations_it_never_saw(fitted, tmp_path):
    model, path = fitted
    extra = tmp_path / "d.kg"
    extra.write_text("movie.0\tinspired_by\tmovie.1\n")
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
                .rank(type(model).load(path, small_graph, to={"user.0"})).top(3))
    scores = [s for _, s in rows]
    assert len(rows) == 3 and scores == sorted(scores, reverse=True)


def test_can_score_a_relation_backwards(small_graph, fitted):
    # directed_by runs movie -> person, so ranking movies for a person seed has
    # to read it the other way round
    model, path = fitted
    movie = Node("movie")
    forward = [s for _, s in small_graph.select(movie, Score()).rank(
        type(model).load(path, small_graph, to={"person.0"}, relation="directed_by"))]
    reverse = [s for _, s in small_graph.select(movie, Score()).rank(
        type(model).load(path, small_graph, to={"person.0"},
                         relation="directed_by", reverse=True))]
    assert forward != reverse


def test_guides_greedy(small_graph, fitted):
    model, path = fitted
    strategy = type(model).load(path, small_graph, to={"user.0"})
    user, movie = Node("user"), Node("movie")
    rows = list(small_graph.select(user, movie)
                .where(Has(user, "has_interact", movie))
                .rank(strategy).using(Greedy(k=2)))
    assert rows


def test_is_silent_about_an_unknown_relation(small_graph, fitted):
    model, path = fitted
    movie = Node("movie")
    rows = list(small_graph.select(movie, Score())
                .rank(type(model).load(path, small_graph, to={"user.0"},
                                       relation="no_such_relation")))
    assert all(s == 0.0 for _, s in rows)


# --- link prediction without naming the relation -----------------------------

def test_any_relation_is_the_default(small_graph, fitted):
    """Left unnamed, the score is the best edge of any kind in either direction:
    the only thing that works when seeds are of mixed types."""
    model, path = fitted
    loaded = model.__class__.load(path, small_graph)
    loaded.fit(small_graph)
    assert len(loaded._edges) == 2 * len(small_graph.relations)


def test_naming_a_relation_narrows_it(small_graph, fitted):
    model, path = fitted
    narrow = model.__class__.load(path, small_graph, relation="has_genre")
    narrow.fit(small_graph)
    assert narrow._edges == ((small_graph.relation_code("has_genre"), False),)


def test_any_relation_finds_each_seed_type_its_own_edge(small_graph, fitted):
    """A person is joined to a film by directed_by backwards, a user by
    has_interact forwards. Neither is named, and both still rank."""
    model, path = fitted
    movie = Node("movie")
    for seed in ({"person.0"}, {"user.0"}, {"genre.0"}):
        rows = list(small_graph.select(movie, Score())
                    .rank(model.__class__.load(path, small_graph, to=seed)).top(3))
        assert len(rows) == 3
        assert any(s > 0 for _key, s in rows), seed


# --- seeds are cheap to change, weights are not ------------------------------

def test_seeded_shares_the_weights(small_graph, fitted):
    """A service loads once and re-aims per request; rebinding is linear in the
    graph, so it must not happen again."""
    model, path = fitted
    loaded = model.__class__.load(path, small_graph)
    aimed = loaded.seeded({"user.0"})
    for table, _space in model.tables:
        assert aimed.arrays[table] is loaded.arrays[table]
    assert aimed._to == {"user.0"} and loaded._to is None


def test_seeded_ranks_the_same_as_a_fresh_load(small_graph, fitted):
    model, path = fitted
    movie = Node("movie")
    fresh = list(small_graph.select(movie, Score())
                 .rank(model.__class__.load(path, small_graph, to={"user.0"})))
    reused = list(small_graph.select(movie, Score())
                  .rank(model.__class__.load(path, small_graph).seeded({"user.0"})))
    assert [str(k) for k, _ in fresh] == [str(k) for k, _ in reused]
    assert np.allclose([s for _, s in fresh], [s for _, s in reused])


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
        edges=[write("r.has_interact",
                     [("source", "target", "score")] + list(reversed(rows["has_interact"])))],
        attrs=[write("r.genre", rows["genre"]), write("r.person", rows["person"]),
               write("r.movie", rows["movie"])],
    )


# --- edge weights ------------------------------------------------------------

def test_training_weights_each_example_by_its_edge(small_graph):
    """The weighted run is a different fit, not the same one relabelled -- which
    is the whole claim of moving the score filter out of the loader."""
    from jerboas.models import TransE

    def fit(weighted):
        return train(TransE(factors=4, seed=1), small_graph, epochs=3, batch_size=8,
                     device="cpu", weighted=weighted, report=None
                     ).weights["entity"].weight.detach().numpy().copy()

    assert not np.allclose(fit(True), fit(False))


def test_weighting_is_inert_on_an_unweighted_graph(tmp_path):
    from jerboas.models import TransE

    kg = tmp_path / "u.kg"
    kg.write_text("movie.0\tdirected_by\tperson.0\nmovie.1\tdirected_by\tperson.1\n")
    graph = Graph(kg=str(kg))

    def fit(weighted):
        return train(TransE(factors=4, seed=1), graph, epochs=3, batch_size=8,
                     device="cpu", weighted=weighted, report=None
                     ).weights["entity"].weight.detach().numpy().copy()

    # every weight is 1.0, so weighting by it is multiplying by one
    assert np.allclose(fit(True), fit(False))


def test_where_narrows_what_a_run_learns_from(small_graph):
    """The threshold that used to live in the loader, as an argument to the fit:
    the graph still holds the 1-star interaction, this run just never sees it."""
    from jerboas import Edge
    from jerboas.models.train import admitted, triples

    keep = admitted(small_graph, [Edge("has_interact", score=(3, None))])
    _head, relation, _tail = triples(small_graph, keep)
    kept = [small_graph.relations[code] for code in relation]

    assert kept.count("has_interact") == 4          # 5, 4, 3, 5 -- not the 1 or the 2
    assert kept.count("has_genre") == 3             # unscored relations are untouched
    assert len(triples(small_graph)[0]) == len(kept) + 2


def test_a_marker_naming_no_relation_constrains_every_edge(small_graph):
    from jerboas import Edge
    from jerboas.models.train import admitted

    keep = admitted(small_graph, [Edge(score=(3, None))])
    relations = {small_graph.relations[c] for c in small_graph.out_rels[keep]}
    assert relations == {"has_interact"}            # everything unscored weighs 1


def test_the_filter_is_recorded_in_the_checkpoint(small_graph, tmp_path):
    from jerboas import Edge
    from jerboas.models import TransE

    model = train(TransE(factors=4, seed=1), small_graph, epochs=1, batch_size=8,
                  device="cpu", where=[Edge("has_interact", score=(3, None))], report=None)
    assert model.meta["trained_on"] == "has_interact.score ge 3"
    assert model.meta["trained_edges"] == 10        # 6 in the kg, 4 interactions left

    path = str(tmp_path / "w.npz")
    model.save(path)
    assert TransE.load(path, small_graph).meta["trained_on"] == "has_interact.score ge 3"


def test_a_filter_that_admits_nothing_says_so(small_graph):
    from jerboas import Edge
    from jerboas.models import TransE

    with pytest.raises(ValueError, match="no edge satisfies"):
        train(TransE(factors=4), small_graph, epochs=1,
              where=[Edge(score=(99, None))], report=None)
