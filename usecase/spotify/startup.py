"""What a start does: read the graph, fit or load the factorization, warm up.

On the whole graph a start is minutes the first time and seconds after that --
the graph is cached beside its files and the factorization in checkpoints/ --
and every step says how long it took, since a container log that is silent for
minutes looks like a hang.
"""

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager

import jerboas as jb
from jerboas import DiffusedMatrixFactorization, v

from recommend import extend

DATA_DIR = os.environ.get("SPOTIFY_DIR", "./data/spotify/graph-100k")

# startup takes minutes on the whole graph, and says so in uvicorn's own log
log = logging.getLogger("uvicorn.error")

# where the fitted factorization is kept between starts: one per graph, under
# checkpoints/, which the container mounts from the host
CHECKPOINT = os.environ.get(
    "SPOTIFY_CHECKPOINT",
    f"./checkpoints/spotify.{os.path.basename(os.path.normpath(DATA_DIR))}.dmf.npz")

# how many playlists a song must appear in before the factorization may learn
# from it (see `supported`) -- a claim about the data rather than a tuning knob,
# which is why it is here and not hidden in the strategy
SUPPORT = 20

FACTORS = 32
ITERATIONS = 8

READABLE = {"song": "name", "artist": "name", "album": "name", "playlist": "name"}


def load_graph() -> jb.Graph:
    """The playlists and what they contain, plus who performed what.

    `spotify.contains` carries a score per edge -- a song's place in the
    playlist -- which nothing here reads: what a position means is a question
    this service does not ask, which is why the walk weighs every edge alike.

    Cached beside the data: the whole graph takes minutes to read and under a
    second to map back, and the files are what the cache checks itself against."""
    return jb.Graph(
        kg=f"{DATA_DIR}/spotify.kg",
        edges=[f"{DATA_DIR}/spotify.contains"],
        attrs=[f"{DATA_DIR}/spotify.{one}"
               for one in ("song", "artist", "album", "playlist")],
        readable=READABLE,
        cache=f"{DATA_DIR}/.cache",
    )


def supported(graph: jb.Graph) -> jb.Frame:
    """The songs a factorization may learn from, and will then know anything
    about.

    A song in one playlist is not evidence a latent space can hold: with 680 000
    songs whose median support is a single playlist, factorizing everything is
    factorizing noise, and the result ranks soundtrack themes above pop."""
    return (graph.nodes(song="song")
            .with_columns(seen=v.song.contains.count())
            .filter(v.seen >= SUPPORT).select("song"))


def interactions(graph: jb.Graph, known: jb.Frame) -> jb.Frame:
    """The playlist edges into those songs: what the factorization is fitted on.

    Only a fit needs them, and on the whole graph they are 66 million rows, so
    a start that loads a stored factorization never builds them."""
    return graph.edges("contains").filter(v.target.is_in(known))


def fit(graph: jb.Graph) -> tuple[DiffusedMatrixFactorization, jb.Frame]:
    """The factorization on the supported subgraph: loaded when a previous start
    stored this one, fitted and stored otherwise.

    Minutes on the whole graph, so it is paid once rather than at every start.
    "This one" means the hyperparameters and the support threshold the
    checkpoint recorded are the ones declared here -- a stored model fitted
    with other settings is refitted and replaced, never served."""
    known = supported(graph)
    wanted = DiffusedMatrixFactorization(
        factors=FACTORS, iterations=ITERATIONS,
        item_type="song", user_type="playlist", relation="contains")
    stored = _stored(graph, {**wanted.config(), "support": SUPPORT})
    if stored is not None:
        log.info("spotify: factorization loaded from %s", CHECKPOINT)
        return stored, known
    wanted.where = interactions(graph, known)
    wanted.fit(graph)
    os.makedirs(os.path.dirname(CHECKPOINT) or ".", exist_ok=True)
    # by uri: a song's position is the graph's numbering, its uri is Spotify's
    wanted.save(CHECKPOINT, graph, alias="uri", support=SUPPORT)
    log.info("spotify: factorization fitted and stored in %s", CHECKPOINT)
    return wanted, known


def _stored(graph: jb.Graph, expected: dict[str, object]) -> DiffusedMatrixFactorization | None:
    """The stored factorization, when there is one and it is the one expected."""
    if not os.path.exists(CHECKPOINT):
        return None
    model = DiffusedMatrixFactorization.load(CHECKPOINT, graph)
    differs = {key: (model.meta.get(key), value) for key, value in expected.items()
               if model.meta.get(key) != value}
    if differs:
        log.info("spotify: %s was fitted with other settings %s; refitting",
                 CHECKPOINT, differs)
        return None
    return model


def warm(graph: jb.Graph, model: DiffusedMatrixFactorization, known: jb.Frame) -> None:
    """One request before the first caller's.

    The first request builds what every later one reads -- the word index over
    song titles, each relation's bounds in the store, `Concentration`'s
    products over the whole graph -- which on the whole graph is half a minute.
    Asked here, with the title of whichever song comes first, it is paid while
    the service is starting rather than by whoever asks first."""
    title = graph.nodes(song="song").head(1).attrs(song="name").pl["song.name"][0]
    extend(graph, model, known, [title], 5)


@contextmanager
def timed(what: str) -> Generator[None, None, None]:
    """Say how long a startup step took."""
    start = time.perf_counter()
    yield
    log.info("spotify: %s ready in %.1f s", what, time.perf_counter() - start)
