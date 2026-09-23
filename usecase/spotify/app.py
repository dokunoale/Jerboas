"""Playlist continuation (FastAPI service), on the Spotify Million Playlist Dataset.

    ./run.sh spotify                                  # in a container
    uvicorn app:app --app-dir usecase/spotify         # or straight from here

    curl -X POST localhost:8000/extend -H 'content-type: application/json' \
         -d '{"songs": ["Toxic", "Lose Control", "Bad Romance"], "k": 5}'

You bring a handful of songs; the service answers with five more that belong
beside them. There is no playlist node for what you brought -- the input is a
list of songs, and what stands in for the playlist you are building is the
crowd of real ones that already contain those songs.

Three signals, and the query says how much each counts:

  * the **graph** finds the candidates: the playlists holding your songs, and
    what else they hold. Nothing is scored yet; this is only who is in the room.
  * the **count** is the evidence: how many of those playlists also hold this
    song. It is a strong signal on its own and it is damped rather than trusted
    flat -- twice as many playlists is not twice as good an answer.
  * the **model** is the taste: a matrix factorization of the playlist-song
    matrix, asked how near a candidate is to what you brought. On its own it is
    a poor recommender here (see `TRAINED_ON`); as a re-ranker over songs the
    crowd already agrees on, it is what separates *the same artists* from
    *the same decade*.
"""

import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scipy.cluster.vq import kmeans2

import jerboas as jb
from jerboas import Concentration, DiffusedMatrixFactorization, Words, reverse, v

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

# how many equally-titled songs to weigh against each other before picking one
TIES = 8

# how many performers a name written after a tab may mean
PERFORMERS = 4

# whether a performer put a candidate in the pool, rather than a title alone
PINNED = "pinned"

FACTORS = 32
ITERATIONS = 8

READABLE = {"song": "name", "artist": "name", "album": "name", "playlist": "name"}


def asked_for(names):
    """What was typed, as (title, performer or None).

    A tab is the only structure the input has: `"Wonderwall\tOasis"` says which
    Wonderwall, and a name without one says nothing about the performer. It is
    there because titles are not unique and a name typed alone cannot say which
    recording it means -- so a caller who knows gets to say."""
    for name in names:
        title, _, performer = (name or "").partition("\t")
        title, performer = title.strip(), performer.strip()
        if title:
            yield title, (performer or None)


def resolve(graph, names):
    """The songs you named, one each.

    A name typed by a person is a search box, not a key: `like` admits the
    closest stored titles rather than the equal one, and what it landed on comes
    back in the answer so a wrong guess is visible rather than silent.

    Every name is resolved in the same frame, which is what makes the choice a
    good one. Candidates arrive by two routes -- `_by_performer` for the names
    that named one, `_by_title` for every name -- and both land in one pool
    tagged with the name they answer (`v.seed.name.needle`) and with whether a
    performer put them there. Then:

      * **an explicit performer wins** where it found anything, and where it
        found nothing the name falls back to its titles. That is one filter over
        the pool rather than a special case, so both routes are always present
        and the data says which survives, per name.

      * **the rest resolve each other.** `coherent` keeps the combination that
        keeps the most company, with the pinned ones in that reckoning as
        anchors: told that one song is Oasis, the others lean towards Oasis.

    The sort is the fallback rather than the decision: with one name, or names
    with nothing in common, nothing is connected and the order stands.
    """
    wanted = list(asked_for(names))
    if not wanted:
        return None
    pool = jb.concat(_by_performer(graph, wanted),
                     _by_title(graph, [title for title, _performer in wanted]))
    return (pool
            # a performer that found something settles its own name and no other
            .filter(pl.col(PINNED) | ~pl.col(PINNED).any().over("asked"))
            .sort(["closeness", "seen"], descending=True)
            .coherent(by=v.asked, through=reverse("contains"))
            .select("seed"))


def _by_title(graph, titles):
    """Every title's closest candidates, all of them in one query.

    `v.seed.name.needle` says which title each candidate answers, so what would
    be a loop over names is a column."""
    return (graph.nodes(seed="song")
            .filter(v.seed.name.like(titles, rule=Words(k=TIES)))
            .with_columns(asked=v.seed.name.needle.cast(pl.String),
                          closeness=v.seed.name.score,
                          seen=v.seed.contains.count(),
                          **{PINNED: pl.lit(False)})
            .select("seed", "asked", "closeness", "seen", PINNED))


def _by_performer(graph, wanted):
    """The candidates a performer allows, for the names that named one.

    The performer is not a filter over the titles that matched -- it narrows
    what is searched. Which matters: the exact-titled `Wonderwall`s are eight
    covers, and the Oasis recording, filed as `Wonderwall - Remastered`, is not
    among them at any pool size worth using. Asked of Oasis' songs instead, it
    is the only answer there is.

    One query for every pinned name: the performers are searched together and
    `v.artist.name.needle` says which one each is, so the title that goes with
    it is a join rather than an iteration."""
    pins = {performer: title for title, performer in wanted if performer}
    if not pins:
        return _no_candidates(graph)
    people = (graph.nodes(artist="artist")
              .filter(v.artist.name.like(list(pins), rule=Words(k=PERFORMERS)))
              .with_columns(who=v.artist.name.needle))
    if not len(people):
        return _no_candidates(graph)
    songs = (people.hop(seed=reverse("performed_by"))
             .attrs(seed="name")
             .with_columns(asked=pl.col("who").cast(pl.String)
                           .replace_strict(pins, default=None)))
    return (songs
            .filter(pl.col("seed.name").str.to_lowercase()
                    .str.contains(pl.col("asked").str.to_lowercase(), literal=True))
            .with_columns(closeness=pl.lit(1.0), seen=v.seed.contains.count(),
                          **{PINNED: pl.lit(True)})
            .select("seed", "asked", "closeness", "seen", PINNED))


def _no_candidates(graph):
    """An empty pool shaped like the others, so a concat of it is a concat and
    not a special case."""
    return (graph.nodes(seed="song").head(0)
            .with_columns(asked=pl.lit(None, dtype=pl.String),
                          closeness=pl.lit(0.0), seen=pl.lit(0, dtype=pl.Int64),
                          **{PINNED: pl.lit(False)})
            .select("seed", "asked", "closeness", "seen", PINNED))


def clustered(graph, model, seeds, concentration):
    """The seeds, each labelled with the part of the playlist it belongs to.

    A playlist is one thing when it is about one thing and several when it is
    not, and only whoever asked knows which. `concentration` says how finely to
    split it: at 0 the whole playlist is one field and the answers come from
    wherever in it they score best, at 1 every song is its own and the answers
    cover all of them. In between, k-means over the latent space.

    Splitting is what stops a broad playlist from being answered entirely out of
    its largest corner."""
    ids = seeds.ids("seed")
    parts = 1 + round(concentration * (len(ids) - 1))
    if parts <= 1:
        return seeds.with_columns(part=pl.lit(0, dtype=pl.Int32))
    space = model.embeddings(graph)[ids]
    labels, _ = kmeans2(space, min(parts, len(ids)), minit="++", seed=0)[::-1]
    return seeds.with_columns(part=pl.Series("part", labels, dtype=pl.Int32))


def candidates(graph, seeds, known):
    """Every song the factorization knows that the playlists holding your songs
    also hold, and how many of them do -- counted within the part of the
    playlist the seed belongs to.

    Two steps out and one back: seed -> the playlists containing it -> what else
    they contain. The playlists are named because the count is over them.

    Planned (`jb.optimize`): on the whole graph a handful of popular seeds reach
    tens of thousands of playlists and millions of rows. The walk runs a slice at
    a time, each slice keeps only the songs worth counting, and the count is
    folded across slices -- so the rows never exist all at once."""
    with jb.optimize():
        counted = (seeds
                   .hop(playlist=reverse("contains"))
                   .hop(rec="contains")
                   .filter(~v.rec.is_in(seeds), v.rec.is_in(known))
                   .group_by(v.rec, v.part).len("shared"))
    return counted.with_columns(shared=v.shared.cast(pl.Float64))


def rank(counted, seeds, model, k, parts, temperature):
    """Three signals, and the query says how much each counts.

    `shared` is the crowd's evidence, damped: six hundred of those playlists is
    a better answer than sixty, not ten times better, and a raw count buries the
    other two. `taste` is the model's. `gathered` is whether the song belongs
    anywhere at all -- a film score sits in five hundred playlists with nothing
    in common, and answering with it is answering with the average of
    everything.

    Then the best of each part, so a playlist about two things is answered about
    both, topped up from the best of anywhere when the parts cannot fill k."""
    scored = (counted
              .with_columns(taste=model.seeded(seeds).on("rec"),
                            gathered=Concentration(model, relation="contains").on("rec"))
              .with_columns(score=v.taste * v.shared.log1p() * v.gathered))
    each = -(-k // max(parts, 1))
    covered = (scored.top(each, by=v.score, over="part", temperature=temperature)
               .unique("rec"))
    if len(covered) < k:
        rest = (scored.filter(~v.rec.is_in(covered))
                .top(k - len(covered), by=v.score, temperature=temperature))
        covered = jb.concat(covered, rest).unique("rec")
    return covered.sort("score", descending=True).head(k)


def describe(frame, column):
    """A column of songs as "title -- performer", which is the only form in
    which a song is identifiable: titles are not unique and this dataset holds
    several masters of the same recording."""
    return (frame.hop(artist="performed_by").attrs(**{column: "name"}, artist="name")
            if len(frame) else frame.attrs(**{column: "name"}))


def extend(graph, model, known, songs, k, concentration=0.0, temperature=0.0):
    seeds = resolve(graph, songs)
    if seeds is None or not len(seeds):
        return [], []
    found = describe(seeds, "seed")
    named = [f"{row['seed.name']} -- {row['artist.name']}"
             for row in found.rows(named=True)]

    parts = clustered(graph, model, seeds, concentration)
    counted = (candidates(graph, parts, known)
               .attrs(rec="name")
               # another master of a song you gave me is not a suggestion
               .filter(~v.rec.name.is_in(found.pl["seed.name"].to_list())))
    if not len(counted):
        return named, []

    ranked = describe(rank(counted, seeds, model, k,
                           parts.pl["part"].n_unique(), temperature), "rec")
    return named, [
        {
            "song": row["rec.name"],
            "artist": row["artist.name"],
            "score": row["score"],
            "playlists": int(row["shared"]),
        }
        for row in ranked.rows(named=True)
    ]


def supported(graph):
    """The songs a factorization may learn from, and will then know anything
    about.

    A song in one playlist is not evidence a latent space can hold: with 680 000
    songs whose median support is a single playlist, factorizing everything is
    factorizing noise, and the result ranks soundtrack themes above pop."""
    return (graph.nodes(song="song")
            .with_columns(seen=v.song.contains.count())
            .filter(v.seen >= SUPPORT).select("song"))


def interactions(graph, known):
    """The playlist edges into those songs: what the factorization is fitted on.

    Only a fit needs them, and on the whole graph they are 66 million rows, so
    a start that loads a stored factorization never builds them."""
    return graph.edges("contains").filter(v.target.is_in(known))


# --- startup -----------------------------------------------------------------

def load_graph():
    """The playlists and what they contain, plus who performed what.

    `spotify.contains` carries a score per edge -- a song's place in the
    playlist -- which nothing here reads: what a position means is a question
    this service does not ask.

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


def fit(graph):
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


def _stored(graph, expected):
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


def warm(graph, model, known):
    """One request before the first caller's.

    The first request builds what every later one reads -- the word index over
    song titles, each relation's bounds in the store, `Concentration`'s
    products over the whole graph -- which on the whole graph is half a minute.
    Asked here, with the title of whichever song comes first, it is paid while
    the service is starting rather than by whoever asks first."""
    title = graph.nodes(song="song").head(1).attrs(song="name").pl["song.name"][0]
    extend(graph, model, known, [title], 5)



@contextmanager
def timed(step):
    """Say how long a startup step took: on the whole graph startup is minutes,
    and a container log that is silent for minutes looks like a hang."""
    start = time.perf_counter()
    yield
    log.info("spotify: %s ready in %.1f s", step, time.perf_counter() - start)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with timed("graph"):
        app.state.graph = load_graph()
    with timed("fit"):
        app.state.model, app.state.known = fit(app.state.graph)
    with timed("warm"):
        warm(app.state.graph, app.state.model, app.state.known)
    yield
    app.state.graph = app.state.model = app.state.known = None


app = FastAPI(title="Jerboas playlist continuation", lifespan=lifespan)


class ExtendRequest(BaseModel):
    songs: list[str]
    k: int = Field(default=5, ge=1, le=50)
    # how finely to read the playlist: 0 treats it as one field and answers from
    # wherever in it the scores are best, 1 answers about every song in it
    concentration: float = Field(default=0.0, ge=0.0, le=1.0)
    # 0 answers the same thing every time; 1 lets the noise be as large as the
    # spread of the scores it is disturbing
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)


class Suggestion(BaseModel):
    song: str
    artist: str | None = None
    score: float
    # how many of the playlists holding your songs hold this one too
    playlists: int


class ExtendResponse(BaseModel):
    # what the fuzzy match landed on, so "Bad Romance" turning into something
    # else is something you can see
    songs: list[str]
    suggestions: list[Suggestion]


@app.post("/extend", response_model=ExtendResponse)
def post_extend(body: ExtendRequest):
    if not body.songs:
        raise HTTPException(status_code=422, detail="name at least one song")
    named, suggestions = extend(app.state.graph, app.state.model, app.state.known,
                                body.songs, body.k, body.concentration,
                                body.temperature)
    if not named:
        raise HTTPException(status_code=404, detail="no song matched")
    return ExtendResponse(songs=named, suggestions=suggestions)


@app.get("/health")
def get_health():
    graph = app.state.graph
    sized = lambda kind: graph.block(kind)[1] - graph.block(kind)[0]
    return {"status": "ok", "playlists": sized("playlist"), "songs": sized("song"),
            "known": len(app.state.known)}
