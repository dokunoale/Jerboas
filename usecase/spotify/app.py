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

import os
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import jerboas as jb
from jerboas import DiffusedMatrixFactorization, reverse, v

DATA_DIR = os.environ.get("SPOTIFY_DIR", "./data/spotify/graph-100k")

# What a factorization may learn from. A song in one playlist is not evidence of
# anything a latent space can hold: with 680 000 songs whose median support is a
# single playlist, factorizing everything is factorizing mostly noise, and the
# result ranks soundtrack themes above pop. Restricted to the songs the crowd has
# actually placed more than once, the same model becomes a usable re-ranker.
#
# It is a claim about the data rather than a tuning knob, which is why it is
# written here and not hidden in the strategy.
SUPPORT = 20

# how many equally-titled songs to weigh against each other before picking one
TIES = 8

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
    """The songs you named, one best match each.

    A name typed by a person is a search box, not a key: `like` admits the
    closest stored titles rather than the equal one, and what it landed on comes
    back in the answer so a wrong guess is visible rather than silent.

    All the names in one query, because `v.seed.name.needle` says which of them
    each row is an answer to: the grouping the tie-break needs is a column
    rather than a loop.

    Three things decide, in this order. A **performer** given after a tab is a
    constraint and settles it outright. Failing that, **closeness**, and then
    **how many playlists hold it** -- which is a claim about this dataset rather
    than about the name, and the crude version of the right answer.

    The right answer is `Frame.coherent`, which picks the combination of
    candidates most connected to each other, and this does not use it yet.
    Measured against real playlists it loses to popularity and loses by more as
    the playlist grows -- 66.7% against 75.0% at three songs, 57.5% against
    97.5% at eight. The cause is upstream of the choice: `like` scores any
    containment 1.0, so `Toxicity` ties with `Toxic` and the pool fills with
    near-misses that a popularity-free measure is happy to prefer. What is
    missing is a closeness that ranks a containing title by how tightly it
    contains -- which `fuzzy.closest` computes, taking the shortest match first,
    and throws away."""
    wanted = list(asked_for(names))
    if not wanted:
        return None
    pinned = [(title, performer) for title, performer in wanted if performer]
    loose = [title for title, performer in wanted if not performer]

    found = [_by_performer(graph, title, performer) for title, performer in pinned]
    found = [one for one in found if one is not None]
    loose += [title for (title, _p), one in zip(pinned, found + [None] * len(pinned))
              if one is None]
    if loose:
        found.append(_by_title(graph, loose))
    return jb.concat(*found) if found else None


def _by_title(graph, titles):
    """The best match for each title, all of them in one query.

    `v.seed.name.needle` says which title each candidate answers, so the choice
    is a grouping rather than a loop over names -- and the choice itself is
    `coherent`: the combination of candidates that keep the most company with
    each other, two songs keeping company when a playlist holds both.

    Measured by handing back the titles of real playlists and counting how many
    resolve to the songs those playlists actually held, it beats taking the most
    played candidate by a distance -- 100% against 66.7% at three titles, 98.4%
    against 81.2% at sixteen -- because a set of names says something no name
    says alone.

    The sort before it is the fallback rather than the decision: with one name,
    or names with nothing in common, nothing is connected and the order stands.
    """
    return (graph.nodes(seed="song")
            .filter(v.seed.name.like(titles, k=TIES))
            .with_columns(asked=v.seed.name.needle,
                          closeness=v.seed.name.score,
                          seen=v.seed.contains.count())
            .sort(["closeness", "seen"], descending=True)
            .coherent(by=v.asked, through=reverse("contains"))
            .select("seed"))


def _by_performer(graph, title, performer):
    """A title among one performer's songs.

    The performer is not a filter over the titles that matched -- it narrows
    what is searched. Which matters: the exact-titled `Wonderwall`s are eight
    covers, and the Oasis recording, filed as `Wonderwall - Remastered`, is not
    among them at any pool size worth using. Asked of Oasis' songs instead, it
    is the only answer there is.

    None when the performer or the title finds nothing, so the caller can fall
    back to the title alone rather than be answered with silence."""
    people = graph.nodes(artist="artist").filter(v.artist.name.like(performer, k=4))
    if not len(people):
        return None
    songs = (graph.nodes(seed=people).hop(seed_song=reverse("performed_by"))
             .select("seed_song").rename({"seed_song": "seed"}))
    if not len(songs):
        return None
    matched = (songs.filter(v.seed.name.like(title, k=TIES))
               .with_columns(closeness=v.seed.name.score,
                             seen=v.seed.contains.count())
               .sort(["closeness", "seen"], descending=True).head(1).select("seed"))
    return matched if len(matched) else None


def candidates(graph, seeds):
    """Every song the playlists holding your songs also hold, and how often.

    Two steps out and one back: seed song -> the playlists containing it -> what
    else they contain. The playlists are named because the count is over them;
    nothing else about them is wanted."""
    return (graph.nodes(seed=seeds)
            .hop(playlist=reverse("contains"))
            .hop(rec="contains")
            .filter(~v.rec.is_in(seeds))
            .group_by(v.rec).agg(shared=pl.len().cast(pl.Float64)))


def rank(counted, seeds, model, k):
    """The crowd's evidence, shaped by the model's taste.

    `log1p` on the count is the whole of the shaping: a song in six hundred of
    those playlists is a better answer than one in sixty, but not ten times
    better, and multiplying by a raw count buries the model entirely. Both
    signals stay their own column, so an answer can say which one carried it."""
    return (counted
            .with_columns(taste=model.seeded(seeds).on("rec"))
            .with_columns(score=v.taste * v.shared.log1p())
            .top(k))


def supported(graph):
    """The interactions a factorization may learn from, and the songs it will
    then know anything about."""
    songs = (graph.nodes(song="song")
             .with_columns(seen=v.song.contains.count())
             .filter(v.seen >= SUPPORT).select("song"))
    return graph.edges("contains").filter(v.target.is_in(songs)), songs


def describe(frame, column):
    """A column of songs as "title -- performer", which is the only form in
    which a song is identifiable: titles are not unique and this dataset holds
    several masters of the same recording."""
    return (frame.hop(artist="performed_by").attrs(**{column: "name"}, artist="name")
            if len(frame) else frame.attrs(**{column: "name"}))


def extend(graph, model, known, songs, k):
    seeds = resolve(graph, songs)
    if seeds is None or not len(seeds):
        return [], []
    # what the match landed on, performer included: "Bohemian Rhapsody" is Queen
    # and also Panic! At The Disco covering Queen, and which one the crowd means
    # is a fact about this dataset rather than about the name
    found = describe(seeds, "seed")
    named = [f"{row['seed.name']} -- {row['artist.name']}"
             for row in found.rows(named=True)]
    titles = found.pl["seed.name"].to_list()

    counted = (candidates(graph, seeds).filter(v.rec.is_in(known))
               .attrs(rec="name")
               # another master of a song you gave me is not a suggestion
               .filter(~v.rec.name.is_in(titles)))
    if not len(counted):
        return named, []

    ranked = describe(rank(counted, seeds, model, k), "rec")
    return named, [
        {
            "song": row["rec.name"],
            "artist": row["artist.name"],
            "score": row["score"],
            "playlists": int(row["shared"]),
        }
        for row in ranked.rows(named=True)
    ]


# --- startup -----------------------------------------------------------------

def load_graph():
    """The playlists and what they contain, plus who performed what.

    `spotify.contains` carries a score per edge -- a song's place in the
    playlist -- which nothing here reads: what a position means is a question
    this service does not ask."""
    return jb.Graph(
        kg=f"{DATA_DIR}/spotify.kg",
        edges=[f"{DATA_DIR}/spotify.contains"],
        attrs=[f"{DATA_DIR}/spotify.{one}"
               for one in ("song", "artist", "album", "playlist")],
        readable=READABLE,
    )


def fit(graph):
    """The factorization, fitted once at startup on the supported subgraph.

    Seconds rather than milliseconds, so it belongs here and not in a request --
    and unlike an embedding there is no checkpoint to load it from, so a restart
    refits it."""
    kept, known = supported(graph)
    model = DiffusedMatrixFactorization(
        factors=FACTORS, iterations=ITERATIONS, where=kept,
        item_type="song", user_type="playlist", relation="contains")
    model.fit(graph)
    return model, known


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = load_graph()
    app.state.model, app.state.known = fit(app.state.graph)
    yield
    app.state.graph = app.state.model = app.state.known = None


app = FastAPI(title="Jerboas playlist continuation", lifespan=lifespan)


class ExtendRequest(BaseModel):
    songs: list[str]
    k: int = Field(default=5, ge=1, le=50)


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
                                body.songs, body.k)
    if not named:
        raise HTTPException(status_code=404, detail="no song matched")
    return ExtendResponse(songs=named, suggestions=suggestions)


@app.get("/health")
def get_health():
    graph = app.state.graph
    return {"status": "ok",
            "playlists": graph.block("playlist")[1] - graph.block("playlist")[0],
            "songs": graph.block("song")[1] - graph.block("song")[0],
            "known": len(app.state.known)}
