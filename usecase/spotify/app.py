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


def resolve(graph, names):
    """The songs you named, one best match each -- and among equally good
    matches, the one the crowd means.

    A name typed by a person is a search box, not a key: `like` admits the
    closest stored titles rather than the equal one, and what it landed on comes
    back in the answer so a wrong guess is visible rather than silent.

    One name at a time, and several matches each, because titles are not unique
    and closeness cannot break the tie: `Wonderwall` is Oasis and also Ryan
    Adams covering Oasis, both spelled identically. Which one somebody typing it
    meant is a claim about this dataset -- the one in more playlists -- so the
    use case makes it rather than the library guessing.

    That claim is the crude version of the right one, and the right one is a
    graph question rather than a popularity one. **Not built** (see below), but
    written down because it is the interesting part:

    A name resolved on its own has only its own popularity to go on. A *set* of
    names has more: the songs somebody names together tend to sit in the same
    playlists, so the assignment to prefer is the one whose chosen songs are
    most connected to each other. `Wonderwall` beside `Champagne Supernova` is
    Oasis; `Wonderwall` beside `Come Pick Me Up` is Ryan Adams. Neither is more
    popular than the other in the abstract -- what decides is the company.

    Concretely: keep every candidate rather than one, tagged with the name it
    answers; ask the graph for the pairwise connection between candidates, which
    is one two-hop query and returns the whole matrix at once; then choose one
    candidate per name to maximise the total. The choice is `k**n` by brute
    force and a few passes of coordinate ascent -- best candidate for each name
    given the others, repeated -- in practice.

    Until then there is a cheaper escape, also unbuilt: let a caller pin a title
    by writing the performer after a tab, `"Wonderwall\tOasis"`, which turns a
    guess into a constraint. The two compose rather than compete -- an explicit
    performer pins one name and the coherence resolves the rest."""
    found = []
    for name in (one.strip() for one in names):
        if not name:
            continue
        matches = (graph.nodes(seed="song")
                   .filter(v.seed.name.like(name, k=TIES))
                   .with_columns(closeness=v.seed.name.score,
                                 seen=v.seed.contains.count())
                   .sort(["closeness", "seen"], descending=True).head(1))
        if len(matches):
            found.append(matches.select("seed"))
    return jb.concat(*found) if found else None


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
