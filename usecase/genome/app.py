"""Watchlist recommender (FastAPI service), on the MovieLens tag genome.

    ./run.sh genome                                   # in a container
    uvicorn app:app --app-dir usecase/genome          # or straight from here

    curl -X POST localhost:8000/suggest -H 'content-type: application/json' \
         -d '{"watchlist": ["Blade Runner", "The Matrix", "Aliens"], "k": 10}'

You bring a watchlist; the service answers with films that belong beside it. The
tag genome is what makes that possible: it scores every film against every one
of its 1128 tags, so "coherent" has a measurement behind it rather than a genre
label.

Nothing is trained here. Where the cold-start service fits an embedding because
the affinity it needs is not in the data, this one reads a number that already
is -- which is the whole difference between the two, and why this file is half
the size.

Two hops carry the query: watchlist -> a tag both films are strong on -> a
candidate. What ranks them is `Sum` over the tags they carry -- one shared tag is
a coincidence, twenty is a taste -- which is a question about the matches rather
than about the graph, and therefore an aggregate rather than a strategy.
"""

from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import jerboas as jb
from jerboas import Node, Edge, Path, Score, Like, Sum

DATA_DIR = "./data/genome"

# how strong a tag must be to count as one of a film's own. The genome is a
# dense matrix -- every film scores against every tag -- so this is where a tag
# stops being noise, and it is a request parameter rather than a constant
# because the honest answer depends on how broad an answer you want.
STRENGTH = 0.9

# leading articles MovieLens files at the end of a title (see filed_as)
ARTICLES = ("the", "a", "an", "le", "la", "les", "il", "lo", "der", "die", "das")


def filed_as(title):
    """A title the way MovieLens files it: the leading article goes to the end,
    so *The Matrix* is stored as *Matrix, The*.

    Worth the four lines. `Like` treats a title that contains the needle as a
    perfect match, so "The Matrix" untranslated lands on *The Matrix Revisited*
    -- a documentary -- while the film itself is left to come back as its own
    top suggestion."""
    article, _, rest = title.partition(" ")
    if rest and article.lower() in ARTICLES:
        return f"{rest}, {article}"
    return title


def resolve(graph, titles):
    """Watchlist entries as film keys, one best match each.

    A watchlist is typed by a person, so it is matched the way a search box
    matches: `Like` admits the closest stored title rather than a region around
    it, and what it landed on comes back in the response so a wrong guess is
    visible rather than silent."""
    wanted = [filed_as(title.strip()) for title in titles if title and title.strip()]
    if not wanted:
        return set()
    movie = Node("movie")
    return set(graph.select(movie).where(Like(movie.title.is_in(wanted), k=1)))


def suggest(graph, watchlist, k, strength):
    seeds = resolve(graph, watchlist)
    if not seeds:
        return [], []

    seed, tag, rec, path = Node("movie"), Node("tag"), Node("movie"), Path()
    # one marker walked forwards out of the watchlist and backwards into a
    # candidate. `carried` is named rather than written twice because `.inverse`
    # makes a new marker each time, and the ranking has to mean *that* edge
    strong = Edge("has_tag", score=(strength, None))
    carried = strong.inverse

    ranked = graph.select(rec, Score()).where(
        path == [seed, strong, tag, carried, rec],
        seed.is_in(seeds),
        ~rec.is_in(seeds),                       # already on the list
    ).rank(Sum(carried.score)).top(k)

    films = [film for film, _score in ranked]
    if not films:
        return sorted(str(s.title) for s in seeds), []

    shared = _shared_tags(graph, seeds, films, strength)
    return sorted(str(s.title) for s in seeds), [
        {
            "title": film.attrs["title"],
            "year": film.attrs.get("year"),
            "score": score,
            "shared": shared[film],
        }
        for film, score in ranked
    ]


def _shared_tags(graph, seeds, films, strength, limit=5):
    """Why each of the winners won: the tags it shares with the watchlist,
    strongest first.

    A second query rather than a column of the first. The ranking collapses to
    one row per film -- that is what an aggregate is for -- so the evidence has
    to be asked for separately, and asking it of the k winners is cheaper than
    carrying it for the thousands that lost. It runs the walk from the films
    this time, which is the same pattern read from the other end."""
    film, tag, seed, path = Node("movie"), Node("tag"), Node("movie"), Path()
    strong = Edge("has_tag", score=(strength, None))

    weighed = defaultdict(dict)
    for walk in graph.select(path).where(
        path == [film, strong, tag, strong.inverse, seed],
        film.is_in(films),
        seed.is_in(seeds),
    ):
        found, name = walk[0], walk[2]
        weighed[found][str(name.attrs["name"])] = graph.weight_of(
            int(found), int(name), graph.relation_code("has_tag"), normalized=True)

    return {found: sorted(tags, key=tags.get, reverse=True)[:limit]
            for found, tags in weighed.items()}


# --- startup -----------------------------------------------------------------

def load_graph():
    """Only the files this service reads: the tag genome and what a film is.

    The ratings are in `data/genome` too and are left out on purpose -- a
    million edges nothing here walks. A use case chooses its graph."""
    return jb.Graph(
        kg=f"{DATA_DIR}/genome.kg",
        edges=[f"{DATA_DIR}/genome.has_tag"],
        attrs=[
            f"{DATA_DIR}/genome.movie",
            f"{DATA_DIR}/genome.tag",
            f"{DATA_DIR}/genome.genre",
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = load_graph()
    yield
    app.state.graph = None


app = FastAPI(title="Jerboas watchlist recommender", lifespan=lifespan)


class SuggestRequest(BaseModel):
    watchlist: list[str]
    k: int = Field(default=10, ge=1, le=100)
    # the floor is the one the dataset was built with: below it the graph holds
    # no edge to walk. Raising it narrows the answer and speeds it up
    strength: float = Field(default=STRENGTH, ge=0.3, le=1.0)


class Suggestion(BaseModel):
    title: str
    year: int | None = None
    score: float
    shared: list[str]


class SuggestResponse(BaseModel):
    # what the fuzzy match actually landed on, so "Alien" turning into "Aliens"
    # is something you can see
    watchlist: list[str]
    suggestions: list[Suggestion]


@app.post("/suggest", response_model=SuggestResponse)
def post_suggest(body: SuggestRequest):
    if not body.watchlist:
        raise HTTPException(status_code=422, detail="the watchlist is empty")
    matched, suggestions = suggest(app.state.graph, body.watchlist, body.k, body.strength)
    if not matched:
        raise HTTPException(status_code=404, detail="no film matched the watchlist")
    return SuggestResponse(watchlist=matched, suggestions=suggestions)


@app.get("/health")
def get_health():
    graph = app.state.graph
    return {"status": "ok", "films": graph.block("movie")[1] - graph.block("movie")[0],
            "tags": graph.block("tag")[1] - graph.block("tag")[0]}
