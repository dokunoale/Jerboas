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
candidate. What ranks them is the sum of the tag weights they carry -- one
shared tag is a coincidence, twenty is a taste -- which is a question about the
matches rather than about the graph, and therefore a group_by rather than a
strategy. There is no ranking object in this file at all: it is two hops, a
filter and an aggregate.
"""

from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import jerboas as jb
from jerboas import v

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

    Worth the four lines. `like` treats a title that contains the needle as a
    perfect match, so "The Matrix" untranslated lands on *The Matrix Revisited*
    -- a documentary -- while the film itself is left to come back as its own
    top suggestion."""
    article, _, rest = title.partition(" ")
    if rest and article.lower() in ARTICLES:
        return f"{rest}, {article}"
    return title


def resolve(graph, titles):
    """Watchlist entries as a frame of films, one best match each.

    A watchlist is typed by a person, so it is matched the way a search box
    matches: `like` admits the closest stored title rather than a region around
    it, and what it landed on comes back in the response so a wrong guess is
    visible rather than silent."""
    wanted = [filed_as(title.strip()) for title in titles if title and title.strip()]
    if not wanted:
        return None
    return graph.nodes(seed="movie").like(v.seed.title, wanted, k=1).select("seed")


def shared(graph, seeds, strength):
    """Every (watchlist film, tag, candidate) the genome connects, as a frame.

    One walk, written once: the ranking groups it and the explanation reads it,
    so the two cannot drift apart -- where they used to be two queries repeating
    the same pattern in the hope of agreeing.

    The tag weight is normalized on the way out of the hop, because a sum of
    unbounded scores is arithmetic on a scale nobody chose."""
    return (graph.nodes(seed=seeds)
            .hop("has_tag", to="tag", as_="strong", norm=True)
            .filter(v.strong.score >= strength)
            .hop("has_tag", to="rec", reverse=True, as_="carried", norm=True)
            .filter(v.carried.score >= strength)
            .filter(~v.rec.is_in(seeds)))                 # already on the list


def suggest(graph, watchlist, k, strength):
    seeds = resolve(graph, watchlist)
    if seeds is None or not len(seeds):
        return [], []
    matched = sorted(seeds.attrs(seed="title").pl["seed.title"].to_list())

    matches = shared(graph, seeds, strength)
    if not len(matches):
        return matched, []

    ranked = (matches
              .group_by(v.rec)
              .agg(score=v.carried.score.sum(),
                   # the evidence, strongest tag first: an aggregate collapses
                   # the rows it was computed from, so what explains a film is
                   # gathered in the same breath as what ranks it.
                   #
                   # Deduplicated, unlike the score: a tag shared with two of
                   # your films is twice the evidence, and the same word twice
                   # in a list of reasons is a bug.
                   shared=v.tag.expr.sort_by(pl.col("carried.score"), descending=True)
                                    .unique(maintain_order=True).head(5))
              .top(k)
              .attrs(rec=["title", "year"]))

    names = _tag_names(graph, ranked)
    return matched, [
        {
            "title": row["rec.title"],
            "year": row["rec.year"],
            "score": row["score"],
            "shared": [names[tag] for tag in row["shared"]],
        }
        for row in ranked.rows(named=True)
    ]


def _tag_names(graph, ranked):
    """The tag ids the aggregate kept, as names. A gather over the few that
    survived, rather than a column carried through the whole walk."""
    ids = {tag for row in ranked.rows(named=True) for tag in row["shared"]}
    if not ids:
        return {}
    frame = graph.nodes(tag=sorted(ids)).attrs(tag="name")
    return dict(zip(frame.pl["tag"].to_list(), frame.pl["tag.name"].to_list()))


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
        readable={"movie": "title", "tag": "name", "genre": "name"},
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
