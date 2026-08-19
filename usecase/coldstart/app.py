"""Cold-start movie recommender (FastAPI service), on MovieLens.

    ./run.sh coldstart                                   # in a container
    uvicorn app:app --app-dir usecase/coldstart          # or straight from here

    curl -X POST localhost:8000/recommend -H 'content-type: application/json' \
         -d '{"people": ["Quentin Tarantino", "Bruce Willis"], "genres": ["Crime"], "k": 10}'

Paths are relative to the repository root, which is both the working directory
of the container and where you would run uvicorn by hand.

There is no user node: the caller names people, genres or films they like, and
the model expands from those to reach candidate movies.

Two things are prepared once at startup and reused by every request. The graph,
because MovieLens does not change at runtime. And the TransD embedding, because
fitting it takes seconds and rebinding a checkpoint is linear in the graph -- so
the checkpoint is loaded once and only the seeds vary per request.
"""

import os
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import jerboas as jb
from jerboas import PageRank, TransD, concat, train, v

DATA_DIR = "./data/movielens"
CHECKPOINT = "./checkpoints/ml.transd.npz"
EPOCHS = 15

# which column each type reads as a name. Declared with the data rather than per
# query: the graph does not guess which of its columns a human reads, and the
# answer is a fact about the dataset.
READABLE = {"movie": "title", "person": "name", "genre": "name"}


def liked(graph):
    """What counts as liking a film, for the model.

    A 1-star rating is an interaction and belongs in the graph; it is not
    evidence of an affinity and does not belong in a model of one. Being a frame
    handed to the fit rather than an argument to the loader is the point: the
    same graph still answers "who rated this at all?"."""
    return graph.edges().filter((v.relation != "has_interact") | (v.score >= 3))


def seeds(graph, wanted):
    """The nodes the caller named, fuzzy-matched per type.

    Per type rather than in one untyped query on purpose: the request already
    says which list is people and which is genres, and `like` admits its k best
    values in whatever it is allowed to look at -- so asking across all of them
    would let "Quentin Tarantino" also pull in the nearest film title."""
    frames = []
    for type_, names in wanted.items():
        asked = [name.strip() for name in (names or []) if name and name.strip()]
        if asked:
            frames.append(graph.nodes(seed=type_)
                          .filter(v.seed.label.like(asked))
                          .select("seed"))
    return concat(*frames) if frames else None


def expanded(graph, found):
    """Every film within two hops of a seed, and how far away it was.

    A candidate reaches a liked attribute directly (one step) or through a bridge
    node (two), so it is two frames and a concat. The bridge's middle step is not
    named, which is what folds it: two attributes leading to the same film are
    one answer, and that is why a two-hop bridge over 15 000 nodes stays small.

    An empty step is any relation in either direction, so the bridge closes
    whichever way the edges happen to be stored."""
    seeds = graph.nodes(seed=found)
    direct = seeds.hop(rec=()).with_columns(hops=pl.lit(1, dtype=pl.Int32))
    bridge = seeds.hop((), rec=()).with_columns(hops=pl.lit(2, dtype=pl.Int32))
    return concat(direct, bridge).filter(v.rec.type == "movie")


def rank_films(graph, model, found, exclude, k):
    """The k films closest to a mixed bag of seeds.

    No single relation joins the seeds to a film -- a person does it through
    directed_by/acted_in read backwards, a genre through has_genre -- so the
    model is asked for the most plausible edge of any kind, which is what a
    heterogeneous seed set needs.

    The two signals are combined in the open: each is a column, each is put on
    its own [0, 1] scale, and how much either counts is written down rather than
    averaged behind the caller's back. `unique` after the sort keeps the best
    route to each film, so one film cannot take three of the k places."""
    return (expanded(graph, found)
            .filter(~v.rec.is_in(exclude))
            .with_columns(kg=model.seeded(found).on("rec").norm(),
                          walk=PageRank(to=found, weighted=True).on("rec").norm())
            .with_columns(score=0.6 * v.kg + 0.4 * v.walk)
            .sort("score", descending=True)
            .unique("rec")
            .head(k)
            .labels("rec", "seed")
            .with_columns(**{"rec.via": v.rec.via}))


def explain(row):
    """How a film connects to what the caller named.

    The walk is not a string to parse or a Path object to unpack -- it is the
    row: `seed` is what was liked, `hops` is how far it was, and `rec.rel` names
    the edge that arrived."""
    seed = row["seed.label"]
    if row["hops"] > 1:
        return f"shares something with {seed}"
    relation = str(row["rec.via"] or "").lstrip("~").replace("_", " ")
    return f"{relation} {seed}"


def recommend(graph, model, people, genres, titles, k):
    found = seeds(graph, {"movie": titles, "person": people, "genre": genres})
    if found is None or not len(found):
        return []
    watched = [key for key in found.keys("seed") if key.type == "movie"]
    ranked = rank_films(graph, model, found, watched, k)
    return [{"title": row["rec.label"], "score": row["score"], "why": explain(row)}
            for row in ranked.rows(named=True)]


# --- startup -----------------------------------------------------------------

def load_graph():
    # every interaction is loaded, rating and all: `ml.has_interact` carries a
    # score per edge, and what counts as a good enough rating is decided by
    # whoever asks. The walk reads it as a weight (PageRank(weighted=True)) and
    # the model is trained without the ratings it should not learn from (see
    # liked(), above) -- neither decision is baked into the graph.
    return jb.Graph(
        kg=f"{DATA_DIR}/ml.kg",
        edges=[f"{DATA_DIR}/ml.has_interact"],
        attrs=[
            f"{DATA_DIR}/ml.movie",
            f"{DATA_DIR}/ml.genre",
            f"{DATA_DIR}/ml.year",
            f"{DATA_DIR}/ml.user",
            f"{DATA_DIR}/ml.person",
        ],
        readable=READABLE,
    )


def load_or_fit(graph, path=CHECKPOINT, epochs=EPOCHS):
    """The embedding, fitted on first boot and reused after.

    Training is a batch job, not something a query can absorb, so it happens here
    rather than inside a request. Serving the checkpoint rather than the model
    just fitted is deliberate: what answers queries is then exactly what is on
    disk, and a restart cannot silently serve something else.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        train(TransD(factors=64), graph, epochs=epochs, device=device(),
              where=liked(graph)).save(path)
    return TransD.load(path, graph)


def device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = load_graph()
    app.state.model = load_or_fit(app.state.graph)
    yield
    app.state.graph = app.state.model = None


app = FastAPI(title="Jerboas cold-start recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    # null and an omitted field mean the same thing: nothing to seed from.
    # Blank entries inside a list are stripped in resolve().
    people: list[str] | None = None
    genres: list[str] | None = None
    titles: list[str] | None = None
    k: int = Field(default=10, ge=1, le=100)


class Recommendation(BaseModel):
    title: str
    score: float
    why: str | None = None


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]


@app.post("/recommend", response_model=RecommendResponse)
def post_recommend(body: RecommendRequest):
    if not (body.people or body.genres or body.titles):
        raise HTTPException(status_code=422, detail="provide at least one of people/genres/titles")
    results = recommend(app.state.graph, app.state.model,
                        body.people, body.genres, body.titles, body.k)
    return RecommendResponse(recommendations=results)


@app.get("/health")
def get_health():
    return {"status": "ok", "trained_at": app.state.model.meta.get("trained_at")}
