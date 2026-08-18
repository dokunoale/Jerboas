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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import jerboas as jb
from jerboas import Node, Edge, Path, Score, Like, PageRank, TransD, train

DATA_DIR = "./data/movielens"
CHECKPOINT = "./checkpoints/ml.transd.npz"
EPOCHS = 15

# what counts as liking a film, for the model. A 1-star rating is an interaction
# and belongs in the graph; it is not evidence of an affinity and does not belong
# in a model of one. Being a training argument rather than a loader argument is
# the point: the same graph still answers "who rated this at all?".
LIKED = [Edge("has_interact", score=(3, None))]


def cold_start_pattern(path, rec, seed):
    # candidate movie -> a liked attribute (1 hop) OR -> attribute -> a liked movie (2 hops).
    # shared by the ranking and explain queries so they can't silently drift apart.
    # Edge() is undirected, so the bridge closes whichever way the edges are stored.
    return (path == [rec, Edge(), seed]) | (path == [rec, Edge(), Node(), Edge(), seed])


# which column each type reads as a name. The graph names its own columns and
# does not guess at which one a human reads, so the question says it.
READABLE = {"movie": "title", "person": "name", "genre": "name"}


def readable(key):
    """A key as a person would name it."""
    return key.attrs.get(READABLE.get(key.type, "name"))


def explain(walk):
    # walk is (rec, rel, [mid, rel,] seed): how rec connects to your likes
    seed = walk[-1]
    if seed.type == "movie" and len(walk) > 3:        # a liked film, via a bridge node
        return f"similar to {readable(seed)} (shared {walk[2].type})"
    return f"{walk[-2].name.replace('_', ' ')} {readable(seed)}"


def seeds(g, wanted):
    """The nodes the caller named, fuzzy-matched per type.

    Per type rather than in one untyped query on purpose: the request already
    says which list is people and which is genres, and `Like` admits its k best
    values in every type it is allowed to look at -- so asking across all of
    them would let "Quentin Tarantino" also pull in the nearest film title."""
    found = set()
    for type_, names in wanted.items():
        asked = [name.strip() for name in (names or []) if name and name.strip()]
        if asked:
            node = Node(type_).alias(readable=READABLE[type_])
            found |= set(g.select(node).where(Like(node.readable.is_in(asked))))
    return found


def rank_films(g, model, seed_keys, exclude, k):
    """The k films closest to a mixed bag of seeds.

    No single relation joins the seeds to a film -- a person does it through
    directed_by/acted_in read backwards, a genre through has_genre -- so the
    model is asked for the most plausible edge of any kind, which is what a
    heterogeneous seed set needs."""
    rec, seed, path = Node("movie"), Node(), Path()
    return g.select(rec, Score()).where(
        cold_start_pattern(path, rec, seed),
        seed.is_in(seed_keys),
        ~rec.is_in(exclude),
    ).rank(model.seeded(seed_keys),                   # learned KG plausibility
           PageRank(to=seed_keys, weighted=True)      # personalized walk: proximity
           ).top(k)


def why(g, films, seed_keys):
    """One walk per film, to explain it with.

    A second search rather than selecting the Path alongside the ranking: a Path
    is one row per walk, and a film reached three ways would take three of the k
    places. Restricted to the films that survived, so it is small."""
    rec, seed, path = Node("movie"), Node(), Path()
    walks = {}
    for walk in g.select(path).where(cold_start_pattern(path, rec, seed),
                                     rec.is_in(films), seed.is_in(seed_keys)):
        walks.setdefault(walk[0], walk)
    return walks


def recommend(g, model, people, genres, titles, k):
    found = seeds(g, {"movie": titles, "person": people, "genre": genres})
    if not found:
        return []
    ranked = rank_films(g, model, found, {key for key in found if key.type == "movie"}, k)
    if not ranked:
        return []
    walks = why(g, {film for film, _ in ranked}, found)
    return [{"title": readable(film),
             "score": score,
             "why": explain(walks[film]) if film in walks else None}
            for film, score in ranked]


# --- startup -----------------------------------------------------------------

def load_graph():
    # every interaction is loaded, rating and all: `ml.has_interact` carries a
    # score per edge, and what counts as a good enough rating is decided by
    # whoever asks. The walk reads it as a weight (PageRank(weighted=True)) and
    # the model is trained without the ratings it should not learn from (see
    # LIKED, below) -- neither decision is baked into the graph.
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
              where=LIKED).save(path)
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
