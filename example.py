"""Cold-start movie recommender (FastAPI service).

    pip install -e '.[api,torch]'
    uvicorn example:app --host 0.0.0.0 --port 8000

    curl -X POST localhost:8000/recommend -H 'content-type: application/json' \
         -d '{"people": ["Quentin Tarantino", "Bruce Willis"], "genres": ["Crime"], "k": 10}'

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


def cold_start_pattern(path, rec, seed):
    # candidate movie -> a liked attribute (1 hop) OR -> attribute -> a liked movie (2 hops).
    # shared by the ranking and explain queries so they can't silently drift apart.
    # Edge() is undirected, so the bridge closes whichever way the edges are stored.
    return (path == [rec, Edge(), seed]) | (path == [rec, Edge(), Node(), Edge(), seed])


def explain(path):
    # path is (rec, rel, [mid, rel,] seed): describe how rec connects to your likes
    seed = path[-1]
    if seed.type == "movie" and len(path) > 3:        # a liked movie, via a bridge node
        return f"similar to {seed.label} (shared {path[2].type})"
    return f"{path[-2].name.replace('_', ' ')} {seed.label}"


def resolve(g, type_, names):
    # seed resolution: fuzzy-match against the type's label column, whichever it
    # happens to be -- so this knows nothing about types. Blank entries are the
    # caller's, not the graph's, so they are dropped here rather than in Like.
    wanted = [name.strip() for name in (names or []) if name and name.strip()]
    if not wanted:
        return set()
    node = Node(type_)
    return set(g.select(node).where(Like(node.label.is_in(wanted))))


def recommend(g, model, people, genres, titles, k):
    movie_keys = resolve(g, "movie", titles)
    seed_keys = resolve(g, "person", people) | resolve(g, "genre", genres) | movie_keys

    if not seed_keys:
        return []

    rec = Node("movie")
    seed = Node()                                             # untyped: bound to seed_keys below
    path = Path()

    # the seeds are of mixed types, so no single relation joins them to a film:
    # a person does it through directed_by/acted_in read backwards, a genre
    # through has_genre. Naming none asks the model for the most plausible edge
    # of any kind, which is what a heterogeneous seed set needs.
    embedding = model.seeded(seed_keys)                       # learned KG plausibility
    pagerank = PageRank(to=seed_keys)                         # personalized (RWR): proximity

    ranked = (
        g.select(rec, Score())
         .where(
             cold_start_pattern(path, rec, seed),
             seed.is_in(seed_keys),
             ~rec.is_in(movie_keys),
         )
         .rank(embedding, pagerank)                           # global ranking, comparable scores
         .top(k)
    )
    top = list(ranked)
    if not top:
        return []

    top_keys = {movie for movie, _ in top}
    explain_paths = g.select(path).where(
        cold_start_pattern(path, rec, seed),
        rec.is_in(top_keys),
        seed.is_in(seed_keys),
    )
    why_by_movie = {}
    for walk in explain_paths:
        why_by_movie.setdefault(walk[0], walk)

    return [
        {
            "title": movie.label,
            "score": score,
            "why": explain(why_by_movie[movie]) if movie in why_by_movie else None,
        }
        for movie, score in top
    ]


# --- startup -----------------------------------------------------------------

def load_graph():
    return jb.Graph(
        kg=f"{DATA_DIR}/ml.kg",
        ui=f"{DATA_DIR}/ml.ui",
        attrs=[
            f"{DATA_DIR}/ml.movie",
            f"{DATA_DIR}/ml.genre",
            f"{DATA_DIR}/ml.year",
            f"{DATA_DIR}/ml.user",
            f"{DATA_DIR}/ml.person",
        ],
        sf=(3, 5),
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
        train(TransD(factors=64), graph, epochs=epochs, device=device()).save(path)
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
