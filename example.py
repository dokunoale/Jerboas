"""Cold-start movie recommender (FastAPI service).

The Graph is loaded once at startup (lifespan) and kept in memory: MovieLens
does not change at runtime, so re-parsing it per request would just be wasted
I/O.

    uvicorn example:app --host 0.0.0.0 --port 8000

    curl -X POST localhost:8000/recommend -H 'content-type: application/json' \
         -d '{"people": ["Quentin Tarantino", "Bruce Willis"], "genres": ["Crime"], "k": 10}'

There is no user node (cold start): the model starts from the nodes you named
and expands two hops to reach candidate movies.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import jerboas as jb
from jerboas import Node, Edge, Path, Score, Like, DiffusedMatrixFactorization, PageRank

DATA_DIR = "./data/movielens"


def cold_start_pattern(path, rec, seed):
    # candidate movie -> a liked attribute (1 hop) OR -> attribute -> a liked movie (2 hops).
    # shared by the ranking and explain queries so they can't silently drift apart.
    # Edge() is undirected, so the bridge closes whichever way the edges are stored.
    return (path == [rec, Edge(), seed]) | (path == [rec, Edge(), Node(), Edge(), seed])


def explain(path):
    # path is (rec, rel, [mid, rel,] seed): describe how rec connects to your likes
    seed = path[-1]
    if seed.type == "movie" and len(path) > 3:        # a liked movie, reached via a bridge node
        return f"similar to {seed.label} (shared {path[2].type})"
    return f"{path[-2].name.replace('_', ' ')} {seed.label}"


def resolve(g, type_, names):
    # seed resolution: fuzzy-match the given names against the type's label
    # column, whichever it happens to be -- so this knows nothing about types
    if not names:
        return set()
    node = Node(type_)
    return set(g.select(node).where(Like(node.label.is_in(names))))


def recommend(g, people, genres, titles, k):
    movie_keys = resolve(g, "movie", titles)
    seed_keys = resolve(g, "person", people) | resolve(g, "genre", genres) | movie_keys

    if not seed_keys:
        return []

    rec = Node("movie")
    seed = Node()                                             # untyped: bound to seed_keys below
    path = Path()

    mf = DiffusedMatrixFactorization(to=seed_keys)            # taste relevance (diffused latent space)
    pagerank = PageRank(to=seed_keys)                         # personalized (RWR): proximity to your likes

    ranked = (
        g.select(rec, Score())
         .where(
             cold_start_pattern(path, rec, seed),
             seed.is_in(seed_keys),
             ~rec.is_in(movie_keys),
         )
         .rank(mf, pagerank)                                  # global ranking, comparable scores
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


# --- FastAPI service ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = jb.Graph(
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
    yield
    app.state.graph = None


app = FastAPI(title="Jerboas cold-start recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    people: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    k: int = 10


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
    results = recommend(app.state.graph, body.people, body.genres, body.titles, body.k)
    return RecommendResponse(recommendations=results)


@app.get("/health")
def get_health():
    return {"status": "ok"}
