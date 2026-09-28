"""Playlist continuation (FastAPI service), on the Spotify Million Playlist Dataset.

    ./run.sh spotify                                  # in a container
    uvicorn app:app --app-dir usecase/spotify         # or straight from here

    curl -X POST localhost:8000/extend -H 'content-type: application/json' \
         -d '{"songs": ["Toxic", "Lose Control", "Bad Romance"], "k": 5}'

    open http://localhost:8000/ui                     # or try it by hand

You bring a handful of songs; the service answers with five more that belong
beside them. The files, in the order a request goes through them:

    names.py       the names you typed, as the songs they meant
    recommend.py   the walk from those songs, and how its answers are ranked
    startup.py     what a start loads: the graph, the factorization, a warm-up
    ui.py          the page at /ui
    app.py         this: the service, its start and its two routes
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import ui
from recommend import PLAYLISTS, extend
from startup import fit, load_graph, timed, warm


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
    # how many of each song's playlists the walk goes through; 0 goes through
    # all of them, which is exact and, for a popular song, seconds slower
    playlists: int = Field(default=PLAYLISTS, ge=0, le=100_000)


class Suggestion(BaseModel):
    song: str
    artist: str | None = None
    score: float
    # how many of the playlists the walk went through hold this one too
    playlists: int


class ExtendResponse(BaseModel):
    # what the fuzzy match landed on, so "Bad Romance" turning into something
    # else is something you can see
    songs: list[str]
    suggestions: list[Suggestion]


@app.post("/extend", response_model=ExtendResponse)
def post_extend(body: ExtendRequest) -> ExtendResponse:
    if not body.songs:
        raise HTTPException(status_code=422, detail="name at least one song")
    named, suggestions = extend(app.state.graph, app.state.model, app.state.known,
                                body.songs, body.k, body.concentration,
                                body.temperature, body.playlists)
    if not named:
        raise HTTPException(status_code=404, detail="no song matched")
    return ExtendResponse(songs=named, suggestions=suggestions)


@app.get("/health")
def get_health() -> dict:
    graph = app.state.graph
    sized = lambda kind: graph.block(kind)[1] - graph.block(kind)[0]
    return {"status": "ok", "playlists": sized("playlist"), "songs": sized("song"),
            "known": len(app.state.known)}


@app.get("/", include_in_schema=False)
def get_root() -> RedirectResponse:
    return RedirectResponse("/ui")


# last, so the routes above are the app's own and /ui is only the page's
app = gr.mount_gradio_app(app, ui.build(app.state), path="/ui")
