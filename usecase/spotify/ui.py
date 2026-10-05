"""A page to try the service by hand: songs in, suggestions out.

Mounted inside the FastAPI app (`app.py`, at /ui), so it answers from the graph
and the factorization the service already holds rather than loading its own:
on the whole graph that is gigabytes, and a second copy would not fit.

The budget is on the page because it is the thing worth seeing: set it to 0 and
the walk goes through every playlist, and the time under the answer says what
that costs.
"""

import time

import gradio as gr

from recommend import PLAYLISTS, extend

EXAMPLES = [
    ["Smells Like Teen Spirit\nCome As You Are\nLithium\nPoker Face\nBad Romance\nToxic",
     5, 0.5, 0.0, PLAYLISTS, False],
    ["Wonderwall | Oasis\nChampagne Supernova | Oasis\nDon't Look Back in Anger | Oasis",
     5, 0.0, 0.0, PLAYLISTS, False],
    ["Toxic\nLose Control\nBad Romance", 5, 0.0, 0.0, 0, False],
]


def names(text: str) -> list[str]:
    """One song per line; `Title | Performer` says which recording, as a tab
    does for the API -- a tab being hard to type in a text box."""
    return ["\t".join(part.strip() for part in line.split("|", 1))
            for line in text.splitlines() if line.strip()]


def build(state) -> gr.Blocks:
    """The page, reading the service's graph and model off `state` (the app's
    state) when it is used, not when it is built -- the page is built before
    the service has loaded anything."""

    def suggest(text: str, k: float, concentration: float, temperature: float,
                playlists: float, exclude_artists: bool) -> tuple[str, list[list]]:
        asked = names(text)
        if not asked:
            raise gr.Error("name at least one song")
        start = time.perf_counter()
        named, suggestions = extend(state.graph, state.model, state.known, asked,
                                    int(k), concentration, temperature, int(playlists),
                                    exclude_artists)
        seconds = time.perf_counter() - start
        if not named:
            raise gr.Error("no song matched")
        walked = f"{int(playlists)} playlists per song" if playlists else "every playlist"
        found = "\n".join(f"- {one}" for one in named)
        summary = f"**Read as**\n\n{found}\n\n*{seconds:.2f} s, walking {walked}*"
        return summary, [[one["song"], one["artist"],
                          f"{one['playlists']} ± {one['error']}" if one["error"]
                          else str(one["playlists"]),
                          round(one["score"], 4)]
                         for one in suggestions]

    with gr.Blocks(title="Jerboas playlist continuation") as page:
        gr.Markdown("## Playlist continuation\nA few songs in, the ones that belong "
                    "beside them out.")
        with gr.Row():
            with gr.Column():
                songs = gr.Textbox(label="Songs", lines=6,
                                   placeholder="one per line -- Title | Performer to say which")
                k = gr.Slider(1, 20, value=5, step=1, label="Suggestions")
                concentration = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05, label="Concentration",
                    info="0 reads the songs as one playlist, 1 answers each of them")
                temperature = gr.Slider(
                    0.0, 2.0, value=0.0, step=0.1, label="Temperature",
                    info="0 answers the same every time; above it, a sample")
                playlists = gr.Slider(
                    0, 1000, value=PLAYLISTS, step=25, label="Playlists per song",
                    info="how many the walk goes through; 0 is all of them, exactly")
                exclude_artists = gr.Checkbox(
                    label="Exclude input artists", value=False,
                    info="drop suggestions by the performers you named")
                ask = gr.Button("Suggest", variant="primary")
            with gr.Column():
                summary = gr.Markdown()
                table = gr.Dataframe(headers=["song", "artist", "playlists", "score"],
                                     interactive=False)
        inputs = [songs, k, concentration, temperature, playlists, exclude_artists]
        gr.Examples(EXAMPLES, inputs=inputs)
        ask.click(suggest, inputs=inputs, outputs=[summary, table])
        songs.submit(suggest, inputs=inputs, outputs=[summary, table])
    return page
