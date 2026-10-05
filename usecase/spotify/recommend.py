"""Five songs that belong beside a handful you name.

There is no playlist node for what you brought -- the input is a list of songs,
and what stands in for the playlist you are building is the crowd of real ones
that already contain those songs.

Three signals, and the query says how much each counts:

  * the **walk** is the evidence: a random walk from your songs to the
    playlists holding them and on to what else those hold. How likely it is to
    end on a song is how much the crowd vouches for it. Each of your songs
    shares one vote among its playlists, however many there are, and each
    playlist shares its own among its songs -- so a song in fifty thousand
    playlists does not outvote four in fifty, and a playlist of a thousand
    songs does not vouch for each of them as much as one of twelve.
  * the **model** is the taste: a matrix factorization of the playlist-song
    matrix, asked how near a candidate is to what you brought. On its own it is
    a poor recommender here; as a re-ranker over songs the crowd already agrees
    on, it is what separates *the same artists* from *the same decade*.
  * **gathered** is whether a song belongs anywhere at all -- a film score sits
    in five hundred playlists with nothing in common.

The walk goes through a budget of playlists per song rather than all of them
(`PLAYLISTS`). On the whole graph, measured over a thousand real playlists
(benchmark/playlists.py), that takes the slowest requests from seven seconds to
a third of one and costs about a point of hit rate; walking all of them is
`playlists=0`. What the budget costs the count, it says rather than hides:
`shared` is an estimate of the whole crowd's, drawn from the chance each
sampled playlist had of being drawn, and the answer carries how far that
estimate may be off (`error`, zero when the walk was exact).
"""

from typing import Any

import polars as pl
from scipy.cluster.vq import kmeans2

import jerboas as jb
from jerboas import Concentration, DiffusedMatrixFactorization, reverse, step, v

from names import describe, resolve

# how many of each song's playlists the walk goes through: drawn, not chosen,
# since a playlist's merit is what the walk is measuring. 0 walks all of them
PLAYLISTS = 100

# the walk's probability summed over your songs is at most their number; scaled
# to a count's size before it is damped, so the damping bends as it was measured
SCALE = 100.0


def walk(playlists: int = PLAYLISTS) -> tuple[jb.Step, jb.Step]:
    """The two steps, each measuring the probability a random walk takes it:
    from a song to its playlists, and from a playlist to its songs.

    The draw has a seed, so the same songs are answered the same way every
    time: approximate, and reproducible. `inclusion()` keeps the chance each
    drawn playlist had of being drawn, so a count over the walk is an estimate
    of the whole crowd's, not of the hundred that answered."""
    to_playlists = step(reverse("contains"))
    if playlists:
        to_playlists = to_playlists.sample(playlists, by=1, seed=0).inclusion()
    return to_playlists.probability(by=1), step("contains").probability(by=1)


def clustered(graph: jb.Graph, model: DiffusedMatrixFactorization, seeds: jb.Frame,
              concentration: float) -> jb.Frame:
    """The seeds, each labelled with the part of the playlist it belongs to.

    A playlist is one thing when it is about one thing and several when it is
    not, and only whoever asked knows which. `concentration` says how finely to
    split it: at 0 the whole playlist is one field and the answers come from
    wherever in it they score best, at 1 every song is its own and the answers
    cover all of them. In between, k-means over the latent space.

    Splitting is what stops a broad playlist from being answered entirely out of
    its largest corner."""
    ids = seeds.ids("seed")
    parts = 1 + round(concentration * (len(ids) - 1))
    if parts <= 1:
        return seeds.with_columns(part=pl.lit(0, dtype=pl.Int32))
    space = model.embeddings(graph)[ids]
    labels, _ = kmeans2(space, min(parts, len(ids)), minit="++", seed=0)[::-1]
    return seeds.with_columns(part=pl.Series("part", labels, dtype=pl.Int32))


def candidates(graph: jb.Graph, seeds: jb.Frame, known: jb.Frame,
               playlists: int = PLAYLISTS,
               exclude: jb.Frame | None = None) -> jb.Frame:
    """Every song the factorization knows that the walk reaches, within the
    part of the playlist each seed belongs to: how likely the walk is to end
    there is its confidence (`v.rec.score`), and `shared` is in how many of
    the crowd's playlists it sits -- estimated, when the walk drew its
    playlists, from the draw's inclusion probabilities (the estimate's
    standard error rides as `v.shared.score`).

    `exclude` is a frame of artists whose songs may not come back -- the
    input's own performers, when asked. It filters by the relation rather than
    by a walk: `v.rec.performed_by.is_in(exclude)` is whether an edge to that
    set exists, which the graph answers from adjacency without expanding the
    frame.

    Planned (`jb.optimize`): walked in full, a handful of popular seeds reach
    tens of thousands of playlists and millions of rows, and the plan runs it a
    slice at a time. With a budget there is little left to slice."""
    to_playlists, to_songs = walk(playlists)
    with jb.optimize():
        reached = (seeds
                .hop(playlist=to_playlists)
                .hop(rec=to_songs)
                .filter(~v.rec.is_in(seeds), v.rec.is_in(known)))
        if exclude is not None:
            reached = reached.filter(~v.rec.performed_by.is_in(exclude))
        return (reached
                # the walks that end on a song, added up
                .group_by(v.rec, v.part, confidence="sum")
                .len("shared"))


def rank(counted: jb.Frame, seeds: jb.Frame, model: DiffusedMatrixFactorization, k: int,
         temperature: float) -> jb.Frame:
    """Three signals, and the query says how much each counts.

    How likely the walk is to end on a song is the crowd's evidence, damped:
    twice as likely is a better answer, not twice as good a one, and undamped it
    buries the other two. `taste` is the model's, `gathered` whether the song
    belongs anywhere.

    Then k taken in turns from each part, so a playlist about two things is
    answered about both, and a part with little to offer leaves its turns to
    the others. A song reached from two parts counts in the one it scores
    best in."""
    return (counted
            .with_columns(taste=model.seeded(seeds).on("rec"),
                          gathered=Concentration(model, relation="contains").on("rec"))
            .with_columns(score=v.taste * (SCALE * v.rec.score).log1p() * v.gathered)
            .sort("score", descending=True).unique("rec")
            .top(k, by=v.score, spread="part", temperature=temperature)
            # the turns decide which songs; the score, the order they are read in
            .sort("score", descending=True))


def extend(graph: jb.Graph, model: DiffusedMatrixFactorization, known: jb.Frame,
           songs: list[str], k: int, concentration: float = 0.0,
           temperature: float = 0.0, playlists: int = PLAYLISTS,
           exclude_artists: bool = False) -> tuple[list[str], list[dict[str, Any]]]:
    """(what your names resolved to, the suggestions), both readable.

    `exclude_artists` keeps the performers you named out of the answer, so the
    suggestions say what belongs beside them rather than more of the same."""
    seeds = resolve(graph, songs)
    if seeds is None or not len(seeds):
        return [], []
    found = describe(seeds, "seed")
    named = (found.pl.select(pl.format("{} -- {}", "seed.name", "artist.name"))
             .to_series().to_list())

    parts = clustered(graph, model, seeds, concentration)
    counted = (candidates(graph, parts, known, playlists,
                          exclude=found.select("artist") if exclude_artists else None)
               .attrs(rec="name")
               # another master of a song you gave me is not a suggestion
               .filter(~v.rec.name.is_in(found.pl["seed.name"].to_list())))
    if not len(counted):
        return named, []

    ranked = describe(rank(counted, seeds, model, k, temperature), "rec")
    # a count's estimate reads as a count -- no decimals it cannot vouch for --
    # and, when the walk drew its playlists, the error of the estimate beside
    # it; walked in full the count is exact and the error is none. The error
    # rides on `shared` as its confidence, which `.raw` keeps and `.pl` hides.
    # A draw that cut nothing allocates nothing (certainty costs no column,
    # like a confidence of all 1.0), so its absence *is* an exact count.
    shared = pl.col("shared").cast(pl.Float64).round(0).cast(pl.Int64)
    error = (pl.col("__jb_score__shared").cast(pl.Float64).round(0).cast(pl.Int64)
             if "__jb_score__shared" in ranked.raw.columns
             else pl.lit(0, dtype=pl.Int64))
    return named, ranked.raw.select(song="rec.name", artist="artist.name",
                                    score="score", playlists=shared,
                                    error=error).rows(named=True)
