"""Playlist continuation on the whole Spotify graph, exact and with a budget.

The query of usecase/spotify, from ids rather than titles: five songs of a real
playlist are the seeds, the playlists holding them are the crowd, what else the
crowd holds is voted for, and the factorization re-ranks it. The playlist the
seeds came from is left out of the crowd, and the rest of its songs are what a
good answer finds.

Two things are varied, and apart: what a vote weighs (the map) and which
playlists and songs the walk follows at all (the budget). A budget is judged
against the exact walk under the same vote -- otherwise what the vote changes
would be credited to the budget.

Per variant: the latency (median, p90, p99), the share of the exact top 10
under the same vote it returns (`kept`), and hit@10 -- the share of its top 10
the playlist really held -- with a 95% bootstrap interval. Against the exact
walk under the same vote, the paired difference in hit, its interval, and a
Wilcoxon signed-rank p-value: every variant answers the same queries, so the
comparison is per query and not between two averages.

    python -m benchmark.playlists [queries]
    SPOTIFY_DIR=data/spotify/graph-100k python -m benchmark.playlists 50
    VARIANTS="count exact,balanced sample 100" python -m benchmark.playlists
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
import polars as pl
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jerboas as jb
from jerboas import DiffusedMatrixFactorization, step, v

DATA = os.environ.get("SPOTIFY_DIR", "data/spotify/graph")
CHECKPOINT = os.environ.get(
    "SPOTIFY_CHECKPOINT",
    f"checkpoints/spotify.{os.path.basename(os.path.normpath(DATA))}.dmf.npz")
SEEDS, K, SUPPORT = 5, 10, 20

# the same graph arguments as the service, so the same cache answers both
READABLE = {"song": "name", "artist": "name", "album": "name", "playlist": "name"}

# a playlist's specificity: the fewer songs it holds, the more each says
SPECIFIC = 1 / v.playlist.contains.count().log1p()

# What a vote weighs, as (which steps measure a probability, what is summed):
#   count     one per playlist: a seed in fifty thousand playlists outvotes
#             four seeds in fifty
#   balanced  each seed's playlists share one vote: the first step's
#             transition probability
#   walk      the probability of the two-step walk seed -> playlist -> song,
#             which also divides a playlist's vote among its songs
VOTES = {"count": ((False, False), None),
         "balanced": ((True, False), v.playlist.score),
         # the last step's confidence composes the first's: the walk's probability
         "walk": ((True, True), v.rec.score)}

# Which edges the walk follows: (to the playlists, to their songs).
BUDGETS = {"exact": (step("~contains"), step("contains")),
           "sample 100": (step("~contains").sample(100, by=1, seed=0), step("contains")),
           "specific 100": (step("~contains").top(100, by=SPECIFIC), step("contains")),
           "sample 100x50": (step("~contains").sample(100, by=1, seed=0),
                             step("contains").sample(50, by=1, seed=0))}


def steps(vote, budget):
    """The two steps of a variant: the budget's, measuring a probability where
    the vote reads one. Built once, so a map fused into the store is fused
    once."""
    measured, _summed = VOTES[vote]
    return tuple(one.probability(by=1) if wanted else one
                 for one, wanted in zip(BUDGETS[budget], measured))


VARIANTS = {(vote, budget): steps(vote, budget) for vote in VOTES for budget in BUDGETS}


def queries(graph, known, n, rng):
    """(source playlist, seed ids, held-out ids) for n playlists holding enough
    known songs to split."""
    low, high = graph.block("playlist")
    known = set(known.tolist())
    out = []
    while len(out) < n:
        source = int(rng.integers(low, high))
        songs = [one for one in graph.nodes(p=[source]).hop(song="contains").ids("song")
                 if one in known]
        songs = list(dict.fromkeys(songs))
        if len(songs) < SEEDS + 10:
            continue
        rng.shuffle(songs)
        out.append((source, songs[:SEEDS], set(songs[SEEDS:])))
    return out


def answer(graph, model, known, source, seeds, variant):
    to_playlists, to_songs = VARIANTS[variant]
    _measured, summed = VOTES[variant[0]]
    seeds = graph.nodes(seed=seeds)
    with jb.optimize():
        reached = (seeds.hop(playlist=to_playlists)
                   .filter(v.playlist != source)
                   .hop(rec=to_songs)
                   .filter(~v.rec.is_in(seeds), v.rec.is_in(known)))
        counted = (reached.group_by(v.rec).len("shared") if summed is None
                   else reached.group_by(v.rec).agg(shared=summed.sum()))
    if not len(counted):
        return []
    # the vote is damped either way -- log1p of a count, or of a share of the
    # seeds' mass scaled to a count's size, so the damping bends the same way
    scale = 1.0 if summed is None else 100.0
    return (counted.with_columns(taste=model.seeded(seeds).on("rec"))
            .with_columns(score=v.taste * (scale * v.shared.cast(pl.Float64)).log1p())
            .top(K, by=v.score).ids("rec"))


def interval(values, rng, rounds=2000):
    """The mean and its 95% bootstrap interval."""
    values = np.asarray(values, dtype=np.float64)
    means = values[rng.integers(0, len(values), (rounds, len(values)))].mean(axis=1)
    return values.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)


def warm(data):
    """Read the graph's cache once, so no variant pays for pages another one
    then finds in memory: the store is mapped from disk, and on an external
    one the first walk to touch a page is timing the disk."""
    folder = os.path.join(data, ".cache")
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                while handle.read(1 << 24):
                    pass


def main(n=1000):
    began = time.perf_counter()
    graph = jb.Graph(kg=f"{DATA}/spotify.kg", edges=[f"{DATA}/spotify.contains"],
                     attrs=[f"{DATA}/spotify.{one}"
                            for one in ("song", "artist", "album", "playlist")],
                     readable=READABLE, cache=f"{DATA}/.cache")
    model = DiffusedMatrixFactorization.load(CHECKPOINT, graph)
    known = (graph.nodes(song="song").with_columns(seen=v.song.contains.count())
             .filter(v.seen >= SUPPORT).ids("song"))
    asked = queries(graph, known, n, np.random.default_rng(0))
    warm(DATA)
    print(f"{graph}\nloaded in {time.perf_counter() - began:.1f} s, "
          f"{len(asked)} playlists", flush=True)

    wanted = os.environ.get("VARIANTS")
    chosen = [variant for variant in VARIANTS
              if wanted is None or " ".join(variant) in wanted.split(",")]
    # every variant before any is timed: the fused orders are built, the
    # factorization's cache filled. Then per query every variant, in an order
    # that rotates, so drift in the machine falls on all of them alike
    for source, seeds, _held in asked[:20]:
        for variant in chosen:
            answer(graph, model, known, source, seeds, variant)
    seconds, tops = defaultdict(list), defaultdict(list)
    for turn, (source, seeds, _held) in enumerate(asked):
        shift = turn % len(chosen)
        for variant in chosen[shift:] + chosen[:shift]:
            start = time.perf_counter()
            tops[variant].append(answer(graph, model, known, source, seeds, variant))
            seconds[variant].append(time.perf_counter() - start)
        if (turn + 1) % 100 == 0:
            print(f"  {turn + 1} queries", flush=True)

    rng = np.random.default_rng(1)
    print(f"\n{'vote':<9} {'budget':<14} {'p50 s':>6} {'p90 s':>6} {'p99 s':>6} "
          f"{'kept':>6}  {'hit@10 [95% CI]':<22} {'vs exact [95% CI]':<24} {'p':>7}")
    for variant in chosen:
        vote, budget = variant
        exact = tops.get((vote, "exact"), tops[variant])
        kept = np.mean([len(set(top) & set(ref)) / max(len(ref), 1)
                        for top, ref in zip(tops[variant], exact)])
        hit = np.array([len(set(top) & held) / K
                        for top, (_s, _q, held) in zip(tops[variant], asked)])
        reference = np.array([len(set(top) & held) / K
                              for top, (_s, _q, held) in zip(exact, asked)])
        mean, low, high = interval(hit, rng)
        line = (f"{vote:<9} {budget:<14} {np.median(seconds[variant]):6.3f} "
                f"{np.quantile(seconds[variant], 0.9):6.3f} "
                f"{np.quantile(seconds[variant], 0.99):6.3f} {kept:6.1%}  "
                f"{mean:6.1%} [{low:5.1%}, {high:5.1%}]")
        difference = hit - reference
        if budget != "exact" and difference.any():
            d, d_low, d_high = interval(difference, rng)
            p = wilcoxon(hit, reference).pvalue
            line += f"  {d:+6.1%} [{d_low:+5.1%}, {d_high:+5.1%}]  {p:7.1e}"
        print(line, flush=True)


if __name__ == "__main__":
    main(*(int(one) for one in sys.argv[1:]))
