"""How much of an exact walk a budgeted one keeps, and what it costs.

The co-watching walk of run.py -- what a user watched, who else watched it, what
they watched -- taken exactly and then with a budget on its widest step. Three
numbers per variant: the seconds the walk took, the NDCG@10 of what it
recommends, and the share of the exact top 10 it recovers. The last is the one a
budget is judged by: a model scores the candidates either way, so what a budget
can lose is only what it never let the model see.

    python -m benchmark.approx
"""

import os
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jerboas import Graph, step, v
from jerboas.rank import MatrixFactorization
from benchmark.metrics import ndcg_at_k
from benchmark.run import BLOCK
from benchmark.split import read_edges, train_test_split, write_edges


def walk(graph, users, peers):
    """The candidates of run.candidates, with `peers(frame)` choosing which of
    a film's watchers the walk goes on through."""
    watched = graph.nodes(user=users).hop(seen="has_interact").select("user", "seen")
    reached = peers(watched)
    return (reached.select("user", "peer").unique(["user", "peer"])
            .hop(rec="has_interact")
            .select("user", "rec").unique(["user", "rec"])
            .join(watched.rename({"seen": "rec"}), on=["user", "rec"], how="anti"))


VARIANTS = {
    "exact": lambda frame: frame.hop(peer="~has_interact"),
}
# the same budget two ways: cut after the whole expansion (`top ... over`), or
# fused into the step so the expansion never happens (`step`)
for width in (50, 20, 5):
    VARIANTS[f"cut top {width}"] = (
        lambda frame, width=width: frame.hop(peer="~has_interact")
        .top(width, by=v.peer.score, over=["user", "seen"]))
    VARIANTS[f"cut gumbel {width}"] = (
        lambda frame, width=width: frame.hop(peer="~has_interact")
        .top(width, by=v.peer.score, over=["user", "seen"], temperature=1.0, seed=0))
    best, drawn = step("~has_interact").top(width), step("~has_interact").sample(width, seed=0)
    VARIANTS[f"step top {width}"] = lambda frame, best=best: frame.hop(peer=best)
    VARIANTS[f"step sample {width}"] = lambda frame, drawn=drawn: frame.hop(peer=drawn)
    # the watchers who watched least: what they share with you says the most
    specific = step("~has_interact").top(width, by=1 / v.peer.has_interact.count().log1p())
    VARIANTS[f"step specific {width}"] = lambda frame, specific=specific: frame.hop(peer=specific)


def recommend(graph, model, variant, k):
    low, high = graph.block("user")
    out, seconds = defaultdict(list), 0.0
    for start in range(low, high, BLOCK):
        users = list(range(start, min(start + BLOCK, high)))
        began = time.perf_counter()
        candidates = walk(graph, users, VARIANTS[variant])
        seconds += time.perf_counter() - began
        ranked = (candidates.with_columns(score=model.on("rec", "user"))
                  .top(k, by="score", over="user"))
        for user, movie in zip(ranked.keys("user"), ranked.keys("rec")):
            out[str(user)].append(str(movie))
    return out, seconds


def main(kg="data/movielens/ml.kg", edges="data/movielens/ml.has_interact", k=10):
    train, test = train_test_split(read_edges(edges), test_ratio=0.2, seed=42)
    relevant = defaultdict(set)
    for user, movie, _score in test:
        relevant[user].add(movie)
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "train.has_interact")
        write_edges(train, path)
        graph = Graph(kg=kg, edges=[path])
        model = MatrixFactorization()
        model.fit(graph)
        exact = None
        print(f"{'variant':<16} {'walk s':>7} {'ndcg@10':>8} {'kept':>6}")
        for variant in VARIANTS:
            recommended, seconds = recommend(graph, model, variant, k)
            exact = exact or recommended
            ndcg = sum(ndcg_at_k(recommended.get(u, []), r, k)
                       for u, r in relevant.items()) / len(relevant)
            kept = sum(len(set(recommended.get(u, [])) & set(exact[u]))
                       for u in exact) / sum(len(one) for one in exact.values())
            print(f"{variant:<16} {seconds:7.2f} {ndcg:8.4f} {kept:6.1%}")


if __name__ == "__main__":
    main()
