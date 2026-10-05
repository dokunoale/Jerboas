import os
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jerboas import Graph
from jerboas.rank import MatrixFactorization
from benchmark.split import read_edges, write_edges, train_test_split
from benchmark.metrics import precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k


# how many users are expanded at once. Co-watching is a three-hop walk, and the
# middle of it is the widest part of the graph: every film one user watched, times
# everyone else who watched it, times everything *they* watched. Whole-cohort that
# is hundreds of millions of rows; a block at a time it is tens of millions, and
# the answer is identical because users do not interact with each other.
BLOCK = 128


def candidates(graph, users, k):
    """Films reached from a block of users by co-watching, ranked per user.

    Three hops -- what these users watched, who else watched it, what those
    people watched -- with a `unique` after each. That deduplication is what the
    old engine's memoized sub-path search was doing: two users who watched the
    same film reach the same peers, and there is no reason to carry that row
    twice. Without it the middle of the walk is the product of three degrees.
    """
    watched = (graph.nodes(user=users)
               .hop(seen="has_interact")
               .select("user", "seen"))

    reached = (watched
               .hop(peer="~has_interact")
               .select("user", "peer").unique(["user", "peer"])
               .hop(rec="has_interact")
               .select("user", "rec").unique(["user", "rec"]))

    return (reached
            # the films this user has already seen are not recommendations. An
            # anti-join, vectorized, where this was a Python loop asking the
            # graph about one pair at a time
            .join(watched.rename({"seen": "rec"}), on=["user", "rec"], how="anti")
            .with_columns(score=MatrixFactorization().on("rec", "user"))
            .top(k, by="score", over="user"))


def recommend_all(graph, k):
    """Top-k per user, one block of users at a time."""
    low, high = graph.block("user")
    recommended_by_user = defaultdict(list)
    for start in range(low, high, BLOCK):
        block = list(range(start, min(start + BLOCK, high)))
        ranked = candidates(graph, block, k)
        # keyed by the printable form, so results line up with the raw pairs the
        # split reads straight out of the file
        for user, movie in zip(ranked.keys("user"), ranked.keys("rec")):
            recommended_by_user[str(user)].append(str(movie))
    return recommended_by_user


def run(kg_path="data/movielens/ml.kg", edges_path="data/movielens/ml.has_interact",
        k=10, test_ratio=0.2, seed=42):
    train, test = train_test_split(read_edges(edges_path), test_ratio=test_ratio, seed=seed)

    relevant_by_user = defaultdict(set)
    for user, movie, _score in test:
        relevant_by_user[user].add(movie)

    # the training split is a temporary artifact of this run, not data: writing
    # it next to the dataset would leave a file that looks like part of it and
    # changes under whoever runs the benchmark next. The suffix still names the
    # relation, because that is how the loader reads one.
    with tempfile.TemporaryDirectory() as workdir:
        train_path = os.path.join(workdir, "train.has_interact")
        write_edges(train, train_path)
        graph = Graph(kg=kg_path, edges=[train_path])
        recommended_by_user = recommend_all(graph, k)

    scores = defaultdict(list)
    for user_name, relevant in relevant_by_user.items():
        recommended = recommended_by_user.get(user_name, [])

        scores["precision"].append(precision_at_k(recommended, relevant, k))
        scores["recall"].append(recall_at_k(recommended, relevant, k))
        scores["hit_rate"].append(hit_rate_at_k(recommended, relevant, k))
        scores["ndcg"].append(ndcg_at_k(recommended, relevant, k))

    return {name: sum(values) / len(values) for name, values in scores.items() if values}


if __name__ == "__main__":
    k = 10
    results = run(k=k)
    for name, value in results.items():
        print(f"{name}@{k}: {value:.4f}")
