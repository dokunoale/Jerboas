import os
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jerboas import Graph, Node, Edge, Path, Has
from jerboas.strategies import MatrixFactorization
from benchmark.split import read_edges, write_edges, train_test_split
from benchmark.metrics import precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k


def recommend_all(graph, k):
    # one query, one MatrixFactorization instance, for every user: groupby(user)
    # ranks and keeps the top-k within each user's group, instead of looping
    # per-user with a fresh (unfit, uncached) strategy instance each time
    user = Node(type="user")
    rec = Node(type="movie")
    path = Path()

    rows = graph.select(user, rec).where(
        path == [user, Edge(), Node(type="movie"), Edge(), Node(type="user"), rec],
        ~Has(user, "has_interact", rec),   # skip already-watched movies
    ).groupby(user).rank(MatrixFactorization()).top(k)

    # keyed by the printable form, so results line up with the raw pairs the
    # split reads straight out of the file
    recommended_by_user = defaultdict(list)
    for u, movie in rows:
        recommended_by_user[str(u)].append(str(movie))
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
