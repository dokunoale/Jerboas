"""Reading an edge file, and splitting it into train and test.

An edge file is `source <TAB> target <TAB> score` under a header row, and the
score travels with the pair through the split: the graph the benchmark trains on
has to be the same shape as the one the library loads, weights included.
"""

import random
from collections import defaultdict

HEADER = ("source", "target", "score")


def read_edges(path):
    """(source, target, score) rows; an absent score reads as 1, the weight of an
    edge nobody scored."""
    rows = []
    with open(path, "r") as f:
        f.readline()                        # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            score = parts[2] if len(parts) > 2 and parts[2] != "" else "1"
            rows.append((parts[0], parts[1], score))
    return rows


def write_edges(rows, path):
    with open(path, "w") as f:
        f.write("\t".join(HEADER) + "\n")
        for source, target, score in rows:
            f.write(f"{source}\t{target}\t{score}\n")


def train_test_split(rows, test_ratio=0.2, seed=42):
    rng = random.Random(seed)

    by_source = defaultdict(list)
    for source, target, score in rows:
        by_source[source].append((target, score))

    train, test = [], []
    for source, edges in by_source.items():
        edges = edges[:]
        rng.shuffle(edges)

        n_test = round(len(edges) * test_ratio) if len(edges) > 1 else 0
        held_out, kept = edges[:n_test], edges[n_test:]

        train += [(source, target, score) for target, score in kept]
        test += [(source, target, score) for target, score in held_out]

    return train, test
