import random
from collections import defaultdict


def read_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            # la ui puo' avere una terza colonna (rating): viene ignorata
            user, item = parts[0], parts[1]
            pairs.append((user, item))
    return pairs


def write_pairs(pairs, path):
    with open(path, "w") as f:
        for user, item in pairs:
            f.write(f"{user}\t{item}\n")


def train_test_split(pairs, test_ratio=0.2, seed=42):
    rng = random.Random(seed)

    by_user = defaultdict(list)
    for user, song in pairs:
        by_user[user].append(song)

    train, test = [], []
    for user, songs in by_user.items():
        songs = songs[:]
        rng.shuffle(songs)

        n_test = round(len(songs) * test_ratio) if len(songs) > 1 else 0
        test_songs, train_songs = songs[:n_test], songs[n_test:]

        train += [(user, song) for song in train_songs]
        test += [(user, song) for song in test_songs]

    return train, test
