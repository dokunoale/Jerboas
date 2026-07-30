"""TransE: a relation as a translation in the embedding space.

    Bordes et al., "Translating Embeddings for Modeling Multi-relational Data",
    NeurIPS 2013.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

The original, and the baseline TransD is measured against: a relation is one
vector, and a triple holds when head + relation lands on tail. One table for
entities, one for relations, no projection.

Its known limit is the flip side of that simplicity. Because an entity has a
single vector shared across every relation, TransE cannot represent a one-to-many
relation without collapse: if `movie.1 has_genre genre.1` and `movie.1 has_genre
genre.2` both have to hold, then `genre.1` and `genre.2` are pushed to the same
point. MovieLens is full of such relations -- a film has several genres, several
actors -- which is exactly the case TransD's per-relation projection exists to
handle, and the reason to keep both around rather than only the better one.
"""

import torch

from .base import Translational


class TransE(Translational):
    name = "transe"
    tables = ("entity", "relation")

    def score(self, head, relation, tail):
        translated = self.entity(head) + self.relation(relation)
        return -torch.norm(translated - self.entity(tail), p=2, dim=-1)
