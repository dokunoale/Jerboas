"""TransE: a relation as a translation in the embedding space.

    Bordes et al., "Translating Embeddings for Modeling Multi-relational Data",
    NeurIPS 2013.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

The original, and the baseline TransD is measured against: a relation is one
vector, and a triple holds when head plus relation lands on tail.

Its known limit is the flip side of that simplicity. Because an entity has a
single vector shared across every relation, TransE cannot represent a one-to-many
relation without collapse: if `movie.1 has_genre genre.1` and `movie.1 has_genre
genre.2` must both hold, then genre.1 and genre.2 are pushed to the same point.
MovieLens is full of such relations -- a film has several genres, several
actors -- which is exactly the case TransD's per-relation projection exists to
handle, and the reason to keep both rather than only the better one.
"""

from .base import NODE, RELATION, Translational


class TransE(Translational):
    name = "transe"
    tables = (("entity", NODE), ("relation", RELATION))

    def plausibility(self, head, relation, tail):
        return self.norm(self.get("entity", head)
                         + self.get("relation", relation)
                         - self.get("entity", tail))
