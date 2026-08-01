"""TransE, fitted.

    Bordes et al., "Translating Embeddings for Modeling Multi-relational Data",
    NeurIPS 2013.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

The maths and the table layout are in kge.TRANSE; only fitting lives here.

The original, and the baseline TransD is measured against. Its known limit is the
flip side of its simplicity: an entity has a single vector shared across every
relation, so a one-to-many relation cannot be represented without collapse -- if
`movie.1 has_genre genre.1` and `movie.1 has_genre genre.2` must both hold, then
genre.1 and genre.2 are pushed to the same point. MovieLens is full of such
relations, which is exactly the case TransD's per-relation projection exists to
handle, and the reason to keep both rather than only the better one.
"""

from ..kge import TRANSE
from .base import Translational


class TransE(Translational):
    spec = TRANSE
