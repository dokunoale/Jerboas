"""TransD, fitted.

    Ji et al., "Knowledge Graph Embedding via Dynamic Mapping Matrix", ACL 2015.

Adapted from the implementation in hopwise (https://github.com/tail-unica/hopwise,
MIT, Copyright (c) 2020 tail @ UNICA), itself following torchkge.

The maths and the table layout are in kge.TRANSD; only fitting lives here.

One departure from the reference implementation: there is no separate user
embedding table and no relation slot reserved for interactions. Here a user is an
entity and `has_interact` is a relation like any other, so one entity table and
one relation table cover both recommendation and KG completion with the same code
path. That follows from the data model rather than simplifying the method.
"""

from ..kge import TRANSD
from .base import Translational


class TransD(Translational):
    spec = TRANSD
