"""How a search decides what is close: the extension point for admission.

`Strategy` is where ranking is plugged in -- a thing that computes with whatever
it needs and hands back a column. This is the same for the other half. A `Rule`
answers one question, over a column and a set of needles: which rows are close
enough, how close, and to which needle.

    .filter(v.person.name.like(names, rule=Fuzzy(k=3)))
    .filter(v.song.name.like(titles, rule=Words(k=8)))
    .filter(v.chunk.embedding.like(query, rule=Semantic(k=50)))

One verb, three measures, and the same three answers on the other side --
`v.x.score` is how close, `v.x.needle` is to what. A service doing retrieval
over text and a service doing it over embeddings write the same query.

`near` is the exclusive reading of the same verb: the k closest that are *not*
the thing itself. What "the thing itself" means is the rule's to say, and every
rule here says the same: a perfect score. A remaster of a song is that song.

The division of labour is the one `Strategy` already follows. The **measure**
uses whatever it must -- an inverted index, a matmul, difflib -- because a
posting list is not a frame operation and pretending otherwise would mean
scanning, which is what an index exists to avoid. The **selection** is the
library's own vocabulary, so a rule cannot be written that the library could not
have expressed.
"""

from abc import ABC, abstractmethod

import numpy as np

from . import fuzzy
from .fuzzy import words_of


class Search:
    """What a rule is given: one column of one frame, and the graph behind it.

    A rule reads only what it needs -- the values, the node ids, the graph's
    index -- and each of those is worked out once, on being asked."""

    __slots__ = ("frame", "column", "var", "attribute", "_read", "_values", "_ids")

    def __init__(self, frame, column, read):
        self.frame = frame
        self.column = column
        self.var, _, self.attribute = column.partition(".")
        self._read = read              # what materializes the column, if asked
        self._values = None
        self._ids = None

    @property
    def graph(self):
        return self.frame.graph

    @property
    def stored(self):
        """The graph column this one reads, which is not always its name: a
        `label` is whichever column the graph's `readable` map points at, and an
        index is keyed by the column rather than by the path."""
        if not self.type or not self.attribute:
            return None
        return self.frame._named(
            self.type, None if self.attribute == "label" else self.attribute)

    @property
    def type(self):
        """The node type the column belongs to, or None when it belongs to no
        single one -- in which case nothing the graph indexed applies."""
        return self.frame.vars.get(self.var)

    @property
    def values(self):
        """The column's values, one per row of the frame.

        Read on being asked, because a rule that reads the graph's own tables --
        a vector block, an index -- never asks, and materializing a column of
        vectors as values would be work done to be thrown away."""
        if self._values is None:
            self._values = self._read()
        return self._values

    @property
    def ids(self):
        """The node ids of the rows, for a rule that reads the graph's own
        tables rather than the frame's values."""
        if self._ids is None:
            self._ids = self.frame._df[self.var].to_numpy()
        return self._ids

    def local(self):
        """The same ids as positions inside their type's block, which is how
        anything the graph indexed is keyed."""
        return self.ids - self.graph.block(self.type)[0]


class Rule(ABC):
    """A measure of closeness, and how many of the closest to admit."""

    @abstractmethod
    def matches(self, search, needles, exclusive=False):
        """{row: (closeness, needle)} for the rows this rule admits.

        `exclusive` drops what is identical to a needle rather than close to it,
        and drops it before the k are counted -- `near` asks for k answers, not
        for k minus however many were the question."""


class Fuzzy(Rule):
    """Character similarity: a fragment finds the whole, a typo still lands.

    Containment scores a flat 1.0 and the shortest containing value is preferred
    when there are more than k of them; anything else is difflib's ratio above
    `cutoff`. It reads every value of the column, which is the price of asking a
    question no index was built for.

    It cannot tell `Toxic` from `Toxicity`: both contain the needle's characters.
    `Words` can."""

    def __init__(self, k=1, cutoff=0.6):
        self.k = k
        self.cutoff = cutoff

    def matches(self, search, needles, exclusive=False):
        texts = [(row, str(value).lower()) for row, value in enumerate(search.values)
                 if value is not None]
        return fuzzy.best(needles, texts, self.k, self.cutoff, exclusive)


class Words(Rule):
    """Whole-word overlap, answered by the graph's index.

    How close a value is, is how much of the needle it holds, as words: two of
    the needle's two words is 1.0 and one of them is 0.5. Which is the
    distinction a search over titles needs and the one characters cannot make --
    `Toxicity` is one word and holds none of `Toxic`, while
    `Wonderwall - Remastered` holds all of `Wonderwall` and is as close as the
    bare title.

    Graded rather than all-or-nothing for a reason beyond taste: `near` asks for
    what is close and not identical, and under an all-or-nothing reading of
    closeness there is nothing between the two.

    It reads the index rather than the column -- posting lists, unioned and
    counted -- so what it costs is the number of rows that hold any of the words
    rather than the number that exist. Where the index cannot help, because the
    column is computed or the variable holds no single type, it falls back to
    `Fuzzy` rather than answering less well without saying so; and so does a
    needle no row holds a word of, which is what keeps a typo working."""

    def __init__(self, k=1, cutoff=0.0):
        self.k = k
        self.cutoff = cutoff

    def matches(self, search, needles, exclusive=False):
        index = (search.graph.words(search.type, search.stored)
                 if search.stored else None)
        if index is None:
            return Fuzzy(self.k).matches(search, needles, exclusive)

        seats = {local: row for row, local in enumerate(search.local().tolist())}
        found, missed = {}, []
        for needle in needles:
            scored = self._overlap(index, seats, search, needle, exclusive)
            if not scored:
                missed.append(needle)
                continue
            for row, score in scored:
                if score > found.get(row, (0.0, None))[0]:
                    found[row] = (score, needle)
        if missed:
            # no row holds a word of these: misspelt, or asked of a column that
            # does not answer it, and characters are all that is left
            found.update(Fuzzy(self.k).matches(search, missed, exclusive))
        return found

    def _overlap(self, index, seats, search, needle, exclusive):
        """The k rows holding most of the needle, as [(row, share)]."""
        wanted = words_of(needle)
        if not wanted:
            return []                     # held by everything, so by nothing
        postings = [index[word] for word in set(wanted) if word in index]
        if not postings:
            return []
        locals_, held = np.unique(np.concatenate(postings), return_counts=True)
        shares = held / len(set(wanted))

        text_needle = str(needle).lower()
        ranked = []
        for local, share in zip(locals_.tolist(), shares.tolist()):
            row = seats.get(local)
            if row is None or share < self.cutoff:
                continue
            text = str(search.values[row]).lower()
            if exclusive and text == text_needle:
                continue
            # among equal shares the value that adds least: `alien` is inside
            # Alien and Alien 3, and the one that adds least is the one meant
            ranked.append((-share, len(text), row))
        return [(row, -negative) for negative, _length, row in sorted(ranked)[:self.k]]


class Semantic(Rule):
    """Nearness in a vector column: the cosine, clipped at zero.

    Below zero is not a weaker answer but the opposite direction, so it reads as
    no closeness rather than as a negative one. Several query vectors are
    several questions and a row answers whichever it answers best.

    Scored as one matmul against the type's unit block rather than a distance
    per row, which is what makes an exact search worth having before an
    approximate one: a few hundred thousand rows are milliseconds."""

    IDENTICAL = 1.0 - 1e-6

    def __init__(self, k=None, cutoff=0.0):
        self.k = k
        self.cutoff = cutoff

    def matches(self, search, needles, exclusive=False):
        block = search.graph.unit(search.type, search.stored) if search.stored else None
        if block is None:
            raise ValueError(
                f"{search.column!r} is not a vector column; "
                f"{search.type} has: "
                f"{', '.join(sorted(search.graph.vectors.get(search.type, {}))) or 'none'}")
        queries = _unit(needles)
        similarity = np.clip(block[search.local()] @ queries.T, 0.0, 1.0)
        answered = similarity.argmax(axis=1)
        best = similarity.max(axis=1)
        if exclusive:
            best = np.where(best >= self.IDENTICAL, 0.0, best)

        admitted = np.flatnonzero(best >= max(self.cutoff, np.nextafter(0.0, 1.0)))
        if self.k is not None and len(admitted) > self.k:
            keep = np.argpartition(-best[admitted], self.k)[:self.k]
            admitted = admitted[keep]
        return {int(row): (float(best[row]), _needle(needles, int(answered[row])))
                for row in admitted.tolist()}


def _unit(needles):
    """One query vector or several, as unit rows."""
    values = np.asarray(needles, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(lengths > 0, lengths, 1.0)


def _needle(needles, index):
    """Which query vector a row answered, as something a column can hold."""
    values = np.asarray(needles, dtype=np.float32)
    return index if values.ndim > 1 else 0


# What `like` measures with when nothing says otherwise. Characters rather than
# words, because a column the graph never indexed is the general case and the
# fallback should be the one that always works.
DEFAULT = Fuzzy


def default_rule():
    return DEFAULT()
