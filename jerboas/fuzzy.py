"""Character similarity: the measure `rules.Fuzzy` is written in terms of.

The rule is the interface (see rules.py); this is one implementation of one
measure, kept apart because difflib's own shape -- indexing the needle once,
skipping the quadratic work under a cutoff -- is the whole of what makes it
usable on a column of any size.

A set of strings is a search box, not a filter. `frame.like(v.person.name,
["tarantino"])` admits the values closest to the needle rather than the ones
equal to it, so a fragment finds the whole, a typo still lands, and the empty
needle admits nothing rather than everything.

Admission and weight come from one measure, which is what keeps them from
disagreeing: the rows that come back are exactly the ones the ranking would have
put on top, and the measure stays on the frame as a `similarity` column instead
of quietly becoming a ranking term nobody wrote.
"""

import heapq
import re
from difflib import SequenceMatcher

# what separates one word from the next: everything that is not a letter or a
# digit, so `Wonderwall - Remastered` is two words and `Sgt. Pepper's` is three
_WORDS = re.compile(r"[^\w]+", re.UNICODE)


def words_of(text):
    """The words of one value, lowercased and in order."""
    return [word for word in _WORDS.split(str(text).lower()) if word]


def scorer(needle, cutoff=0.0):
    """A prepared measure for one needle: `text -> closeness in [0, 1]`.

    difflib indexes its *second* sequence, so the needle belongs there and gets
    indexed once instead of once per candidate; `real_quick_ratio` and
    `quick_ratio` are O(n) upper bounds that skip the quadratic work for anything
    that cannot reach `cutoff` anyway. Measured on a column of 12 649 names:
    5.6x, and not one score above the cutoff moved."""
    text_needle = str(needle).lower()
    if not text_needle:
        # contained in everything, which would make a blank search the broadest
        # one possible instead of the narrowest
        return lambda text: 0.0
    matcher = SequenceMatcher()
    matcher.set_seq2(text_needle)

    def closeness(text):
        if text_needle in text:
            return 1.0
        matcher.set_seq1(text)
        if matcher.real_quick_ratio() < cutoff or matcher.quick_ratio() < cutoff:
            return 0.0
        return matcher.ratio()
    return closeness


def closest(needle, texts, k, cutoff, exclusive=False):
    """The k rows closest to one needle, as [(row, similarity)].

    A needle that is literally present is already as close as anything can be, so
    the expensive comparison only runs when plain containment cannot fill k --
    the difference between a millisecond and a fifth of a second on a column of
    twelve thousand names.

    Shortest first among the containing ones: "alien" is inside Alien, Aliens,
    Alien 3 and Alien: Resurrection, and the one that adds least is the one that
    was meant. Taking them in column order instead let an exact match lose to
    whatever loaded first, which is a coin toss wearing the shape of a result."""
    text_needle = str(needle).lower()
    contained = sorted((len(text), row) for row, text in texts
                       if text_needle and text_needle in text
                       and not (exclusive and text == text_needle))
    if len(contained) >= k:
        return [(row, 1.0) for _length, row in contained[:k]]
    close = scorer(text_needle, cutoff)
    scored = ((close(text), row) for row, text in texts
              if not (exclusive and text == text_needle))
    return [(row, score) for score, row in heapq.nlargest(k, scored) if score >= cutoff]


def best(needles, texts, k, cutoff, exclusive=False):
    """The k closest rows to each needle, as {row: (closeness, needle)}.

    Folded: a row admitted by two needles is admitted once, keeping the higher
    closeness and the needle that earned it. Which needle that is, is the
    question a set of names asks that one name does not -- so it is answered
    rather than counted and thrown away.

    Two needles that judge a row *equally* well are a tie the measure cannot
    break, and it is not broken here either: the first one asked keeps it. That
    is arbitrary but it is at least stable, and the alternative -- admitting the
    row twice -- would make a set of names return more rows than it has
    answers."""
    found = {}
    for needle in needles:
        for row, score in closest(needle, texts, k, cutoff, exclusive):
            if score > found.get(row, (0.0, None))[0]:
                found[row] = (score, needle)
    return found
