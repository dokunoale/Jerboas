"""From names a person typed to the songs they meant.

A name typed by a person is a search box, not a key: titles are not unique, and
this dataset holds several masters of the same recording. So every name is
searched, the candidates of all of them land in one pool, and the pool resolves
itself (`resolve`).
"""

from collections.abc import Iterator

import polars as pl

import jerboas as jb
from jerboas import Words, reverse, v

# how many equally-titled songs to weigh against each other before picking one
TIES = 8

# how many performers a name written after a tab may mean
PERFORMERS = 4

# whether a performer put a candidate in the pool, rather than a title alone
PINNED = "pinned"


def asked_for(names: list[str]) -> Iterator[tuple[str, str | None]]:
    """What was typed, as (title, performer or None).

    A tab is the only structure the input has: `"Wonderwall\tOasis"` says which
    Wonderwall, and a name without one says nothing about the performer. It is
    there because titles are not unique and a name typed alone cannot say which
    recording it means -- so a caller who knows gets to say."""
    for name in names:
        title, _, performer = (name or "").partition("\t")
        title, performer = title.strip(), performer.strip()
        if title:
            yield title, (performer or None)


def resolve(graph: jb.Graph, names: list[str]) -> jb.Frame | None:
    """The songs you named, one each.

    `like` admits the closest stored titles rather than the equal one, and what
    it landed on comes back in the answer so a wrong guess is visible rather
    than silent.

    Every name is resolved in the same frame, which is what makes the choice a
    good one. Candidates arrive by two routes -- `_by_performer` for the names
    that named one, `_by_title` for every name -- and both land in one pool
    tagged with the name they answer (`v.seed.name.needle`) and with whether a
    performer put them there. Then:

      * **an explicit performer wins** where it found anything, and where it
        found nothing the name falls back to its titles. That is one filter over
        the pool rather than a special case, so both routes are always present
        and the data says which survives, per name.

      * **the rest resolve each other.** `coherent` keeps the combination that
        keeps the most company, with the pinned ones in that reckoning as
        anchors: told that one song is Oasis, the others lean towards Oasis.

    The sort is the fallback rather than the decision: with one name, or names
    with nothing in common, nothing is connected and the order stands.
    """
    wanted = list(asked_for(names))
    if not wanted:
        return None
    pool = jb.concat(_by_performer(graph, wanted),
                     _by_title(graph, [title for title, _performer in wanted]))
    return (pool
            # a performer that found something settles its own name and no other
            .filter(pl.col(PINNED) | ~pl.col(PINNED).any().over("asked"))
            .sort(["closeness", "seen"], descending=True)
            .coherent(by=v.asked, through=reverse("contains"))
            .select("seed"))


def _by_title(graph: jb.Graph, titles: list[str]) -> jb.Frame:
    """Every title's closest candidates, all of them in one query.

    `v.seed.name.needle` says which title each candidate answers, so what would
    be a loop over names is a column."""
    return (graph.nodes(seed="song")
            .filter(v.seed.name.like(titles, rule=Words(k=TIES)))
            .with_columns(asked=v.seed.name.needle.cast(pl.String),
                          closeness=v.seed.name.score,
                          seen=v.seed.contains.count(),
                          **{PINNED: pl.lit(False)})
            .select("seed", "asked", "closeness", "seen", PINNED))


def _by_performer(graph: jb.Graph, wanted: list[tuple[str, str | None]]) -> jb.Frame:
    """The candidates a performer allows, for the names that named one.

    The performer is not a filter over the titles that matched -- it narrows
    what is searched. Which matters: the exact-titled `Wonderwall`s are eight
    covers, and the Oasis recording, filed as `Wonderwall - Remastered`, is not
    among them at any pool size worth using. Asked of Oasis' songs instead, it
    is the only answer there is.

    One query for every pinned name: the performers are searched together and
    `v.artist.name.needle` says which one each is, so the titles that go with
    it are a join rather than an iteration -- all of them, since several names
    may pin the same performer."""
    pins = pl.DataFrame([(performer, title) for title, performer in wanted if performer],
                        schema={"who": pl.String, "asked": pl.String}, orient="row")
    if not len(pins):
        return _no_candidates(graph)
    people = (graph.nodes(artist="artist")
              .filter(v.artist.name.like(pins["who"].unique(maintain_order=True).to_list(),
                                         rule=Words(k=PERFORMERS)))
              .with_columns(who=v.artist.name.needle.cast(pl.String)))
    if not len(people):
        return _no_candidates(graph)
    # each performer found, once per title asked of them
    songs = (people.join(pins, on="who")
             .hop(seed=reverse("performed_by"))
             .attrs(seed="name"))
    return (songs
            .filter(pl.col("seed.name").str.to_lowercase()
                    .str.contains(pl.col("asked").str.to_lowercase(), literal=True))
            .with_columns(closeness=pl.lit(1.0), seen=v.seed.contains.count(),
                          **{PINNED: pl.lit(True)})
            .select("seed", "asked", "closeness", "seen", PINNED))


def _no_candidates(graph: jb.Graph) -> jb.Frame:
    """An empty pool shaped like the others, so a concat of it is a concat and
    not a special case."""
    return (graph.nodes(seed="song").head(0)
            .with_columns(asked=pl.lit(None, dtype=pl.String),
                          closeness=pl.lit(0.0), seen=pl.lit(0, dtype=pl.Int64),
                          **{PINNED: pl.lit(False)})
            .select("seed", "asked", "closeness", "seen", PINNED))


def describe(frame: jb.Frame, column: str) -> jb.Frame:
    """A column of songs as "title -- performer", which is the only form in
    which a song is identifiable: titles are not unique and this dataset holds
    several masters of the same recording."""
    return (frame.hop(artist="performed_by").attrs(**{column: "name"}, artist="name")
            if len(frame) else frame.attrs(**{column: "name"}))
