# Jerboas

**Query a knowledge graph like a dataframe, rank the results like a recommender.**

Most graph libraries make you choose. Either you write a query that *filters* —
crisp, exact, everything-or-nothing — or you leave the query language behind and
score things yourself in Python. Jerboas starts from the idea that filtering and
ranking are the same operation at different temperatures, so both belong in the
query — and that the natural shape for a query with a temperature is a table you
can look at.

```python
import jerboas as jb
from jerboas import v, PageRank

g = jb.Graph(kg="data/example/example.kg",
             edges=["data/example/example.has_interact"],
             attrs=[f"data/example/example.{t}"
                    for t in ("song", "artist", "author", "genre")],
             readable={"song": "name", "artist": "name",
                       "author": "name", "genre": "name"})

seeds = g.nodes("artist").labels("artist").like(v.artist.label, "Golden", k=3)

(g.nodes(seed=seeds).hop(to="song", type="song")
   .with_columns(score=PageRank(to=seeds).on("song"))
   .top(5).attrs(song="name"))
```

```
like("Golden") -> Golden Project, Golden Kids, Golden Collective

shape: (5, 6)
┌──────┬──────┬────────────┬───────────────┬──────────┬─────────────────┐
│ seed ┆ song ┆ song.score ┆ song.rel      ┆ score    ┆ song.name       │
╞══════╪══════╪════════════╪═══════════════╪══════════╪═════════════════╡
│ 101  ┆ 55   ┆ 1.0        ┆ ~performed_by ┆ 0.013566 ┆ Broken Dreams   │
│ 101  ┆ 85   ┆ 1.0        ┆ ~performed_by ┆ 0.013212 ┆ Distant Echo    │
│ 101  ┆ 3    ┆ 1.0        ┆ ~performed_by ┆ 0.013156 ┆ Empty Pulse     │
│ 101  ┆ 1    ┆ 1.0        ┆ ~performed_by ┆ 0.013076 ┆ Restless Lights │
│ 101  ┆ 26   ┆ 1.0        ┆ ~performed_by ┆ 0.012462 ┆ Lost Shadows    │
└──────┴──────┴────────────┴───────────────┴──────────┴─────────────────┘
```

That runs on a fresh clone: the example graph ships with the repo.

## Two objects, and the second is a dataframe

```python
print(g)                  # <Graph: 193 nodes in 5 types (song, artist, ...), 800 edges ...>
print(g.nodes("song"))    # shape: (100, 1) -- a table
```

`Graph` is the data structure: nodes as positions in typed blocks, edges as one
edge-labeled CSR plus its transpose, attributes as typed columns. It holds
everything and decides nothing.

`Frame` is what you ask it — a [polars](https://pola.rs) DataFrame that
remembers which of its columns hold nodes of which graph. It adds the four verbs
a table cannot get from being a table:

| verb | what it does |
|---|---|
| `hop` | one traversal: one result row per edge, and the edge's own columns beside it |
| `like` | graded membership over a text column — the search box |
| `attrs` | a stored attribute as a column |
| `labels` | the column a person reads, per the graph's `readable` map |

Everything else is polars, forwarded verb by verb: `filter`, `with_columns`,
`select`, `group_by` / `agg`, `sort`, `unique`, `join`, `head`, `top`. The list
is finite and documented rather than caught by `__getattr__`, and `.pl` hands
back the DataFrame for anything not on it. `.to_polars()`, `.to_pandas()` and
`np.asarray(frame)` need no adapter, because there is nothing to adapt.

## A variable is a column name

```python
from jerboas import v

v.rec              # the column "rec"
v.rec.year         # the column "rec.year", as a polars expression
v.rec.year >= 1990
```

That is the whole identity system. Two `v.rec` are the same variable because
they are the same string, so there is no rule to learn about when two
identical-looking references are one thing and when they are two — and a frame
prints the variables it is holding, because they are its headers.

`v.x` is sugar: strings work everywhere it does (`top(10, by="score")`), and
`col("rec.sum")` spells out a name that collides with a method.

## Membership is graded

A filter answers *yes* or *no*. `like` answers *how much*, in `[0, 1]`, and that
single idea is what the library is built around.

```python
g.nodes("person").labels("person").like(v.person.label, ["tarantino"])
```

```
┌────────┬───────────────────┬────────────┐
│ person ┆ person.label      ┆ similarity │
╞════════╪═══════════════════╪════════════╡
│ 1972   ┆ Quentin Tarantino ┆ 1.0        │
└────────┴───────────────────┴────────────┘
```

A set of strings is a **search box, not a filter**: `like` admits the `k` values
closest to each needle rather than a region around them. So a fragment finds the
whole (`"tarantino"` → *Quentin Tarantino*), a typo still lands (`"George Lukas"`
→ *George Lucas*), and asking for nothing returns nothing. Admission and weight
come from one measure, so the rows that come back are exactly the ones a ranking
would have put on top.

And the measure stays. `similarity` is an ordinary column: add it to a score,
sort by it, or ignore it — but it is *there*, in the table, instead of quietly
becoming a ranking term nobody wrote.

## One relation, two directions

There is no `directed_by_r`. Each edge is stored once, as an edge-labeled CSR
plus its transpose, and direction belongs to the traversal:

```python
.hop("directed_by", to="person")                  # movie -> person
.hop("directed_by", to="movie", reverse=True)     # person -> the films they directed
.hop(to="other")                                  # wildcard: any relation, either way
```

The wildcard walking both ways is what lets a two-hop bridge close —
`.hop(to="mid").hop(to="rec")` — without duplicating every edge in memory to
fake it. A wildcard hop also brings back a `<edge>.rel` column naming what it
walked, `~has_interact` for a step taken against the stored direction; a named
hop does not, because every row would carry the same word.

## Edges carry a weight, and a weight is a column

A rating, a similarity, a confidence — every edge has one, defaulting to `1.0`
for the ones nobody scored. It lives in the file, so the graph holds all of it,
and *how much of it counts* is asked per query:

```python
(g.nodes(user="user")
   .hop("has_interact", to="rec")
   .filter(v.has_interact.score >= 3))            # ...or rank by it instead
```

There is no score band, no admission mask, no threshold argument: the weight
came back from the hop as a column, and a column is filtered by filtering it.

The stored score is an unbounded float, which is honest and useless to anything
that has to accumulate one. `hop(..., norm=True)` rescales it to `[0, 1]`
**within its own relation** — a 1-5 rating and a cosine similarity are both
floats and mean nothing to each other.

```python
PageRank(weighted=True)                                  # a 5-star step carries more walker
train(model, g)                                          # each example weighted by its edge
train(model, g, where=g.edges("has_interact").filter(v.score >= 3))   # ...or not seen at all
```

That last one is where a threshold belongs. Dropping weak edges at load time
answers "is this good enough?" once, for every query and every model; handing a
training run a *frame of the edges it may see* answers it for that run, leaves
the graph able to say who rated a film at all, and makes "which edges" a question
with a visible answer — `len(where)` — rather than a marker object.

## Aggregates are a group_by

```python
(g.nodes(seed=watchlist)
   .hop("has_tag", to="tag", as_="shared", norm=True).filter(v.shared.score >= 0.9)
   .hop("has_tag", to="rec", reverse=True, as_="carried", norm=True)
   .filter(v.carried.score >= 0.9)
   .filter(~v.rec.is_in(watchlist))
   .group_by(v.rec).agg(score=v.carried.score.sum(),
                        via=v.tag.first())
   .top(10))
```

That is a whole recommender: not "does this share a tag?" but "how much of the
list is it?" — one shared tag is a coincidence, twenty is a taste. Ranking by the
strongest single match instead (`.max()`) answers the other question, and
answers it differently, which is why the aggregate has to be said out loud.

The grouping is said out loud too. There is no rule about which projections
group and which do not, no `Path` that is evidence-but-not-part-of-the-match:
`agg` keeps whatever you ask it for, so a group can carry one tag to explain
itself with alongside the sum that ranked it.

## The walk is the frame

There is no `Path` object, because the columns already are one:

```python
g.nodes(seed=seeds).hop(to="mid").hop(to="rec", type="movie")
# columns: seed, mid, mid.score, mid.rel, rec, rec.score, rec.rel
```

Explaining a result is a projection — `row["seed.label"]`, `row["rec.rel"]` —
rather than unpacking an alternating tuple of keys and relations. Two routes to
the same node are two frames, and `concat` puts them together; the column one
branch lacks comes back null, which is exactly what "reached the other way"
means.

```python
direct = g.nodes(seed=seeds).hop(to="rec", type="movie")
bridge = g.nodes(seed=seeds).hop(to="mid").hop(to="rec", type="movie")
jb.concat(direct, bridge)
```

## Ranking is a column, and the arithmetic is written down

A `Strategy` is the one thing a column cannot be on its own: a score computed
*from the graph* — a random walk, a factorization, an embedding. Sorting by a
stored value is `sort`, ranking by a matched edge's weight is a column, counting
the matches is `agg`; none of those is a strategy, and none of them needs to be.

```python
(frame
   .with_columns(pr=PageRank(to=seeds, weighted=True).on("rec").norm(),
                 kg=TransD.load(path, g, to=seeds).on("rec").norm())
   .with_columns(score=0.6 * v.kg + 0.4 * v.pr)
   .top(10))
```

`on(...)` names the columns the strategy reads. The first is what is being
scored; the rest are context — the user whose taste it is, the seed the row was
reached from. Naming them is the point:

```python
MatrixFactorization().on("rec", "user")     # this user's affinity for this film
MatrixFactorization(user=who).on("rec")     # one person's, for the whole frame
TransD.load(path, g).on("rec", "seed")      # each row against its own seed
```

Nothing is combined behind your back. Two signals used to be min-max normalized
and averaged inside `rank(...)`; now `.norm()` is written where it happens, the
weights are numbers you chose, and every intermediate signal is a column you can
print and sort by on its own.

## Results carry their meaning

A frame holds integer node ids, because that is what indexes an array. When you
want the node itself, ask:

```python
frame.keys("rec")            # [Key, ...] -- .type, .id, .label, .attrs
frame.ids("rec")             # the raw int32 array
frame.attrs(rec="title")     # a column, gathered from the type's block
frame.labels("rec")          # the column the graph's `readable` map names
```

Nothing guesses which column a human reads. `readable={"movie": "title",
"person": "name"}` is declared once, with the data, because it is a fact about
the dataset and not about any one query — and a type that declares none falls
back to its identity rather than to a guess.

## Embeddings are strategies

Fitting one needs torch, so the models are an optional extra and are imported on
demand — a base install stays importable without it.

```python
from jerboas import TransD, train               # pip install jerboas[torch]

model = train(TransD(factors=64), g, epochs=15, device="mps")
model.save("checkpoints/ml.transd.npz")
```

```python
frame.with_columns(score=TransD.load("checkpoints/ml.transd.npz", g, to=seeds).on("rec"))
```

Naming no relation asks for the most plausible edge of *any* kind, in either
direction — link prediction without specifying what link. That is what a mixed
seed set needs: a person reaches a film through `directed_by` read backwards, a
genre through `has_genre`, a user through `has_interact` forwards, and each seed
finds its own. Pass `relation=` to ask the narrower question.

A service loads a checkpoint once and re-aims it per request, since rebinding is
linear in the graph and choosing seeds is not:

```python
model = TransD.load("checkpoints/ml.transd.npz", graph)     # once, at startup
...
model.seeded(seed_keys).on("rec")                            # per request
```

**A model is a strategy you can train.** There is no wrapper and no registry:
`TransD` subclasses `Strategy` exactly as `PageRank` does, so a fitted model goes
straight into a column. One class holds the tables, the arithmetic, the training
and the ranking, and adding a model means writing that one class:

```python
class TransE(Translational):
    name = "transe"
    tables = (("entity", NODE), ("relation", RELATION))

    def plausibility(self, head, relation, tail):
        return self.norm(self.get("entity", head)
                         + self.get("relation", relation)
                         - self.get("entity", tail))
```

`plausibility` is written with the operators numpy and torch spell the same way,
so those three lines serve both the gradient step and the query — `get` returns
an `nn.Embedding` lookup while fitting and an array row once loaded.

Checkpoints are compressed `.npz` under `checkpoints/`, and they are **inert**:
every array is a native numpy dtype, so they load with `allow_pickle=False`. A
pickled `.npz` is executable code wearing a data extension; these are not. Each
one also records its own provenance — when it was fitted, for how long, with
which hyperparameters, on a graph of what size, and over which edges.

Two details that matter more than they look:

**Checkpoints rebind by name.** A node's integer id comes from load order, so a
checkpoint keyed by position would keep loading after you rebuild the graph and
silently score the wrong entities. Identity — type names, relation names, and one
attribute per node — is stored beside the weights and resolved on load.

*Which* attribute is the checkpoint's to name. It defaults to the node's id, and
where the id is the graph's own numbering rather than the world's, the durable
identifier is a column like any other and the checkpoint says so:

```python
model.save("checkpoints/spotify.transd.npz", alias="uri")
TransD.load("checkpoints/spotify.transd.npz", graph)     # reads the alias back
```

**Negative sampling is type-aware.** Corrupting a triple with a uniformly random
entity yields a type-wrong tail 99.8% of the time on MovieLens, so the model can
minimise its loss by learning to tell types apart instead of learning the
relation. Jerboas draws the corruption from the true endpoint's own type block,
so it has to learn something real to score it lower.

On MovieLens (15 369 nodes, 127 k edges, of which 110 k pass the cold-start use
case's training filter) TransD at `factors=64` is 1.97 M parameters, 7.9 MB, and
trains in roughly a second per epoch on Apple MPS.

## Install

```bash
pip install -e .              # numpy + polars + scipy
pip install -e '.[torch]'     # + training
pip install -e '.[api]'       # + the FastAPI examples
pip install -e '.[pandas]'    # + .to_pandas()
pip install -e '.[dev]'       # + pytest
```

## Data

`data/example/` is a small synthetic graph — invented songs, artists and
genres — and ships with the repo, so everything above runs immediately.

`usecase/` holds the services built on the library, one directory each, and
`run.sh` serves one of them in a container named after it — so with a local DNS
domain registered it answers at `<usecase>.<domain>` and several can run at once
(see [usecase/README.md](usecase/README.md)).

```bash
./run.sh coldstart              # -> http://coldstart.test:8000
```

`coldstart` is a FastAPI recommender with no user node: name people, genres or
films you like and it expands from those. It fits TransD on first boot, reuses
the checkpoint after, and answers with an explanation read off the columns of
the walk that connected each result to your seeds. It needs
`pip install -e '.[api,torch]'` and the MovieLens graph below.

```
0.898  Pulp Fiction          directed by Quentin Tarantino
0.892  Reservoir Dogs        directed by Quentin Tarantino
0.786  From Dusk Till Dawn   written by Quentin Tarantino
0.725  True Romance          written by Quentin Tarantino
0.720  Die Hard              acted in Bruce Willis
```

50 ms a request, graph and checkpoint held in memory.

The MovieLens graph used by that use case and the benchmarks is **not** included:
GroupLens' usage licence states that "the user may not redistribute the data
without separate permission", and the IMDb-derived files are non-commercial-use
only. Fetch [ml-100k](https://grouplens.org/datasets/movielens/100k/) and the
[IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)
yourself, and lay them out as:

```
ml.kg              head <TAB> relation <TAB> tail [<TAB> score]
ml.<relation>      a header row, then source <TAB> target [<TAB> score]
ml.<type>          a header row of column names, first column the id
```

The suffix names the thing: `ml.has_interact` is the `has_interact` relation,
`ml.movie` the `movie` type. A relation file is where a weighted relation goes —
`user.196  movie.242  3` — and the score column may be left out, in which case
every edge weighs 1.

**An id is a position.** Ids are dense integers `0..n-1` within a type, so
`song.42` is the 42nd song and resolving a name is arithmetic — no translation
table, no string per node, and no dependence on the order the file happens to
list things in. The loader refuses a gap, a repeat, and anything that is not a
non-negative integer, because each of them is a bug in whatever wrote the file
and none of them is discoverable once the graph is built.

An identifier that belongs to the world rather than to the graph — an IMDb code,
a Spotify uri — is an ordinary column beside the position, the way `ml.person`
holds `imdb` and `spotify.song` holds `uri`. A query reaches it as it reaches
any attribute, and a checkpoint can key on it (see `alias`, above) so that
weights fitted on one build rebind onto another.

Data that does not follow the format yet can be converted at load instead of by
a build script:

```python
jb.Graph(kg=..., edges=[...], renumber=True)
```

Each type's ids are sorted — numbers as numbers — and numbered from zero, and
the id the source used is kept as that type's `label` column. The result is
exactly what a conforming file would have produced, so nothing downstream knows
the difference. The one thing to know is that the positions move whenever the
node set does, so a checkpoint fitted on such a graph must key on the source id:
`model.save(path, alias="label")`.

Any graph in that shape works — nothing in the library is MovieLens-specific.

`data/genome/` is the second one, and the reason weights exist. Drop the
[ml-latest](https://grouplens.org/datasets/movielens/latest/) export into
`data/genome/legacy/` and run:

```bash
python3 data/genome/build.py            # ~40s, writes genome.* beside legacy/
```

Four node types and three relations, two of them weighted by something the data
measured rather than by something a loader decided:

```
genome.kg              movie has_genre genre                    35 217 edges
genome.has_tag         movie -> tag,  relevance 0.3 … 1.0     1 699 936 edges
genome.has_interact    user  -> movie, rating   0.5 … 5.0     1 042 519 edges
genome.movie / .tag / .genre        16 376 films, 1 128 tags, 19 genres
```

27 523 nodes, 2.78 M edges, ~1.1 s to load. The tag genome is a dense matrix —
every film scores against every tag — so `build.py` takes 0.3 as the relevance
at which an edge starts existing, and a query narrows it from there:

```python
(g.nodes(seed="movie").attrs(seed="title").filter(v.seed.title == "Blade Runner")
   .hop("has_tag", to="tag", as_="strong", norm=True).filter(v.strong.score >= 0.95)
   .hop("has_tag", to="rec", reverse=True, as_="carried", norm=True)
   .filter(v.carried.score >= 0.95).filter(v.rec != v.seed)
   .group_by(v.rec).agg(score=v.carried.score.sum(), via=v.tag.first())
   .top(4).attrs(rec="title").labels("via"))
```

```
┌────────────────────┬──────────┬─────────────────┐
│ rec.title          ┆ score    ┆ via.label       │
╞════════════════════╪══════════╪═════════════════╡
│ Oblivion           ┆ 8.865    ┆ dystopic future │
│ Matrix, The        ┆ 7.856071 ┆ cyberpunk       │
│ Fifth Element, The ┆ 7.8075   ┆ future          │
│ Gattaca            ┆ 6.946071 ┆ distopia        │
└────────────────────┴──────────┴─────────────────┘
```

80 ms on 1.7 M edges, and the tag that joined them comes back in the row rather
than being reconstructed afterwards.

`usecase/genome` serves it: give it a watchlist and it answers with films that
belong beside it, ranked by how much of the watchlist's tag profile they carry.
It trains nothing — the affinity is already in the data — and there is no ranking
object anywhere in the file: two hops, a filter and an aggregate.

```bash
./run.sh genome
curl -X POST localhost:8000/suggest -H 'content-type: application/json' \
     -d '{"watchlist": ["Blade Runner", "The Matrix", "Aliens"], "strength": 0.9}'
```

```
32.2  Oblivion                    dystopic future, sci-fi, futuristic
30.3  Terminator, The             dystopic future, future, cyborgs
27.7  Interstellar                space, scifi, science fiction
26.8  Terminator 2: Judgment Day  future, cyborgs, scifi
```

`strength` is the request deciding how strongly a tag must apply before it
counts — the load-time filter that is no longer a load-time filter.

The build samples users (10 000 of 330 766) and keeps each sampled profile
whole; `--users`, `--relevance` and `--seed` move those lines. Tag genome data
carries its own citation requirement — Vig, Sen and Riedl, *The Tag Genome*,
TiiS 2012.

## How it is put together

Nodes are integers in contiguous per-type blocks, which is what makes the
universe an array rather than a dictionary: `keys_by_type` becomes a slice, an
embedding table is one `(n_nodes, factors)` matrix a strategy indexes directly,
and reading an attribute is a gather at `id - block_start` rather than a join.

**A hop is a gather, not a join.** The CSR already indexes what a join would
have to build: node *i*'s edges live in `indices[indptr[i]:indptr[i+1]]`,
contiguously and sorted by relation. So expanding a column of nodes is a
concatenation of slices, which numpy does with no Python loop given the bounds
as arrays — and the row index that comes back is what stitches the result onto
the frame it came from. A named relation narrows each slice to a contiguous
sub-range, counted once per relation and memoized.

Everything else is a table operation, and therefore not ours: the anti-join that
drops what a user has already seen, the `unique` that folds two routes to one
node, the `group_by` that ranks by the whole pattern. Those used to be a
backtracking search, a Python loop over result rows, and a scope stack.

Files are read by column, not by line. A chunk of an edge file becomes three
parallel columns with one `replace` and one `split` — two C loops over the whole
chunk — and the ids come from `dict.fromkeys`, which deduplicates in C and in
first-seen order at once. What is left in Python runs once per *distinct node*
rather than once per edge: 2.78 M edges load in ~1.1 s.

```
jerboas/
  core.py         Strategy, and the Signal that aims one at columns
  graph.py        the data: integer ids, CSR adjacency, typed columns, nodes()/edges()
  frame.py        the query: a polars frame that knows its graph
  traverse.py     one hop, as a gather over CSR slices
  expr.py         v / col -- a variable is a column name
  fuzzy.py        graded membership over a text column
  columns.py      typed, nullable attribute columns
  keys.py         Key -- a node, outside the frame
  checkpoint.py   storing a trained model, and rebinding it by name
  strategies/     the ranking family
  models/         embeddings: strategies you train -- the only place torch lives
```

## Tests

```bash
pytest
```

The torch-dependent tests skip when the extra is not installed.

## License

Apache-2.0. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
attributions — the TransD and TransE formulations were adapted from
[hopwise](https://github.com/tail-unica/hopwise) (MIT).
