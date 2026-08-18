# Jerboas

**Query a knowledge graph like an ORM, rank the results like a recommender.**

Most graph libraries make you choose. Either you write a query that *filters* —
crisp, exact, everything-or-nothing — or you leave the query language behind and
score things yourself in Python. Jerboas starts from the idea that filtering and
ranking are the same operation at different temperatures, so both belong in the
query.

```python
import jerboas as jb
from jerboas import Node, Edge, Path, Score, Like, PageRank

g = jb.Graph(kg="data/example/example.kg",
             edges=["data/example/example.has_interact"],
             attrs=[f"data/example/example.{t}"
                    for t in ("song", "artist", "author", "genre")])

artist = Node("artist")
seeds = set(g.select(artist).where(Like(artist.name.is_in(["Golden"]), k=3)))

song, seed, path = Node("song"), Node(), Path()
g.select(song, Score()).where(
    path == [song, Edge(), seed],
    seed.is_in(seeds),
).rank(PageRank(to=seeds)).top(5)
```

```
Like("Golden") -> Golden Project, Golden Kids, Golden Collective

1.00  Broken Dreams     0.94  Empty Pulse       0.83  Lost Shadows
0.95  Distant Echo      0.93  Restless Lights
```

That runs on a fresh clone: the example graph ships with the repo.

## Membership is graded

A crisp `Condition` answers *yes* or *no*. `Like` answers *how much*, in `[0, 1]`,
and that single idea is what the library is built around.

```python
Like(person.name.is_in(["Quentin Tarantino"]))    # the person you meant
Like(person.name.is_in(["tarantino"]), k=3)       # the three closest
Like(movie.year < 1990, t=0.8)                    # a threshold that leaks
Like(movie.year == 1994, width=5)                 # equality with a tolerance
```

A set of strings is a **search box, not a filter**: `Like` admits the `k` values
closest to each needle rather than a region around them. So a fragment finds the
whole (`"tarantino"` → *Quentin Tarantino*), a typo still lands (`"George Lukas"`
→ *George Lucas*), and asking for nothing returns nothing. Admission and weight
share one measure, so the rows that come back are exactly the ones the ranking
would have put on top.

A `Like` in `where(...)` does **two jobs at once**. It admits a widened crisp
region — so the engine never chases a gaussian across the graph — and it
contributes its graded membership as a ranking weight, automatically. You write
it once; you do not repeat it in `rank(...)`.

That is what "non-deterministic-first" means in practice: a soft constraint
*rewards* matches rather than *excluding* mismatches.

## Results carry their meaning

Queries hand back `Key` values, not strings you have to dissect:

```python
for movie, score in ranked:
    movie.type            # "movie"
    movie.id              # 123              -- its position in the movie block
    movie.label           # 123              -- what names it outside the graph
    movie.attrs["title"]  # "Pulp Fiction"   -- typed at load, not text
    str(movie)            # "movie.123"
```

An id is a position and `label` is what identifies the node to anything outside
the graph — the same number, until a graph is loaded with `renumber=True`, where
`label` holds the id its source used instead.

Nothing guesses which column a human reads. A query that wants to ask one
question across types that spell it differently says so on the variable:

```python
readable = {"movie": "title", "person": "name", "genre": "name"}

Like(Node().alias(readable=readable).readable.is_in(names))
```

`alias` annotates the variable and hands it back, rather than returning a new
one — two `Node`s are two pattern variables, so a copy would quietly split the
pattern.

A `Path` comes back as alternating `Key` and `Rel`, and a `Rel` knows which way
it was walked — which turns an explanation into a projection instead of a
string-parsing exercise:

```
song.0  --performed_by->  artist.0
```

## One relation, two directions

There is no `directed_by_r`. Each edge is stored once, as an edge-labeled CSR
plus its transpose, and direction belongs to the traversal:

```python
person.directed_by.inverse     # the movies a person directed
Edge("has_genre").inverse      # inside a path pattern
Edge()                         # wildcard: any relation, either direction
```

The wildcard walking both ways is what lets a two-hop bridge close —
`[movie, Edge(), genre, Edge(), movie]` — without duplicating every edge in
memory to fake it.

## Edges carry a weight

A rating, a similarity, a confidence — every edge has one, defaulting to `1.0`
for the ones nobody scored. It lives in the file, so the graph holds all of it,
and *how much of it counts* is asked per query rather than decided once at load:

```python
watched = Edge("has_interact")

g.select(rec, Score()).where(
    path == [user, watched, rec],
    watched.score >= 3,              # or: Edge("has_interact", score=(3, 5))
).rank(Weight())                     # …or rank by it instead of filtering on it
```

A predicate on a score compiles the way a predicate on a node does: evaluated
once against every stored edge, handed to the engine as a mask. Filtering by
weight therefore costs what not filtering costs.

The stored score is an unbounded float, which is honest and useless to anything
that has to accumulate one. `.norm()` rescales it to `[0, 1]` **within its own
relation** — a 1-5 rating and a cosine similarity are both floats and mean
nothing to each other:

```python
watched.score.norm() >= 0.5      # half-way up, whatever this relation's scale is
rank(Weight(how="min"))          # a walk is as good as its weakest step
PageRank(weighted=True)          # a 5-star step carries more of the walker
MatrixFactorization(weighted=True)   # explicit feedback: the target is the rating
train(model, g)                  # each example weighted by its edge
train(model, g, where=[Edge("has_interact", score=(3, None))])   # …or not seen at all
```

That last one is where a threshold belongs. Dropping weak edges at load time
answers "is this good enough?" once, for every query and every model; asking it
of a training run answers it for that run, and leaves the graph able to say who
rated a film at all. Each marker constrains its own relation — the knowledge
graph is not judged by a rating scale it has nothing to do with.

An unweighted graph is the same graph: every weight is `1.0`, every
normalization is `1.0`, and every one of those reads exactly as it did before
weights existed.

## Aggregates ask about the matches

`node.rel.count()` is a fact about the graph: the same number whatever the query
asked. An aggregate is a fact about what *this* pattern matched.

```python
shared = Edge("has_tag", score=(0.9, None))
carried = shared.inverse                     # named: `.inverse` makes a new marker

g.select(rec, Score()).where(
    path == [seed, shared, tag, carried, rec],
    seed.is_in(watchlist), ~rec.is_in(watchlist),
).rank(Sum(carried.score)).top(10)
```

That is a whole recommender: not "does this share a tag?" but "how much of the
list is it?" — one shared tag is a coincidence, twenty is a taste. Ranking by the
strongest single match instead (`Max`) answers the other question, and answers it
differently, which is why the aggregate has to be said out loud.

`Sum`, `Count`, `Mean`, `Min`, `Max` take an edge marker's score, and `Count`
also takes a node of the pattern. **Selecting is grouping**: a result reached
five ways comes back once, folded. A `Path` in `select(...)` is evidence of a
match rather than part of it, so it never splits a group — you can rank by an
aggregate and still hand back one walk to explain each row with.

## The one rule

> **Methods are only the pipeline verbs** — `select`, `where`, `rank`, `top`,
> `groupby`, `using` — and they live only on `Graph`/`Query`. They are the only
> things that touch the graph.
> **Everything you pass to a method is a passive object**: it describes *what*
> you want, never *how*.

| Family | Interface | Objects | Passed to |
|---|---|---|---|
| **Reference** | `Ref` / `Expr` | `Node`, `Attr`, `Edge`, `EdgeScore`, `Path`, `Degree`, `Sum`/`Count`/… | `select(...)`, `rank(...)` |
| **Condition** | `Condition` | `Compare`, `Like`, `In`, `Has`, `Match`, `And`/`Or`/`Not` | `where(...)` |
| **Strategy** | `Strategy` | `PageRank`, `Connectivity`, `MatrixFactorization`, `Weight`, `Ascending`, `TransD`, … | `rank(...)` |
| **Engine** | `Engine` | `Default`, `Greedy` | `using(...)` |

Adding a matcher is a new `Condition`. Adding a ranker is a new `Strategy`.
Neither edits the `Query`: every object describes itself to the compiler.

The predicate surface follows Polars, so most of it is already familiar:

```python
node.year >= 1990         node.name.is_in([...])      node.year.is_between(a, b)
node.title.contains("x")  node.rel.count() >= 2       a & b,  a | b,  ~a
```

`&`, `|` and `~` — not `and`, `or`, `not`, which reduce their operands with
`bool()` and would keep only half of what you wrote. A Condition has no truth
value to give, so it raises rather than letting the query quietly return more
rows than it should. The same goes for `1990 <= node.year <= 2000`, which Python
expands into an `and`: say `node.year.is_between(1990, 2000)`.

Ranking scores rather than sorts, and a score means *more is better* — which a
degree implies and a column does not. `Ascending` says the other direction, for
anything scoreable:

```python
rank(node.rel.count())            rank(Ascending(movie.title))     # A to Z
rank(Descending(movie.year))      rank(Ascending(Sum(carried.score)))
```

Results are a `Sequence`, so the language handles them: `len`, `in`, `reversed`,
`.index`, unpacking and comprehensions all work, `q[:5]` *is* `top(5)` rather
than a full evaluation thrown away, and `np.asarray(q)` and
`pandas.DataFrame(q, columns=q.columns)` need no adapter. A `Graph` is a
container of nodes the same way: `len(g)`, `"movie.12" in g`, `g["movie.12"]`.

Refs have identity semantics on purpose — two `Node("movie")` values are two
different pattern variables — so reuse the same object across `select` and
`where`. Get it wrong and the query says so rather than quietly returning
everything.

## Embeddings are strategies

Fitting one needs torch, so the models are an optional extra and are imported on
demand — a base install stays importable without it.

```python
from jerboas import TransD, train               # pip install jerboas[torch]

model = train(TransD(factors=64), g, epochs=15, device="mps")
model.save("checkpoints/ml.transd.npz")
```

```python
g.select(rec, Score()).rank(
    TransD.load("checkpoints/ml.transd.npz", g, to=seeds)
).top(10)
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
rank(model.seeded(seed_keys))                                # per request
```

**A model is a strategy you can train.** There is no wrapper and no registry:
`TransD` subclasses `Strategy` exactly as `PageRank` does, so a fitted model goes
straight into `rank(...)`. One class holds the tables, the arithmetic, the
training and the ranking, and adding a model means writing that one class:

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
which hyperparameters, on a graph of what size.

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

On MovieLens (15 369 nodes, 127 k edges, of which 110 k pass the
cold-start use case's training filter) TransD at `factors=64` is 1.97 M parameters, 7.9 MB, and trains
in roughly a second per epoch on Apple MPS.

## Install

```bash
pip install -e .              # numpy + scipy
pip install -e '.[torch]'     # + training
pip install -e '.[api]'       # + the FastAPI example
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
the checkpoint after, and answers with an explanation drawn from the path that
connected each result to your seeds. It needs `pip install -e '.[api,torch]'`
and the MovieLens graph below.

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

27 523 nodes, 2.78 M edges, ~2.4 s to load. The tag genome is a dense matrix —
every film scores against every tag — so `build.py` takes 0.3 as the relevance
at which an edge starts existing, and a query narrows it from there:

```python
strong = Edge("has_tag", score=(0.95, None))

g.select(rec, Score(), path).where(
    path == [seed, strong, bridge, strong.inverse, rec],
    seed.title == "Blade Runner", ~(rec.title == "Blade Runner"),
).rank(Weight(how="min")).top(5)
```

```
1.00  Johnny Mnemonic   via dystopic future     1.00  Terminator, The  via dystopic future
1.00  Brazil            via dystopic future     1.00  Gattaca          via distopia
```

Half a second on 2.78 M edges, and the tag that joined them comes back with the
row rather than being reconstructed afterwards.

`usecase/genome` serves it: give it a watchlist and it answers with films that
belong beside it, ranked by how much of the watchlist's tag profile they carry.
It trains nothing — the affinity is already in the data.

```bash
./run.sh genome
curl -X POST localhost:8000/suggest -H 'content-type: application/json' \
     -d '{"watchlist": ["Blade Runner", "The Matrix", "Aliens"], "strength": 0.9}'
```

```
1.00  Terminator, The       dystopic future, future, cyborgs
0.86  Oblivion              dystopic future, sci-fi, futuristic
0.84  Empire Strikes Back   imdb top 250, destiny, science fiction
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
and every constraint on a node — its type, identity, attributes, degrees —
compiles into a single boolean mask. The engines' per-candidate test is
`mask[node]`.

Files are read by column, not by line. A chunk of an edge file becomes three
parallel columns with one `replace` and one `split` — two C loops over the whole
chunk — and the ids come from `dict.fromkeys`, which deduplicates in C and in
first-seen order at once. What is left in Python runs once per *distinct node*
rather than once per edge: 2.78 M edges load in 2.4 s, where the obvious loop
took 13.7 s.

```
jerboas/
  core.py         Ref / Expr / Condition / Strategy / Engine / Compiler
  graph.py        the data: integer ids, CSR adjacency, typed columns
  columns.py      typed, nullable attribute columns
  keys.py         Key / Rel -- what a query hands back
  refs.py         Node, Attr, Edge, Path, Degree
  conditions.py   Compare, Like, In, Has, Match, And/Or/Not
  query.py        Query + the compiler that builds admission masks
  ir.py           the neutral IR an Engine consumes
  engine.py       Default, Greedy
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
