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

seeds = g.nodes(artist="artist").filter(v.artist.label.like("Golden", k=3))

(g.nodes(seed=seeds).hop(song="~performed_by")
   .with_columns(score=PageRank(to=seeds).on("song"))
   .top(5).labels("song"))
```

```
like("Golden") -> Golden Project, Golden Kids, Golden Collective

shape: (5, 4)
┌──────┬──────┬──────────┬─────────────────┐
│ seed ┆ song ┆ score    ┆ song.label      │
╞══════╪══════╪══════════╪═════════════════╡
│ 101  ┆ 55   ┆ 0.013566 ┆ Broken Dreams   │
│ 101  ┆ 85   ┆ 0.013212 ┆ Distant Echo    │
│ 101  ┆ 3    ┆ 0.013156 ┆ Empty Pulse     │
│ 101  ┆ 1    ┆ 0.013076 ┆ Restless Lights │
│ 101  ┆ 26   ┆ 0.012462 ┆ Lost Shadows    │
└──────┴──────┴──────────┴─────────────────┘
```

That runs on a fresh clone: the example graph ships with the repo.

## What it is for

Jerboas is a retrieval layer for graphs, meant to be **general** and
**model-aware**: the ranking is not something you bolt on after the query, it is
part of it. One source and one step for retrieval *and* recommendation, with the
models trained through the same library that serves them — so a graph can back a
recommender, a graph RAG, an ordinary RAG whose retrieval and reasoning happen
to run on a graph, or an LLM reasoning over one, without a different tool for
each.

That is why the query is a dataframe and not a language of its own: what comes
out has to feed the rest of an ML stack, and what goes in has to be composable
with it.

## The frame is a view of a dataframe that is already there

The mental model, and everything else follows from it: a graph *is* a dataframe
— one column for every position you could name in a walk, one row for every walk
that exists, and beside each column every fact derivable about it. That frame is
never built; it could not be. What you hold is a view of it, and there are only
two things you ever do:

- **name** something — `hop`, `attrs`, `labels`, `with_columns`. A hop does not
  add data; it makes visible a column that was always in the walk space, and the
  rows "multiply" only because projecting onto one column had folded them.
  `unique` folds them back.
- **restrict** it — `filter`, `top`, `like`. Fewer rows, never fewer facts.

From which the rule about columns, which is otherwise arbitrary:

> **A column is in the frame if and only if you named it.**

So `filter` never widens the frame — what it read to decide is taken off again —
and `select` never narrows what it was asked to produce. And so a hop adds one
column, not three: what the step *measured* is an attribute of the column it
revealed, not a column of its own (see below).

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
| `hop` | one traversal: one result row per edge |
| `paths` | several lengths of walk in one frame, folded between the steps |
| `attrs` | a stored attribute as a column |
| `labels` | the column a person reads, per the graph's `readable` map |

Four, and no more, because everything else a graph can be asked is a *condition*
— and conditions go where conditions go:

```python
.filter(v.movie.year >= 1990)                 # an attribute
.filter(v.movie.has_genre.count() >= 2)       # a relation's arity
.filter(v.movie.directed_by.is_in(people))    # an edge exists
.filter(v.movie.label.like("tarantino"))      # graded membership
.filter(v.rec.type == "movie")                # the type of a node
```

Everything else is polars, forwarded verb by verb: `filter`, `with_columns`,
`select`, `group_by` / `agg`, `sort`, `unique`, `join`, `head`, `top`. The list
is finite and documented rather than caught by `__getattr__`, and `.pl` hands
back the DataFrame for anything not on it. `.to_polars()`, `.to_pandas()` and
`np.asarray(frame)` need no adapter, because there is nothing to adapt.

## A variable is a column name, resolved late

```python
from jerboas import v

v.rec              # the column "rec"
v.rec.year         # the column "rec.year"
v.rec.year >= 1990
```

That is the whole identity system. Two `v.rec` are the same variable because
they are the same string, so there is no rule to learn about when two
identical-looking references are one thing and when they are two — and a frame
prints the variables it is holding, because they are its headers.

What makes it more than sugar is *when* the name is decided. Nothing becomes a
polars expression where it is written: `v.movie.has_genre >= 2` is a little tree
of names, and the Frame — the only thing that knows the graph — resolves it. A
name resolves in this order, and the order is the whole rule:

1. **a column the frame already has**
2. **an attribute of that variable's type** — read out of the graph, on demand
3. **a relation of the graph** — arity with `.count()`, existence with `.is_in(...)`

```python
.filter(v.movie.year >= 1990)                 # no .attrs() first: it is read
.filter(v.movie.has_genre.count() >= 2)       # a relation's arity
.filter(v.movie.directed_by.is_in(people))    # the edge exists, unexpanded
```

A filter filters rows, not columns: what it read to decide is taken off again,
so the frame's shape does not change under it. `.attrs(movie="year")` is how a
column stays.

When a type has an attribute named like a relation, the tie is refused rather
than guessed, and `v.movie.attr.knows` / `v.movie.rel.knows` say which. `.expr`
drops out to raw polars — the column by that exact name, unresolved — and plain
strings still work wherever a name is wanted (`top(10, by="score")`).

**A name is never shadowed by a method.** Attribute access always grows the
path, so `v.person.name` is the person's name and `v.movie.count` is a column
called `movie.count`. What would be a method elsewhere is reached by *calling*
the name instead:

```python
v.person.name        # the column person.name
v.rec.count()        # how many rec, in an aggregate
v.carried.score.sum()
```

Without that rule a method on the class would win the attribute lookup and
`v.person.name` would silently mean something else — which is the worst way for
a query language to be wrong, since it returns an answer.

## Nearness is the same idea, over vectors

A column of lists in an attributes frame is a **vector**, not an attribute — the
dtype says which — kept as one `(n, d)` float32 block per type. What a query asks
of one is nearness, which is graded membership again:

```python
g.nodes(chunk="chunk").filter(v.chunk.embedding.near(query, k=50))
                      .sort(v.chunk.embedding.score, descending=True)
```

Same three properties as `like`: it admits the k nearest rather than a region,
the cosine becomes the column's confidence, and the two cannot disagree because
they are one computation. Several query vectors are several questions, and a row
answers whichever it answers best. A cosine below zero is not a weaker answer but
the opposite direction, so it reads as no confidence rather than a negative one.

The search is exact — a matmul against a few hundred thousand rows is
milliseconds, and there is nothing to be wrong about. An index belongs behind the
same expression later, where "approximate" shows up as a lower confidence rather
than as a different API.

## Membership is graded, and the measure is kept

A filter answers *yes* or *no*. `like` answers *how much*, in `[0, 1]`, and that
single idea is what the library is built around.

```python
g.nodes("person").filter(v.person.label.like("tarantino"))
```

A set of strings is a **search box, not a filter**: `like` admits the `k` values
closest to each needle rather than a region around them. So a fragment finds the
whole (`"tarantino"` → *Quentin Tarantino*), a typo still lands (`"George Lukas"`
→ *George Lucas*), and asking for nothing returns nothing.

It is a condition like any other, so it goes in `filter` — but the measure that
decided admission is not thrown away. It becomes the **confidence of the column
it judged**:

```python
(g.nodes("person")
   .filter(v.person.label.like("tarantino", k=3))
   .with_columns(closeness=v.person.label.score)
   .sort(v.person.label.score, descending=True))
```

Admission and weight are one computation, so they cannot disagree — the rows
that come back are exactly the ones a ranking would have put on top.

## Every column has a confidence

That is not a special case for `like`. **Every column carries a `[0, 1]`
confidence per row**, and `v.A.score` reads it whatever produced `A`:

| what produced the column | what its confidence is |
|---|---|
| `hop` | the weight of the edge that revealed it, on that relation's own scale |
| `like` | how close the value was to what you asked for |
| anything else | `1.0` — nothing put it in doubt |

```python
.hop(tag="has_tag").filter(v.tag.score >= 0.9)     # the edge's relevance
.filter(v.person.label.like("tarantino"))
 .sort(v.person.label.score, descending=True)      # the same reading
```

A column also remembers **how** it was reached — `v.tag.via` is the relation the
hop walked, `~has_interact` for a step taken against the stored direction;
`v.person.label.needle` is which of the things you searched for a row is an
answer to; and `v.rec.type` is the node type its ids fall in.

That last pair is what lets a set of names resolve **each other**. `coherent`
keeps one candidate per name — the combination that keeps the most company:

```python
(g.nodes(seed="song")
   .filter(v.seed.name.like(["Wonderwall", "Come Pick Me Up"], k=8))
   .with_columns(asked=v.seed.name.needle, seen=v.seed.contains.count())
   .sort("seen", descending=True)                  # the fallback, when nothing connects
   .coherent(by=v.asked, through=reverse("contains")))
```

A name alone has only its own popularity to go on. A set of them has more: two
songs keep company when a playlist holds both, so `Wonderwall` beside
`Champagne Supernova` is Oasis and beside `Come Pick Me Up` is Ryan Adams —
neither being the more popular in the abstract. Handing back the titles of real
playlists and counting how many resolve to the songs those playlists held:
**100% against 66.7%** at three titles, **98.4% against 81.2%** at sixteen.

None of these is a column. They are attributes of one, kept in shadow
columns polars keeps aligned for free, hidden from `columns` and from `print`,
and **not allocated at all when they say the same thing about every row**: on an
unweighted graph, confidence costs nothing. They follow their column through
`rename`, `select` and `drop`, because that is what being an attribute of it
means.

Folding rows folds their confidence, by the mean of what went in:

```python
.group_by(v.rec).agg(...)                       # mean, the default
.group_by(v.rec, confidence="min").agg(...)     # as good as its weakest member
.group_by(v.rec, confidence=None).agg(...)      # forget it
```

`"max"`, `"product"` and a callable are the rest. Only the grouped columns keep
one: inventing a confidence for a number the aggregation just made up would be
inventing one for something nobody measured.

That per-relation scale is not a detail. A 1-5 rating and a cosine similarity
are both floats and mean nothing to each other, so a confidence is min-maxed
*within its own relation* — which is what makes it a quantity you can compare,
threshold and sum. The raw stored weight is still there, in `g.edges(...)`,
where it is what it is.

## One relation, two directions

There is no `directed_by_r`. Each edge is stored once, as an edge-labeled CSR
plus its transpose, and direction belongs to the traversal:

```python
.hop(person="directed_by")           # movie -> person
.hop(movie="~directed_by")           # person -> the films they directed
.hop(other=())                       # any relation, either way
```

The keyword is the name of the new column and its value is the relation walked
to fill it — the same shape as `g.nodes(rec="movie")`, where the keyword names
and the value says what. That is the answer to "what is `rec`": it is `AS rec`,
not `TO rec`.

`~name` is that relation read backwards — the spelling `v.x.via` prints, so what
you write is what you later read. A step may also be a collection, meaning "any
of these", and the **empty** collection means any relation at all: no constraint
on the relation is the empty set of constraints.

```python
.hop(step=("has_genre", "~directed_by"))     # either, at this step
```

The prefix is spelling rather than syntax, and `reverse()` says the same thing
without it — worth preferring in code meant to last, since the character could
change and `~` already means `not` in a predicate:

```python
.hop(movie=reverse("directed_by"))           # == "~directed_by"
reverse(reverse("x")) == "x"                 # and it maps over a collection
```

The empty step walking both ways is what lets a two-hop bridge close without
duplicating every edge in memory to fake it. Which relation it walked comes back
as `v.mid.via`.

### Every argument is a step

```python
.hop((), rec="has_genre")            # any relation, then has_genre
.hop(mid=(), rec=())                 # two steps, both kept
```

A **keyword** names the column the step's arrivals are kept in. A **positional**
step is walked and not kept — and that is what lets it be *folded*: two routes
that meet at an unnamed intermediate carry identical rows onward, and a row that
differs only where nothing was named is not a different row. The memoized
sub-path search the old engine needed is here a consequence of not having given
something a name.

The last step must be named, because where the walk ends is what the frame
holds; Python already requires positional arguments to come first, so the rule
costs nothing to obey.

Walking leaves from the **rightmost column of nodes**. To leave from another,
`select` it and `join` the result back — that is what working on a dataframe is
for, and it is why there is no `from=`.

Two lengths are two frames:

```python
direct = seeds.hop(rec=())
bridge = seeds.hop((), rec=())
jb.concat(direct.with_columns(hops=pl.lit(1)),
          bridge.with_columns(hops=pl.lit(2)))
```

### What a hop actually does

A hop is a gather, and it is worth seeing once, because everything about its
cost follows from it. Say the frame holds three artists, and the step is
`performed_by` read backwards:

```
frame:      artist = [100, 101, 102]

bounds:     lo = [500, 508, 513]      where each one's edges start in the CSR
            hi = [508, 513, 518]      and where they end
counts:          [  8,   5,   5]

gather:     rows    = [0 0 0 0 0 0 0 0  1 1 1 1 1  2 2 2 2 2]
            targets = [0 20 30 31 33 52 66 89  1 3 26 55 85  2 15 25 64 92]
```

`rows` is the whole trick: it says which row of the *old* frame each result came
from, so the new frame is `frame[rows]` with the target and the edge's weight
added beside it. No key, no join, no hash table — the CSR was already the index a
join would have had to build.

Two consequences worth knowing:

**A hop is an inner join.** A node with no matching edge contributes no rows and
simply drops out. A node with eight contributes eight, so a frame grows by the
degree of what it walks — which is why `unique` between two hops matters: two
users who watched the same film reach the same neighbours, and carrying that row
twice is what the old engine's memoized sub-path search existed to avoid.

**The walk is the cheap half.** The expensive half is `frame[rows]`, which drags
every column the frame already has into every new row. So `select` away what the
next step does not need — and write the filter, because a hop does not build its
rows until something needs them:

```python
frame.hop(rec="~has_genre").filter(v.rec.year >= 1990)
```

The predicate reaches the Frame before it reaches polars, and everything it
reads is about the node just reached, so it is applied to the arrays the walk
produced rather than to the rows they would have become. This is the compiler's
old admission mask, obtained by writing an ordinary filter — and it pays in
proportion to what the frame is carrying: on a MovieLens hop it is **24% faster
on a 14-column frame** and a wash on a 2-column one, which is the same cost
model read from the other side.

This is why `hop` has no `where=`, no `type=` and no `norm=`: each of those was
a condition wearing a parameter's clothes, and a condition written as one is
both clearer and no slower. What is left is steps, and nothing else.

There is deliberately **no** parameter for the edge's weight either. It was
written, measured and removed: filtering the weight array before building the
frame beats polars' own comparison only below about 1% selectivity, and costs
twice as much at 39%. A knob whose right setting requires knowing the
selectivity curve is worse than no knob — so a weight stays what it is, the
column's confidence, and `.filter(v.rec.score >= 0.9)` after the hop is both the
spelling and the fast path (it is pushed too, being about the step).

## What a training run may see

A rating, a similarity, a confidence — every edge has one, defaulting to `1.0`
for the ones nobody scored. It lives in the file, so the graph holds all of it,
and *how much of it counts* is asked per query: after a hop it is the reached
column's confidence, and `g.edges(...)` hands back the raw table when the raw
number is the quantity you mean.

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
   .hop(tag="has_tag").filter(v.tag.score >= 0.9)
   .hop(rec="~has_tag").filter(v.rec.score >= 0.9)
   .filter(~v.rec.is_in(watchlist))
   .group_by(v.rec).agg(score=v.rec.score.sum(),
                        through=v.tag.first())
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
g.nodes(seed=seeds).hop(mid=(), rec=()).filter(v.rec.type == "movie")
# columns: seed, mid, rec   -- and v.mid.via, v.rec.score, ... on each
```

Explaining a result is a projection — `row["seed.label"]`, `v.rec.via` — rather
than unpacking an alternating tuple of keys and relations. Two routes to the
same node are two frames, and `concat` puts them together; the column one branch
lacks comes back null, which is exactly what "reached the other way" means. Or,
in one verb that folds between the steps, `paths` (below).

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

A strategy's score is an expression like any other, so it composes with
arithmetic and with columns: `0.7 * kg.norm() + 0.3 * v.similarity` is a
sentence, not a special case.

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

## Breaking change in 0.2

0.2 replaces the query API. `Graph` and the data format are unchanged, and so
are the models and their checkpoints; everything between `g` and a result is
different. `select/where/rank/top`, `Node`, `Edge`, `Path`, `Like`, `Has`, the
`Condition`/`Ref`/`Engine` families and the pluggable engines are gone, and
what replaces them is above. There is no compatibility layer: 0.1 queries do
not run.

| 0.1 | 0.2 |
|---|---|
| `g.select(rec).where(...)` | `g.nodes(rec="movie").filter(...)` |
| `Node("movie")`, `Node()` | a column name |
| `path == [a, Edge(), b]` | `.hop(b=())` |
| `Like(node.name.is_in(x))` | `.filter(v.node.name.like(x))` |
| `Has(a, "r", b)` / `~Has(...)` | `.filter(v.a.r.is_in(b))` |
| `node.rel.count()` | `v.node.rel.count()` |
| `rank(a, b)` | `.with_columns(score=0.6 * a.norm() + 0.4 * b.norm())` |
| `Sum(edge.score)` | `.group_by(...).agg(score=v.x.score.sum())` |
| `Score()` projection | the `score` column |
| `Weight(how="min")` | `v.x.score`, and arithmetic |
| `using(Greedy(k))` | `.top(k, by=..., over=...)` between two hops |

## A graph out of anything polars reads

```python
Graph.from_frames({"has_interact": pl.read_parquet("ratings.parquet")},
                  attrs={"movie": movies, "chunk": chunks},
                  source=("user", "user_id"), target=("movie", "movie_id"),
                  score="rating")
```

A column may name nodes two ways: as source keys — `movie.123`, the format the
files use — or as a `(type, column)` pair, a column of ids and the type they
belong to. The second is what data from anywhere else looks like, with the type
in the schema rather than in the value.

Everything downstream is the file loader's, ids-are-positions included, and a
graph is complete when it exists — which is why this takes what it needs in one
call rather than being built up.

## When a walk is too big to take whole

A hop expands a frame by the degree of what it walks, and the expansion happens
before any filter can reduce it. On a small frame that is nothing; on a large one
it is the whole problem.

```python
with jb.optimize(rows=5_000_000):
    frame = (watched.hop(peer="~has_interact", rec="has_interact")
                    .filter(v.rec.is_in(wanted)))
```

Inside `optimize` a hop describes itself instead of taking place, the filters
written after it join the description, and the whole of it runs when something
reads the frame — in slices, each walked, filtered and reduced before the next
one starts. The answer is the same; the peak is a slice instead of the lot. On
MovieLens the query above is killed by the kernel eagerly and completes in
1.9 GB deferred.

**`rows` is a budget on what a step produces, and it is not an estimate.** The
graph keeps every node's degree, so the exact size of an expansion is
`degree[nodes].sum()` before a step is taken — 16 807 190 for the walk above.
The slices are cut where that running total crosses the budget, so a slice out
of a hub is shorter than one out of a leaf, and a walk that fits inside the
budget is not deferred at all.

The budget is applied again at **every** step, which is what makes the spelling
stop mattering: `.hop(a=..., b=...)` and `.hop(a=...).hop(b=...)` are the same
walk and now cost the same, because either way the second step is cut against
the middle it actually landed on.

| | | |
|---|---:|---:|
| one call, `rows=5M` | 97 s | 1.95 GB |
| two calls, `rows=5M` | 101 s | 2.42 GB |
| one call, `rows=1M` | 80 s | 3.13 GB |
| one call, `rows=20M` | — | killed |

Below a floor the budget stops buying anything, because what is left is the
answer itself: the surviving slices are accumulated rather than streamed. They
are concatenated without a rechunk, so the result exists once rather than twice,
but a query whose *answer* does not fit is still not helped. That, and choosing
the budget for you, is what a planner would add.

One more thing follows from working in slices: a predicate that aggregates
(`>= v.rec.score.mean()`) sees its slice rather than the whole result.

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

`spotify` is the third one, on the Million Playlist Dataset: name a few songs
and it answers with five more that belong beside them. There is no playlist node
for what you brought — what stands in for it is the crowd of real playlists that
already contain your songs.

```
-> ['Toxic -- Britney Spears', 'Bad Romance -- Lady Gaga']

0.86  ...Baby One More Time    Britney Spears    [447 playlists]
0.85  Poker Face               Lady Gaga         [530]
0.85  Womanizer                Britney Spears    [433]
0.84  Hollaback Girl           Gwen Stefani      [646]
```

Three signals, and the query says how much each counts: the **graph** finds the
candidates, the **count** is the evidence (damped — twice as many playlists is
not twice as good an answer), and the **model** is the taste. On its own the
factorization is a poor recommender on this data, and the use case says why; as
a re-ranker over songs the crowd already agrees on, it is what separates *the
same artists* from *the same decade*. 0.2–1 s a request on the 100 000-playlist
cut, and it needs `pip install -e '.[api]'` plus the dataset below.

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
(g.nodes(seed="movie").filter(v.seed.title == "Blade Runner")
   .hop(tag="has_tag").filter(v.tag.score >= 0.95)
   .hop(rec="~has_tag").filter(v.rec.score >= 0.95, v.rec != v.seed)
   .group_by(v.rec).agg(score=v.rec.score.sum(), through=v.tag.first())
   .top(4).labels("rec", "through"))
```

```
┌────────────────────┬──────────┬─────────────────┐
│ rec.label          ┆ score    ┆ through.label   │
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
node, the `group_by` that ranks by the whole pattern, the window in
`top(k, by=..., over=...)` that keeps the best k per user — or, sitting between
two hops, the k most promising partial walks, which is a beam search and was an
engine once. Those used to be a backtracking search, a Python loop over result
rows, a scope stack and a pluggable `Greedy`.

Files are read by column, not by line. A chunk of an edge file becomes three
parallel columns with one `replace` and one `split` — two C loops over the whole
chunk — and the ids come from `dict.fromkeys`, which deduplicates in C and in
first-seen order at once. What is left in Python runs once per *distinct node*
rather than once per edge: 2.78 M edges load in ~1.1 s.

### What a relation can say without being walked

```python
.filter(v.movie.has_genre.count() >= 2)     # its arity, as a fact about the graph
.filter(v.movie.directed_by.is_in(people))  # an edge to one of these exists
```

The count is a fact about the graph — the same number whatever the query asked,
which is what tells it apart from `group_by(...).agg(count)`. Neither expands
the frame: existence with a set walks the *given* side backwards and collects
what reaches it, so the cost is the degree of that set. Pass the smaller one.

```
jerboas/
  core.py         Strategy, and the Signal that aims one at columns
  graph.py        the data: integer ids, CSR adjacency, typed columns, nodes()/edges()
  frame.py        the query: a polars frame that knows its graph
  traverse.py     one hop, as a gather over CSR slices
  expr.py         v / col -- names, resolved by the frame that has the graph
  resolve.py      a hop that has not built its rows, and what resolves a name
  optimize.py     deferring a walk so its cost has a ceiling
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
