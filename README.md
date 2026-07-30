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

g = jb.Graph(kg="data/example/example.kg", ui="data/example/example.ui")

artist = Node("artist")
seeds = set(g.select(artist).where(Like(artist.id.is_in(["Golden"]))))

song, seed, path = Node("song"), Node(), Path()
g.select(song, Score()).where(
    path == [song, Edge(), seed],
    seed.is_in(seeds),
).rank(PageRank(to=seeds)).top(5)
```

```
Like("Golden") -> artist.Golden_Project, artist.Golden_Kids, artist.Golden_Collective

1.00  Broken_Dreams     0.94  Empty_Pulse       0.83  Lost_Shadows
0.95  Distant_Echo      0.93  Restless_Lights
```

That runs on a fresh clone: the example graph ships with the repo.

## Membership is graded

A crisp `Condition` answers *yes* or *no*. `Like` answers *how much*, in `[0, 1]`,
and that single idea is what the library is built around.

```python
Like(person.label.is_in(["Quentin Tarantino"]))   # fuzzy string match
Like(movie.year < 1990, t=0.8)                    # a threshold that leaks
Like(movie.year == 1994, width=5)                 # equality with a tolerance
```

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
    movie.type           # "movie"
    movie.id             # 123
    movie.label          # "Pulp Fiction"   -- resolved per type
    movie.attrs["year"]  # 1994             -- typed at load, not text
    str(movie)           # "movie.123"
```

`label` is virtual: it reads `title` on a movie and `name` on a person, so one
query serves every type without knowing any of their schemas.

```python
Like(Node(any_type).label.is_in(names))
```

A `Path` comes back as alternating `Key` and `Rel`, and a `Rel` knows which way
it was walked — which turns an explanation into a projection instead of a
string-parsing exercise:

```
Falling_Drift  --performed_by->  Golden_Project
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

## The one rule

> **Methods are only the pipeline verbs** — `select`, `where`, `rank`, `top`,
> `groupby`, `using` — and they live only on `Graph`/`Query`. They are the only
> things that touch the graph.
> **Everything you pass to a method is a passive object**: it describes *what*
> you want, never *how*.

| Family | Interface | Objects | Passed to |
|---|---|---|---|
| **Reference** | `Ref` / `Expr` | `Node`, `Attr`, `Edge`, `Path`, `Degree` | `select(...)` |
| **Condition** | `Condition` | `Compare`, `Like`, `In`, `Has`, `Match`, `And`/`Or`/`Not` | `where(...)` |
| **Strategy** | `Strategy` | `PageRank`, `Connectivity`, `MatrixFactorization`, `Embedding`, … | `rank(...)` |
| **Engine** | `Engine` | `Default`, `Greedy` | `using(...)` |

Adding a matcher is a new `Condition`. Adding a ranker is a new `Strategy`.
Neither edits the `Query`: every object describes itself to the compiler.

The predicate surface follows Polars, so most of it is already familiar:

```python
node.year >= 1990         node.name.is_in([...])      node.year.is_between(a, b)
node.title.contains("x")  node.rel.count() >= 2       a & b,  a | b,  ~a
```

Refs have identity semantics on purpose — two `Node("movie")` values are two
different pattern variables — so reuse the same object across `select` and
`where`. Get it wrong and the query says so rather than quietly returning
everything.

## Embeddings: train with torch, serve without it

Scoring a translational embedding is a projection and a norm, so a checkpoint
holds plain arrays and the ranking path needs numpy alone. **Only training needs
torch**, which means the machine answering queries never has to install it.

```python
from jerboas.models import TransD, train        # pip install jerboas[torch]

model = train(TransD(factors=64), g, epochs=15, device="mps")
model.save("kg.npz")
```

```python
from jerboas import Embedding                   # base install

g.select(rec, Score()).rank(Embedding.load("kg.npz", g, to=seeds)).top(10)
```

One strategy, many models: `Embedding` reads the model name out of the
checkpoint, so adding TransE or TransH never touches the ranking layer.

Two details that matter more than they look:

**Checkpoints rebind by name.** A node's integer id comes from load order, so a
checkpoint keyed by position would keep loading after you rebuild the graph and
silently score the wrong entities. Identity — type names, raw ids, relation
names — is stored beside the weights and resolved on load.

**Negative sampling is type-aware.** Corrupting a triple with a uniformly random
entity yields a type-wrong tail 99.8% of the time on MovieLens, so the model can
minimise its loss by learning to tell types apart instead of learning the
relation. Jerboas draws the corruption from the true endpoint's own type block,
so it has to learn something real to score it lower.

On MovieLens (15 369 nodes, 110 k edges) TransD at `factors=64` is 1.97 M
parameters, 7.9 MB, and trains in roughly a second per epoch on Apple MPS.

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

The MovieLens graph used by `example.py` and the benchmarks is **not** included:
GroupLens' usage licence states that "the user may not redistribute the data
without separate permission", and the IMDb-derived files are non-commercial-use
only. Fetch [ml-100k](https://grouplens.org/datasets/movielens/100k/) and the
[IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)
yourself, and lay them out as:

```
ml.kg          head <TAB> relation <TAB> tail      (movie.1  has_genre  genre.4)
ml.ui          user <TAB> item [<TAB> rating]
ml.<type>      a header row of column names, first column the id
```

Any graph in that shape works — nothing in the library is MovieLens-specific.

## How it is put together

Nodes are integers in contiguous per-type blocks, which is what makes the
universe an array rather than a dictionary: `keys_by_type` becomes a slice, an
embedding table is one `(n_nodes, factors)` matrix a strategy indexes directly,
and every constraint on a node — its type, identity, attributes, degrees —
compiles into a single boolean mask. The engines' per-candidate test is
`mask[node]`.

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
  checkpoint.py   the trained-embedding format (numpy only)
  strategies/     the ranking family
  models/         trainable models -- the only place torch is imported
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
