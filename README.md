# Jerboas

A non-deterministic-first, ORM-like graph query library.

```python
import jerboas as jb
from jerboas import Node, Edge, Path, Score, Like, PageRank

g = jb.Graph(kg="data/example/example.kg", ui="data/example/example.ui")

song, seed, path = Node("song"), Node(), Path()
seeds = set(g.select(Node("genre")).where(Node("genre").id == "reggae"))

g.select(song, Score()).where(
    path == [song, Edge(), seed],
    seed.is_in(seeds),
).rank(PageRank(to=seeds)).top(10)
```

## The one rule

> **Methods are only the pipeline verbs** — `select`, `where`, `rank`, `top`,
> `groupby`, `using` — and they live only on `Graph`/`Query`. They are the only
> things that touch the graph.
> **Everything you pass to a method is a passive object**: it describes *what*
> you want, never *how*.

Adding a matcher is a new `Condition`; adding a ranker is a new `Strategy`.
Neither edits the `Query`.

## The four object families

| Family | Interface | Objects | Passed to |
|---|---|---|---|
| **Reference** | `Ref` / `Expr` | `Node`, `Attr`, `Edge`, `Path`, `Degree` | `select(...)` |
| **Condition** | `Condition` | `Compare`, `Like`, `In`, `Has`, `Match`, `And`/`Or`/`Not` | `where(...)` |
| **Strategy** | `Strategy` | `PageRank`, `Connectivity`, `MatrixFactorization`, `Embedding`, … | `rank(...)` |
| **Engine** | `Engine` | `Default`, `Greedy` | `using(...)` |

Graded membership is the primitive. A crisp `Condition` returns membership in
`{0, 1}`; `Like` returns it in `[0, 1]`. A `Like` in `where(...)` does both — it
widens admission *and* contributes its membership as a ranking weight.

## What a query returns

`Key` values, not strings: `.type`, `.id`, `.label`, `.attrs`, printing as
`movie.123`. A `Path` yields alternating `Key` and `Rel`, where a `Rel` knows the
direction it was walked.

`label` is virtual and resolves per type — `title` for a movie, `name` for a
person — so one query serves every type:

```python
Like(Node(type_).label.is_in(names))
```

## Relations have one name and two directions

There is no `directed_by_r`. The graph stores each edge once, as an
edge-labeled CSR plus its transpose:

```python
person.directed_by.inverse        # the movies a person directed
Edge("has_genre").inverse         # in a path pattern
Edge()                            # wildcard: any relation, either direction
```

## Embeddings

Fitting a knowledge-graph embedding needs torch; **serving one does not**.
Training writes a checkpoint of plain arrays that `Embedding` reads with numpy
alone, so a deployment installs the base package.

```python
from jerboas.models import TransD, train           # pip install jerboas[torch]

model = train(TransD(factors=64), g, epochs=15, device="mps")
model.save("ml.transd.npz")
```

```python
from jerboas import Embedding                      # no torch needed

g.select(rec, Score()).rank(Embedding.load("ml.transd.npz", g, to=seeds)).top(10)
```

Checkpoints record the identity of every row — type names, raw ids, relation
names — and rebind by name. A checkpoint keyed by position would keep loading
after the graph was rebuilt in a different order and silently score the wrong
entities.

On MovieLens (15369 nodes, 110k edges) TransD at `factors=64` is 1.97M
parameters, 7.9 MB, and trains in about a second per epoch on Apple MPS.

## Data

`data/example/` is a small synthetic graph (songs, artists, genres) and ships
with the repo, so everything above runs on a fresh clone.

The MovieLens graph the benchmarks and `example.py` use is **not** included:
GroupLens' usage licence states that "the user may not redistribute the data
without separate permission", and the IMDb-derived files are non-commercial-use
only. Obtain [ml-100k](https://grouplens.org/datasets/movielens/100k/) and the
[IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)
yourself, then build `data/movielens/` with the loader's expected layout:

```
ml.kg          head <TAB> relation <TAB> tail        (`movie.1  has_genre  genre.4`)
ml.ui          user <TAB> item [<TAB> rating]
ml.<type>      a header row of column names, first column the id
```

Any graph in that layout works -- nothing in the library is MovieLens-specific.

## Install

```bash
pip install -e .              # numpy + scipy
pip install -e '.[torch]'     # + training
pip install -e '.[api]'       # + the FastAPI example
pip install -e '.[dev]'       # + pytest
```

## Layout

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

Apache-2.0. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for the third-party
attributions — the TransD and TransE formulations were adapted from
[hopwise](https://github.com/tail-unica/hopwise) (MIT).
