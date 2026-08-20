# Roadmap

Jerboas is a **retrieval layer for graphs where ranking is first-class**: the
scoring is part of the query rather than a stage after it, and the models that
do the scoring are trained through the same library that serves them.

That sentence is the filter for everything below. A feature earns a place here
if it makes retrieval and ranking one operation, or if it removes a reason
someone cannot use the library at all. Breadth for its own sake does not.

Ordered by what unblocks the most, not by what is most interesting to build.
Items 1, 2, 4 and most of 5 are **done** (see below); 3 is done as far as a
peephole goes and not as a planner.

---

## 1. A graph built from frames — done

`Graph.from_frames` takes a frame of edges (or a mapping from relation to
frame) and a mapping of attribute frames. A column names nodes as source keys or
as a `(type, column)` pair. What follows is the loader's, ids-are-positions
included.

**The problem.** `Graph(kg=..., edges=[...], attrs=[...])` takes file paths and
nothing else. Real data lives in Parquet, Postgres, an object store, another
graph database. Re-exporting it into `ml.has_interact` files is a tax paid
before anyone can evaluate whether the library is any good, and it makes jerboas
something you migrate *to* rather than something you add.

**The shape.**

```python
Graph.from_edges(frame, source="from", relation="rel", target="to", score="w")
     .with_attrs("movie", movie_frame)
```

Anything polars reads is then a graph: `pl.read_parquet`, `pl.read_database`,
an arrow table someone else produced. The file loader stays — it is fast and it
is the format the datasets ship in — and becomes one constructor among several
rather than the only one.

**What it costs.** The loader's invariant is that an id is a position
(`0..n-1` per type), and a frame from elsewhere will not honour it. `renumber=`
already exists for exactly this and does the sorting; the work is routing a
frame through the same `_finalize` path the file reader uses, and being honest
in the docs about which build produces stable ids and which does not.

**Why first.** Everything else on this list benefits people who are already
using it. This one decides who can start.

---

## 2. Vector columns, and a graded nearness condition — done

A list column in an attrs frame is a vector; `v.chunk.embedding.near(q, k=50)`
admits the k nearest and keeps the cosine as the column's confidence. Exact, by
matmul against a unit block. An index is the part still to come.

**The problem.** Attribute columns are int64, float64 or text
(`columns.build`). There is no vector, no index, no k-nearest anything. So the
first operation of every RAG pipeline — "the 50 chunks nearest this query
embedding" — cannot be expressed at all.

**The shape.** It is already designed, by analogy: nearness is graded
membership over a vector column exactly as `like` is over a text one.

```python
.filter(v.chunk.embedding.near(query, k=50))
.sort(v.chunk.embedding.score, descending=True)
```

Same three properties as `like`: it admits the k best rather than a region, the
measure that decided admission becomes the column's confidence, and the two
cannot disagree because they are one computation.

**How.** A `(n, d)` float32 array per type beside the columns; exact
brute-force scoring first, because a matmul against a few hundred thousand rows
is milliseconds and correctness is easier to argue about; an optional index
(hnswlib, faiss) behind the same expression later, where "approximate" becomes
visible as a lower confidence rather than as a different API.

**Why second.** It is the one missing primitive rather than a missing
convenience, and it lands on the existing design without bending it.

---

## 3. `jb.optimize`: laziness and a planner — the batching done, the planner not

**What exists.** `jb.optimize(rows=n)` defers a hop and the conditions about
where it lands, then runs the walk in slices. The budget is on what a step
*produces* and is exact rather than estimated -- the graph keeps every node's
degree, so an expansion's size is `degree[nodes].sum()` before a step is taken.
Slices are cut where that running total crosses the budget, so a slice out of a
hub is shorter than one out of a leaf, a walk that fits is not deferred at all,
and the budget is applied again at every step -- which is what makes
`.hop(a=..., b=...)` and `.hop(a=...).hop(b=...)` cost the same. A query the
kernel kills eagerly completes in 1.9 GB.

**What does not.**

*Nothing chooses the budget.* The default is a number, not a decision. A planner
would pick it from the memory it is allowed and the width of the rows.

*The answer is accumulated, not streamed.* The slices are concatenated without a
rechunk, so the result exists once rather than twice, but a walk whose *answer*
does not fit is still not helped -- and below a floor a smaller budget buys
nothing, because what is left is the answer.

*Nothing chooses the direction.* Which end of a pattern to expand from is a
choice with a large cost difference and nobody makes it, though the degrees that
would decide it are the same ones the slicing already reads.

*A predicate that aggregates sees its slice.* A planner would know which
predicates are row-local and refuse to defer the others; this one assumes.

## 4. Reducing several confidences to one — done

`.confidence("min" | "max" | "mean" | "product" | "sum" | callable)`.

**The open end.** Confidence deliberately does not compose along a walk:
`v.tag.score` is the step that revealed `tag`, `v.rec.score` the step after.
That was the right call — combining a string similarity, an edge weight and a
model's plausibility is a decision, not an operation. But a multi-hop pipeline
ends with several confidences and no canonical way to reduce them, and if the
library never decides, every caller invents their own.

**The shape.** A verb that makes the rule explicit rather than implicit:

```python
.confidence("min")                  # a row is as certain as its weakest step
.confidence(lambda *scores: ...)    # or say it yourself
```

with the same vocabulary `group_by(confidence=...)` already uses. Small, and
worth doing only once there is enough real usage to know which default is
honest.

---

## 5. Resolving a set of names by what connects them — done

`v.x.name.needle` says which of the things searched for a row is an answer to,
and `Frame.coherent(by=..., through=...)` keeps one row per group: the
combination of candidates that keep the most company with each other, two
candidates keeping company when something the graph knows holds both.

Measured by handing back the titles of real Spotify playlists and counting how
many resolve to the songs those playlists actually held:

| titles | most played | coherent |
|---:|---:|---:|
| 3 | 66.7% | **100.0%** |
| 16 | 81.2% | **98.4%** |

Two things that cost several wrong turns and are worth keeping written down.
**Whether** two candidates keep company beats **how often** (89.1% at sixteen),
which beats dividing that count by how far each reaches (77.5%): counting
favours the popular, dividing overshoots to the obscure, and the question being
asked is neither. And the connection is one hop to where they *meet* plus a
self-join, not a walk out and back -- the way back visits everything else the
meeting place holds and then throws all of it away.

What is left is cost: the pool for each name comes from `like`, which scans
every stored value, and that is most of a request. A name whose right answer is
not in the pool at all -- Oasis' `Wonderwall` is filed as `Wonderwall -
Remastered` and the exact-titled ones are eight covers -- is still out of reach,
which is what the separator below is for.

## 5b. A closeness that ranks how tightly a title contains

`like` scores any containment 1.0, so `Toxicity` ties with `Toxic` and
`Wonderwall - Remastered` ties with `Wonderwall`. The information exists --
`fuzzy.closest` takes the shortest containing match first -- and is thrown away
rather than becoming part of the score. Until it is, widening a pool to reach a
long title also fills it with near-misses.

## 6. Smaller things

| | |
|---|---|
| **A model has no checkpoint** | An embedding is fitted by a batch job and loaded from a file; a factorization is fitted at startup and refitted at every restart -- 23 s on the Spotify cut, and minutes on the whole of it. The arrays are the same shape as an embedding's, so `checkpoint.py` already knows how to store them. |
| **`like` scans every value** | Most of a request on Spotify: 680 000 titles gathered and measured for every search. Containment could be answered by an index over the words rather than by walking the column. |
| **`.to_pandas()` needs pyarrow** | An extra (`jerboas[pandas]`), which is fine, but the error when it is missing is polars' and mentions neither jerboas nor the extra. |
| **No 0.1 compatibility** | Deliberate, and the migration table in the README is the whole of it. If anyone is on 0.1, a `jerboas.legacy` shim is a week of work and probably not worth it. |
| **The whole Spotify graph is untried** | 70.8 M edges, and everything here was measured on the 100 000-playlist cut. |

Done since this list was written: `.pl` hides the shadow columns and `.raw` does
not; `unique` says whose confidence survives; `frame.py` gave up the resolver
and the pending hop to `resolve.py`; and `chunked(n)` is the interim the
benchmark was doing by hand.

---

## Not doing

**Model-zoo breadth.** Adding a translational model is a `tables` tuple and a
`plausibility()` — three or four lines. The value is that the extension point is
that small, not that there are forty of them. Benchmark libraries do breadth
better and it is not what this is for.

**A query language of its own.** The query is a dataframe so that what comes
out feeds the rest of an ML stack and what goes in composes with it. A DSL with
its own syntax would undo the reason the rewrite happened.

**Being generalist.** Recommendation and graph RAG are served today; ordinary
RAG becomes possible with item 2. Agent orchestration is somebody else's layer,
and jerboas should arrive there as an engine rather than as a framework.
