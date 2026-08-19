# Roadmap

Jerboas is a **retrieval layer for graphs where ranking is first-class**: the
scoring is part of the query rather than a stage after it, and the models that
do the scoring are trained through the same library that serves them.

That sentence is the filter for everything below. A feature earns a place here
if it makes retrieval and ranking one operation, or if it removes a reason
someone cannot use the library at all. Breadth for its own sake does not.

Ordered by what unblocks the most, not by what is most interesting to build.

---

## 1. A graph built from frames

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

## 2. Vector columns, and a graded nearness condition

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

## 3. `jb.optimize`: laziness and a planner

**The problem.** Every verb materialises. The one exception is the peephole
that lets a filter written after a hop apply to the arrays the hop produced —
useful, measured at −24% on a wide frame, and not a planner. A two-hop wildcard
over a large graph still exhausts memory unless the caller writes `unique()`
between the steps and batches the input by hand, as `benchmark/run.py` does.

**The shape.** Opt-in, so the eager path stays the one that is easy to reason
about:

```python
from jerboas import optimize

with optimize():
    ...                      # verbs build a plan; it runs at the first read
```

**What a planner would have that others do not.** The graph knows its degrees,
per relation and per direction, for free (`graph.degree` is memoized). That is a
cardinality estimate no join optimizer over anonymous tables can get, and it is
what would let the planner choose which end of a pattern to expand from, when to
fold, and when to batch.

**Why third.** It unlocks scale, and scale matters once 1 and 2 have brought
graphs that are not MovieLens. It is also the largest piece here by some
distance.

---

## 4. Reducing several confidences to one

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

## 5. Smaller things, each already known

| | |
|---|---|
| **`.pl` exposes shadows** | Dropping to polars shows `__jb_score__rec`. Documented, but a convention that leaks. A `.pl` that hides them by default, with `.raw` for the whole thing, would cost nothing. |
| **`unique()` and confidence** | Deduplicating keeps an arbitrary row's confidence. `group_by` has a rule now; `unique` should say what its rule is. |
| **`frame.py` is 1 071 lines** | The largest file after the loader. The resolver and the pending hop are each a coherent piece and could be their own module. |
| **Batching is manual** | `BLOCK = 128` in the benchmark is the caller doing what a planner would. A `chunked(n)` helper is the honest interim step. |
| **`.to_pandas()` needs pyarrow** | An extra (`jerboas[pandas]`), which is fine, but the error when it is missing is polars' and mentions neither jerboas nor the extra. |
| **No 0.1 compatibility** | Deliberate, and the migration table in the README is the whole of it. If anyone is on 0.1, a `jerboas.legacy` shim is a week of work and probably not worth it. |

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
