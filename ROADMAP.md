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

## 5. Resolving a set of names by what connects them

**The problem.** `like` resolves one name at a time, so a name with several
equally-close matches is decided by something outside the query -- popularity in
`usecase/spotify`, load order before that. `Wonderwall` is Oasis and also Ryan
Adams covering Oasis, and neither is the right answer in the abstract.

**The shape.** A *set* of names carries information one name does not: songs
somebody names together tend to sit in the same playlists, films together in the
same tastes. So keep every candidate rather than one, ask the graph for the
pairwise connection between candidates -- one two-hop query, the whole matrix at
once -- and choose one candidate per name to maximise the total. `k**n` by brute
force; a few passes of coordinate ascent in practice.

**Why it belongs here rather than in a use case.** Every service on this library
begins by turning names into nodes, and every one of them has this problem. And
it is the smallest real instance of what the library is for: a graph used for
*resolution*, not only for retrieval.

**The cheap escape, worth having anyway.** Let a caller pin a name with a
separator -- `"Wonderwall\tOasis"` -- which turns a guess into a constraint.
It composes with the above rather than replacing it.

## 6. Smaller things, each already known

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
