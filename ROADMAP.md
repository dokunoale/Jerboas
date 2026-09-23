# Roadmap

Jerboas is a **retrieval layer for graphs where ranking is first-class**: the
scoring is part of the query rather than a stage after it, and the models that
do the scoring are trained through the same library that serves them.

That sentence is the filter for everything below. A feature earns a place here
if it makes retrieval and ranking one operation, or if it removes a reason
someone cannot use the library at all. Breadth for its own sake does not.

Ordered by what unblocks the most, not by what is most interesting to build.
Everything except item 3 is **done** (see below); 3 is done as far as a peephole
goes and not as a planner.

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

## 3. `jb.optimize`: laziness and a planner — done

Inside `optimize` every hop is planned (plan/). A plan is a list of stages.
Each row-local condition joins the earliest stage that can decide it, and a
condition that reads other rows waits for the whole answer, together with the
conditions written beside it. A stage that must land in a set cheaper than the
frame's expansion is walked from the set (`traverse.Reach`). The budget is
rows, or bytes priced at the frame's row width, and by default half the free
memory. The slices are a generator: `Frame.batches()` streams them, and `top`,
`unique` and decomposable `group_by` aggregates fold them without building the
whole. On the README's MovieLens walk that is 89 s -> 6.6 s.

**What is left.**

*Direction is chosen per stage, from its own landing set only.* A condition two
stages later that narrows the end does not reach back to choose where the walk
starts. Meeting in the middle across stages would need the stages' sets
propagated backwards.

*`agg` over raw polars expressions is not folded.* `pl.len()` in an `agg` is
taken whole. `group_by(...).len()` and `v.x.count()` are folded.

*The row-local test for a raw polars expression reads its printed form.* An
allowlist would be safer, but there is no way to walk a polars expression tree
from Python.

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

Cost and reach were both what was left, and both are handled below: the pool now
comes from an index rather than a scan, and a name whose right answer is not in
the pool at all -- Oasis' `Wonderwall` is filed as `Wonderwall - Remastered`,
and the exact-titled ones are eight covers -- is reachable by writing the
performer after a tab, which narrows what is searched rather than filtering what
matched.

## 5b. What a search measures with is a rule -- done

`like` used to be difflib, and the use case that wanted whole words had to work
around it. It is now `.filter(v.x.name.like(needles, rule=...))`, where a `Rule`
(rules.py) answers three things -- which rows, how close, to which needle -- and
the query does not change when the measure does: `Fuzzy` (characters), `Words`
(an inverted index over whole words), `Semantic` (cosine). `near` is the same
verb exclusively: the k closest that are not the thing itself, which every rule
here reads as a perfect score.

That settles the closeness this section was asking for, though not the way it
proposed. `Toxicity` is one word and holds none of `Toxic`, so it scores 0
rather than tying; `Wonderwall - Remastered` holds all of `Wonderwall` and does
tie with the bare title, which is right -- a remaster is that song, and which of
them is meant is what `coherent` and the `\t` separator decide.

The index is where the cost went: a Spotify request is **0.17 s against 1.2 s**,
and the evaluation above **1 s a query against 14 s**. The rule reads posting
lists, so what it costs is the rows holding any of the words rather than the
rows that exist, and it falls back to `Fuzzy` where no index applies rather than
answering worse without saying so.

A rule uses whatever it must -- an index, a matmul, difflib -- but selects with
the library's own vocabulary, so a rule cannot be written that the library could
not have expressed.

## 5c. Reading a playlist as one thing or several -- done

Two additions the recommender wanted, both of them general.

`Concentration(space, relation=)` asks whether what a node touches points one
way or every way: the resultant length of its neighbours' vectors, in `[0, 1]`.
It tells a song in five hundred playlists about the same thing from a song in
five hundred playlists about anything -- film scores and classical around 0.49,
trap around 0.98 -- which a degree cannot, and which dividing by the degree gets
wrong in the other direction. Two sparse products over the whole graph,
memoized.

`top(..., temperature=)` makes the ranking a sample rather than a maximum, by
Gumbel's trick, measured in the scores' own spread so it means the same thing
whatever the column holds.

On top of them the service takes a `concentration`: at 0 the playlist is one
field and the answers come from wherever in it they score best, at 1 every song
is its own and the answers cover all of them, in between k-means over the latent
space with `top(over="part")` answering each. Asked
`Smells Like Teen Spirit / Come As You Are / Lithium / Poker Face / Bad Romance
/ Toxic`, 0.0 answers all grunge, 0.5 grunge plus Britney, 1.0 both halves.

## 6. Smaller things

| | |
|---|---|
| **A model has no checkpoint** | An embedding is fitted by a batch job and loaded from a file; a factorization is fitted at startup and refitted at every restart -- 23 s on the Spotify cut, and minutes on the whole of it. The arrays are the same shape as an embedding's, so `checkpoint.py` already knows how to store them. |
| **`.to_pandas()` needs pyarrow** | An extra (`jerboas[pandas]`), which is fine, but the error when it is missing is polars' and mentions neither jerboas nor the extra. |
| **No 0.1 compatibility** | Deliberate, and the migration table in the README is the whole of it. If anyone is on 0.1, a `jerboas.legacy` shim is a week of work and probably not worth it. |
| **The first request on the whole Spotify graph is slow** | Tried: with `cache=` the graph maps back in 0.4 s (140 s to build the first time), warm requests take 2-5 s, peak RSS 2.7 GB on an 8 GB machine. The first request takes 33 s because it builds the word index, the relation bounds and `Concentration`'s products; a service could warm them at startup. The factorization fit (250 s) is the item above. |

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
