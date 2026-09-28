# Roadmap

Jerboas is **the retrieval layer in front of a model, built to trade exactness
for bounded latency where the model makes exactness worth little.**

A system that answers with a model usually has two halves: a store that finds
the candidates and a model that ranks them. When the store is exact and the
query is heavy -- a multi-hop walk over a large graph -- the store is the
bottleneck, and it is paying for a precision the next stage does not have: the
model is approximate, often stochastic, and scores whatever it is handed. So
the store may give up some determinism in exchange for speed, **provided the
exchange is measured, controlled, and reproducible.**

That sentence is the filter for everything below. A feature earns a place if it
makes a walk cheaper at a measured cost in answer quality, or if it makes that
cost visible. Breadth for its own sake does not.

The dataframe stays what it is: a graph *is* a frame of every walk, a query is a
view of it, and everything that is not a traversal is polars (see the README).
That part works and is not where the effort goes next.

---

## The bet: a hop is a decoding step

Generating text with a language model is a loop: at each step a distribution
over the next token, a stack of *logits processors* that reshapes it (penalties,
masks, temperature), and a *selector* that keeps a few (greedy, top-k, top-p,
sampling, beam). A walk over a graph has the same shape:

| decoding | walking |
|---|---|
| logits over the vocabulary | the edges leaving the frontier |
| logits processors | maps stacked over those edges: the edge's weight, a node's specificity, PageRank, a factorization's taste |
| a mask to `-inf` | a `filter` |
| greedy, top-k, top-p, sampling, beam | which edges the walk actually follows |
| beam width | the budget per step |

Read this way, determinism is a setting of the selector rather than a property
of the engine. No budget is today's exact walk. A budget of n per row keeps the
frame at `frontier × n` rather than `frontier × degree`, which is where the time
goes. The maps say where to spend it.

## Where it stands: the budgeted step

```python
.hop(peer=step("~has_interact").top(20))                           # the 20 heaviest
.hop(peer=step("~has_interact").sample(20, seed=0))                # 20 drawn by weight
.hop(peer=step("~has_interact").top_p(0.9))                        # the nucleus
.hop(playlist=step("~contains").top(100, by=1 / v.playlist.contains.count().log1p()))
.hop(rec=step("contains").top(500, by=cheap).top(50, by=v.rec.x * v.user.y))   # a cascade
.hop(playlist=step("~contains").sample(100, by=1, seed=0).probability(by=1))
```

Every argument of `hop` is a `Step` (query/step.py): relations, and selectors
that chain. A selector's `by` is a map over the step's arrival. When it reads
nothing else it is a fact about the graph: computed once over every edge of the
relations and **fused into the store** (`traverse.Fused`), with each node's
edges in the order the map puts them, so the n best are the first n of its
slice, and a running sum, so a draw or a nucleus is a binary search. A budgeted
step then costs what it keeps, whatever the degree. When the map also reads the
frame the walk leaves from (whose taste, which seed) it is evaluated on each
row's candidates, as arrays, before any row is built. A draw is a
counter-based hash of (seed, node, draw), so it is the same however the walk is
sliced: for one seed it is a walk over one sparsified graph. `probability()`
makes what a step measured the transition probability of a random walk,
renormalized over what was kept.

**Measured.** MovieLens-100k: the three-hop co-watching walk of
`benchmark/run.py`, with a matrix factorization re-ranking what it reaches
(`benchmark/approx.py`):

| peers per film | walk s | NDCG@10 | exact top 10 kept |
|---|---:|---:|---:|
| all (exact) | 3.40 | 0.0538 | 100% |
| 20 heaviest | 1.54 | 0.0538 | 97.8% |
| 5 heaviest | 0.64 | 0.0540 | 83.6% |
| 5 most specific | 0.33 | 0.0544 | 75.7% |
| 5 heaviest, cut after the whole hop | 1.15 | 0.0540 | 83.6% |

MovieLens is small, and its NDCG hardly moves whatever the walk does: the
factorization ranks the popular films first, and every variant reaches them.
It is the sanity check, not the evidence.

Spotify MPD, the whole graph (4.3 M nodes, 70.8 M edges, an 8 GB laptop, peak
2.4 GB): playlist continuation from 5 songs of a real playlist, with the rest
held out and the playlist itself left out of the crowd. 1000 playlists, warm and
interleaved (`benchmark/playlists.py`). The *vote* is the map:

- `count`: one per playlist;
- `balanced`: each seed's playlists share one vote, i.e. the first step's
  `probability()`;
- `walk`: the two-step random walk's probability, i.e. `probability()` on both
  steps.

The *budget* is the selector. `kept` and the paired difference are against the
exact walk under the same vote. The interval is a 95% bootstrap, p a Wilcoxon
signed-rank test.

| vote | budget | p50 s | p90 s | p99 s | kept | hit@10 | vs exact | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| count | exact | 0.253 | 1.761 | 6.985 | 100% | 23.8% [22.3, 25.3] | | |
| count | sample 100 playlists | 0.062 | 0.101 | 0.299 | 64.5% | 28.2% | +4.5 [+3.7, +5.2] | 2e-29 |
| count | 100 most specific | 0.056 | 0.095 | 0.203 | 44.5% | 30.2% | +6.4 [+5.4, +7.6] | 1e-28 |
| count | sample 100, then 50 songs | 0.048 | 0.067 | 0.119 | 56.8% | 28.1% | +4.4 [+3.5, +5.3] | 3e-23 |
| balanced | exact | 0.255 | 1.599 | 7.710 | 100% | 29.0% [27.4, 30.6] | | |
| balanced | sample 100 playlists | 0.062 | 0.104 | 0.258 | 85.0% | 28.6% | -0.4 [-0.8, -0.0] | 0.12 |
| balanced | 100 most specific | 0.056 | 0.102 | 0.236 | 64.0% | 29.9% | +0.9 [+0.1, +1.7] | 0.03 |
| balanced | sample 100, then 50 songs | 0.048 | 0.064 | 0.127 | 71.1% | 28.4% | -0.6 [-1.1, -0.0] | 0.14 |
| walk | exact | 0.284 | 1.648 | 7.471 | 100% | **32.4%** [30.8, 34.1] | | |
| walk | sample 100 playlists | 0.062 | 0.102 | 0.298 | 79.7% | 31.3% | -1.1 [-1.6, -0.7] | 1e-05 |
| walk | 100 most specific | 0.056 | 0.096 | 0.232 | 64.7% | 30.3% | -2.1 [-2.8, -1.3] | 1e-06 |
| walk | sample 100, then 50 songs | 0.048 | 0.067 | 0.132 | 71.4% | 30.7% | -1.7 [-2.2, -1.2] | 1e-07 |

What is established:

1. **The budget bounds the tail.** The exact walk's p99 is 7-7.7 s and its p90
   1.6-1.8 s, because a popular seed reaches tens of thousands of playlists.
   Budgeted, p99 is 0.12-0.30 s and p90 under 0.11 s: 25-60x at p99 and about
   4-5x at the median. Latency stops depending on the degree of what was
   asked.
2. **Under the same map the cost is small, and now measured.** Under
   `balanced` no budget costs a significant point, and the specific one gains
   0.9. Under `walk`, the best map, budgets cost 1.1-2.1 points, and that cost
   is significant. `walk` with 100 sampled playlists (31.3%, p99 0.30 s) still
   beats every exact walk under a lesser map.
3. **The map matters more than the budget.** From `count` to `walk` is +8.6
   points at the same cost. Under `count`, budgets "gain" 4-6 points, but that
   is the map in disguise: a budget per seed caps how much one seed votes,
   which is what `balanced` says outright. This was the first reading of the
   prototype, and the control is what corrected it.
4. **Overlap with the exact answer does not predict quality.** Under
   `balanced`, 64% of the exact top 10 kept is +0.9 points of hit. A budget
   judged by its recall against the exact walk would be rejected for being
   different, not for being worse.
5. **A cold store lies about speed.** The prototype's first run put the exact
   walk at 2.9 s mean, because it ran first and paid for the pages of a graph
   mapped from an external disk. The benchmark now reads the cache before it
   times anything.

---

## I. Non-determinism, stated precisely

### 1. Approximate and stochastic are two knobs

*Approximate* is a budget: fewer edges, a subset of the exact candidates.
*Stochastic* is a draw: a different subset each time. Speed comes from the
first, diversity from the second, and a service that wants speed usually wants
the same answer twice. So they stay separate: `top(n)` is approximate and
deterministic, `sample(n, seed=s)` is approximate and reproducible, and only
`sample(n)` without a seed varies. The draws are counter-based (the idea of
Salmon et al., 2011), which is what makes a seed mean the same thing across
slices, processes and machines.

`Frame.top(temperature=)` stays as the selector of the *final* answer, by
Gumbel-top-k (Kool et al., 2019). Within a walk, `top_p(p)` is the nucleus
(Holtzman et al., 2020): the fewest best edges holding p of the row's mass. On
a heavy-tailed degree distribution it is the budget that adapts, a few edges
where one dominates and many where none does. It is implemented, fused and
dynamic, and not yet measured: it needs a map that is not flat to mean
anything, and on Spotify's playlist edges the only maps tried so far are.

### 2. Judged downstream, with an error bar

"The model is approximate anyway" holds only if the retrieval's error is small
next to the model's and is measured where it lands. Point 4 above is why recall
against the exact walk is the wrong yardstick. The yardsticks are:

- **the downstream metric** (hit, NDCG) at a latency, as a Pareto frontier, not
  one point;
- **the latency distribution** (median, p90, p99), because a budget's value is
  mostly in the tail;
- **an error bar.** 60 queries separated 20% from 29% but not 27% from 29%. At
  1000 queries, paired, a one-point difference is resolved (the table above),
  since every variant answers the same queries.

A step further is to make the error part of the answer. A walk that samples
knows each row's inclusion probability, so an aggregate over it can be
*estimated* rather than truncated: the Horvitz-Thompson estimator (1952) makes
`group_by(...).len()` over a sampled walk an unbiased estimate of the exact
count, with a variance. This is online aggregation (Hellerstein et al., 1997)
and ripple joins (Haas & Hellerstein, 1999) carried to joins, and Wander Join
(Li et al., 2016) is exactly that over random walks through a join graph. A hop
*is* a join (README), so it applies as it stands. In Jerboas the probability
would be a shadow column beside the confidence, and the estimate a confidence
on the aggregated column.

### 3. What a budget cannot see, and the techniques to try

A budget decides at step one on what step one knows. A playlist that looks
unremarkable may hold exactly the songs the model would rank first, and a
greedy cut never finds out. This is where most of the work is. Each technique
below is a candidate selector, measured on the same grid:

| technique | idea | literature | in Jerboas |
|---|---|---|---|
| **sample, don't prune** | a draw in proportion to the map reaches the tail sometimes, and is unbiased with the right weights | Horvitz & Thompson 1952; Wander Join (Li et al. 2016); neighbour sampling in GNNs: GraphSAGE (Hamilton et al. 2017), FastGCN (Chen et al. 2018), LADIES (Zou et al. 2019) | `sample(n, by=)`: done. The inclusion probability as a column: to do |
| **lookahead** | rank an edge by what it can still lead to, not only by itself: an upper bound on the next step's best | A* (Hart et al. 1968); beam search | a map that reads a node's degree *into a landing set* (`known`), which the planner already computes (`Reach`) |
| **exact top-k, early stop** | when the score is a monotone sum of per-node maps, stop once no unseen candidate can enter the top k. Exact and faster | Threshold Algorithm (Fagin et al. 2003); WAND (Broder et al. 2003); Block-Max WAND (Ding & Suel 2011) | the fused order *is* TA's sorted list. The deterministic end of the spectrum, and the baseline an approximation has to beat |
| **random-walk estimators** | estimate a personalized PageRank or a visit count by short walks with a stopping rule, not a matrix | Monte Carlo PPR (Fogaras et al. 2005; Avrachenkov et al. 2007); FAST-PPR (Lofgren et al. 2014); bidirectional PPR (Lofgren et al. 2016); Pixie (Eksombatchai et al. 2018) | `PageRank` computes the whole vector today. A walk-based `PageRank` with a budget of steps is the same Strategy at a fraction of the cost |
| **per-seed aggregation** | combine per-seed evidence so an item reached from several seeds beats one reached many times from one | Pixie's multi-pin boosting (Eksombatchai et al. 2018); item-based top-N normalization (Deshpande & Karypis 2004) | done: `probability()` on the first step is the `balanced` vote, on both steps the two-step walk's probability. Pixie's boosting (summing square roots of per-seed visits) is still to try |
| **diverse selection** | keep a budget of k that covers the frontier rather than k near-duplicates | MMR (Carbonell & Goldstein 1998); DPPs (Kulesza & Taskar 2012), fast greedy MAP (Chen et al. 2018); facility location, whose greedy is within 1-1/e (Nemhauser et al. 1978); k-medoids | a selector `diverse(k)` with distance `1 - p` between edges. k-means does not apply, since `1 - p` is not a metric and a graph has no centroid, but k-medoids and facility location do. The Spotify service's k-means over the latent space is the same idea done by hand |

## II. Optimizing the stack of maps

A map is a weight per edge. Several stacked are the distribution a step selects
from. How they are stored, ordered and combined is the second half of the work.

1. **Static maps fused into the store (done).** A map that reads only the step
   is computed once over the relations a step names, one direction, and
   stored as an order (int32 per edge), the map itself (float32) and, when
   drawn from or shared out, a running sum (float64). On Spotify's 66 M
   playlist edges that is roughly 265 + 265 + 530 MB. One relation needs no
   position array, being one contiguous run per node, and a constant map fuses
   into nothing. A stack of static maps is one expression, `by=a * b`, and so
   one order. To do: draws in O(1) instead of O(log degree) with alias tables
   (Walker 1977; Vose 1991).
2. **Dynamic maps on the candidates (done).** A map that reads the frame the
   walk leaves from is evaluated on each row's candidates as arrays, handed
   only the columns it names, before any row of the frame is built. It costs
   the degree, as the exact walk does, but not the rows.
3. **Cheap before expensive (done as a chain; early stopping to do).**
   Selectors chain, `top(500, by=cheap).top(50, by=dear)`, so the static map
   chooses in O(width) what the dynamic one is paid for: cascade ranking (Wang,
   Lin & Metzler 2011), the same idea as Viola & Jones's cascade (2001). What
   is missing is stopping on bounds: II.1's order gives each segment's maximum
   for free, so a cascade could stop as TA does.
4. **The space maps stack in (done for walks).** For a walk the natural space
   is probabilities: a `probability()` step multiplies its transition by the
   confidence of the node it leaves from, so the last step's confidence is the
   probability of the walk, and `group_by(confidence="sum")` is the chance of
   ending on each node. Everywhere else confidence still does not compose, as
   the README says. The two readings coexist, one per step. What is not done
   is the sampled walk's inclusion probability, which is what the estimator in
   I.2 needs.
5. **Embedding maps behind an index.** A taste that is a dot product is a
   maximum-inner-product search: HNSW (Malkov & Yashunin 2018), FAISS (Johnson
   et al. 2019), asymmetric LSH for MIPS (Shrivastava & Li 2014). `near` already
   has the shape, and an approximate index shows up as a lower confidence, not
   as another API. Integrated, not rewritten.
6. **Which maps.** The prototype re-found old ones: inverse user frequency
   (Breese et al. 1998) as `specific`, per-seed normalization as `balanced`,
   and the random walk's own probability as `walk`, which is the best measured
   (+8.6 points over counting).
   Which maps are worth fusing is an empirical question per dataset, and the
   benchmark grid is how it gets asked.

## III. What changes in the library

Disruptive changes only when they remove code, speed something up, and there is
no cleaner design.

- **One step type (done).** Every argument of `hop` becomes a `Step` at the
  door, and a `Step` answers every question the planner and the walk ask of it
  (relations, width, degree, target type, reaches, walk). Eight helpers left
  `frame.py`, which lost 80 lines, and budgets over several relations came out
  of the same change.
- **The beam idiom went (done).** `.hop(mid=()).top(5, over="seed")` is a
  budget written as a cut after a full expansion. The "cut" rows of
  `benchmark/approx.py` measure it slower and no better than `step(...).top`,
  and the README and `Frame.top` now point to the step.
- **`balanced` as a vote (done)**, as `probability()`. It is not a special
  vote: it is what a random walk measures.
- **Selection in turns (done).** `top(k, spread="part")` takes k across
  groups in turns, the partition form of diverse selection (I.3). It replaced
  a quota, a top-up and two dedupes in the Spotify service.
- **One table of confidence rules (done).** `FOLD` was defined twice,
  identically, in `frame.py` and `resolve.py`. It is one table now, with
  `"sum"` added for walks' mass.
- **A provenance of nothing is refused (done).** `v.nobody.score` answered 1.0
  for a column the frame does not have, which is the worst way to be wrong.
  Found by the step's map, fixed in the resolver.
- **`learn/` produces maps.** Training is what fills a map, and a checkpoint is
  a map ready to fuse or to evaluate on a frontier. It is not a pillar of its
  own.

## IV. Protocol

The library first. A publication only if the frontier holds up, and then this is
its method:

- **Datasets.** MovieLens-100k (small, dense, the sanity check), Spotify MPD on
  the whole graph (large, heavy-tailed: where budgets matter), and one
  multi-hop question-answering set over a knowledge graph, such as MetaQA
  (Zhang et al. 2018), so the claim is not recommendation-only.
- **Baselines.** The exact walk. A Pixie-style random walk with early stopping.
  GNN-style fixed fan-outs. Retrieval by ANN alone, without the graph.
- **Measures.** Latency median, p90 and p99, peak memory, the downstream metric
  with a confidence interval, and the Pareto frontier over budgets.
- **Hygiene.** Warm stores, variants interleaved per query, seeds recorded,
  ≥ 1000 queries, paired tests. Point 5 of the prototype is why.
  `benchmark/playlists.py` does all of it: the cache read once before
  anything is timed, a rotating order per query, bootstrap intervals and a
  Wilcoxon signed-rank test against the exact walk under the same vote.

## Not doing

**Model-zoo breadth.** A model is a map. The value is how cheaply a map is
applied, not how many there are.

**A query language of its own.** The query is a dataframe so that what comes out
feeds an ML stack and what goes in composes with it.

**Unmeasured approximation.** No budget is a default, and none ships without a
row in the benchmark grid saying what it costs.

**Being an index.** ANN search, full-text search and storage are other
libraries' work, integrated behind a map or a rule. What Jerboas owns is the
walk between them.

---

## References

- Avrachenkov, Litvak, Nemirovsky, Osipova. *Monte Carlo methods in PageRank computation: when one iteration is sufficient.* SIAM J. Numer. Anal., 2007.
- Breese, Heckerman, Kadie. *Empirical analysis of predictive algorithms for collaborative filtering.* UAI, 1998.
- Broder, Carmel, Herscovici, Soffer, Zien. *Efficient query evaluation using a two-level retrieval process* (WAND). CIKM, 2003.
- Carbonell, Goldstein. *The use of MMR, diversity-based reranking for reordering documents and producing summaries.* SIGIR, 1998.
- Chen, Ma, Xiao. *FastGCN: fast learning with graph convolutional networks via importance sampling.* ICLR, 2018.
- Chen, Zhang, Zhou. *Fast greedy MAP inference for determinantal point process to improve recommendation diversity.* NeurIPS, 2018.
- Deshpande, Karypis. *Item-based top-N recommendation algorithms.* ACM TOIS, 2004.
- Ding, Suel. *Faster top-k document retrieval using block-max indexes.* SIGIR, 2011.
- Eksombatchai et al. *Pixie: a system for recommending 3+ billion items to 200+ million users in real-time.* WWW, 2018.
- Fagin, Lotem, Naor. *Optimal aggregation algorithms for middleware.* JCSS, 2003.
- Fogaras, Rácz, Csalogány, Sarlós. *Towards scaling fully personalized PageRank.* Internet Mathematics, 2005.
- Haas, Hellerstein. *Ripple joins for online aggregation.* SIGMOD, 1999.
- Hamilton, Ying, Leskovec. *Inductive representation learning on large graphs* (GraphSAGE). NeurIPS, 2017.
- Hart, Nilsson, Raphael. *A formal basis for the heuristic determination of minimum cost paths.* IEEE TSSC, 1968.
- Hellerstein, Haas, Wang. *Online aggregation.* SIGMOD, 1997.
- Holtzman, Buys, Du, Forbes, Choi. *The curious case of neural text degeneration.* ICLR, 2020.
- Horvitz, Thompson. *A generalization of sampling without replacement from a finite universe.* JASA, 1952.
- Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs.* IEEE Trans. Big Data, 2019.
- Kool, van Hoof, Welling. *Stochastic beams and where to find them: the Gumbel-top-k trick.* ICML, 2019.
- Kulesza, Taskar. *Determinantal point processes for machine learning.* Foundations and Trends in ML, 2012.
- Li, Wu, Yi, Zhao. *Wander join: online aggregation via random walks.* SIGMOD, 2016.
- Lofgren, Banerjee, Goel, Seshadhri. *FAST-PPR: scaling personalized PageRank estimation for large graphs.* KDD, 2014.
- Lofgren, Banerjee, Goel. *Personalized PageRank estimation and search: a bidirectional approach.* WSDM, 2016.
- Malkov, Yashunin. *Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs.* IEEE TPAMI, 2018.
- Nemhauser, Wolsey, Fisher. *An analysis of approximations for maximizing submodular set functions.* Math. Programming, 1978.
- Salmon, Moraes, Dror, Shaw. *Parallel random numbers: as easy as 1, 2, 3.* SC, 2011.
- Shrivastava, Li. *Asymmetric LSH (ALSH) for sublinear time maximum inner product search.* NeurIPS, 2014.
- Viola, Jones. *Rapid object detection using a boosted cascade of simple features.* CVPR, 2001.
- Vose. *A linear algorithm for generating random numbers with a given distribution.* IEEE TSE, 1991.
- Walker. *An efficient method for generating random variables with general distributions.* ACM TOMS, 1977.
- Wang, Lin, Metzler. *A cascade ranking model for efficient ranked retrieval.* SIGIR, 2011.
- Zhang, Dai, Kozareva, Smola, Song. *Variational reasoning for question answering with knowledge graph* (MetaQA). AAAI, 2018.
- Zou, Hu, Wang, Jiang, Sun, Gu. *Layer-dependent importance sampling for training deep and large graph convolutional networks* (LADIES). NeurIPS, 2019.
