"""Fitting an embedding to a Graph.

The dataloader is not a component here, because it does not need to be: the
graph's CSR *is* the triple store. Expanding `out_indptr` gives the head column,
and `out_rels`/`out_indices` are already the relation and tail columns, contiguous
and integer. A batch is a slice of a shuffled index array.

Negative sampling is where knowing the types pays. Corrupting a tail by drawing
uniformly from all N nodes almost always produces a trivially wrong triple -- a
person where a genre belongs -- and the model learns to tell types apart instead
of learning the relation. Drawing from the *type block of the true tail* keeps the
corruption type-correct, so the only way to score it lower is to have learned
something about the relation itself.

Which edges a run learns from is a `where=` of its own. The graph holds every
edge it was given; a training job is entitled to a narrower view of it -- and
that is the honest place for a threshold on a score, because it is a claim about
this fit, not about what the data contains.
"""

import time

import numpy as np
import torch

from ..query.frame import RELATION



def examples(graph, where=None):
    """The triples a run may learn from, as four parallel arrays.

    With no filter, every stored edge, read straight off the out-CSR -- the CSR
    *is* the triple store, which is why this library has no dataloader.

    A filter is a frame of edges, because that is what a frame of edges is for:

        train(model, g, where=g.edges("has_interact").filter(v.score >= 3))

    Below a rating of 3 the edge is noise and the model should not see it. Saying
    it here rather than at load time answers the question for this run only, and
    leaves the graph still able to say who rated a film at all.
    """
    weights = graph.weights(normalized=True)[0]
    if where is None:
        heads = graph.sources()
        return (heads.astype(np.int64), np.asarray(graph.out_rels),
                np.asarray(graph.out_indices), weights)

    frame = where.pl if hasattr(where, "pl") else where
    heads = frame["source"].to_numpy().astype(np.int64)
    tails = frame["target"].to_numpy().astype(np.int64)
    codes = {name: code for code, name in enumerate(graph.relations)}
    rels = np.asarray([codes[name] for name in frame[RELATION].to_list()], dtype=np.int64)
    scores = frame["score"].to_numpy()
    low, high = graph.weight_bounds()
    span = high[rels] - low[rels]
    scaled = np.where(span > 0, (scores - low[rels]) / np.where(span > 0, span, 1.0), 1.0)
    return heads, np.asarray(rels), tails, scaled


def _describe(where):
    """The filter as one line of provenance: six months on, a checkpoint should
    still say which edges it was allowed to see."""
    if where is None:
        return ""
    frame = where.pl if hasattr(where, "pl") else where
    relations = sorted(set(frame[RELATION].to_list()))
    return f"{len(frame)} edges of {', '.join(relations)}"


def type_bounds(graph):
    """Per node, the half-open bounds of its type block -- the range a
    type-correct corruption is drawn from."""
    low = np.empty(graph.n_nodes, dtype=np.int64)
    high = np.empty(graph.n_nodes, dtype=np.int64)
    for type_ in graph.types:
        start, stop = graph.block(type_)
        low[start:stop], high[start:stop] = start, stop
    return low, high


def train(model, graph, epochs=50, batch_size=4096, lr=0.01, device="cpu",
          seed=42, weighted=True, where=None, report=print):
    """Fit `model` to `graph` in place, and return it.

    A batch job, not a pipeline verb: far too slow to sit inside a query's
    fit(), and run once ahead of them all.

    Two ways to tell the run what an edge is worth, and they compose:

        where=g.edges("has_interact").filter(v.score >= 3)   # don't learn from these
        weighted=True                                    # learn less from weak ones

    `where` is the hard reading: below a rating of 3, say, the edge is noise and
    the model should not see it. `weighted` is the soft one -- each example scaled by its edge's
    normalized score, which on a graph carrying none is 1.0 everywhere and
    changes nothing.
    """
    if not model.built:
        model.build(graph)
    model.arrays = None          # the fitted weights, not the ones read out before
    device = torch.device(device)
    model.to(device)

    head, relation, tail, weights = examples(graph, where)
    if len(head) == 0:
        raise ValueError("nothing to train on: the graph has no edges"
                         if where is None else
                         "nothing to train on: the filter frame is empty")
    low, high = type_bounds(graph)

    model.meta.update(epochs=epochs, batch_size=batch_size, lr=lr,
                      device=str(device), sampler="type-aware", weighted=weighted,
                      trained_on=_describe(where) or "every edge",
                      trained_edges=len(head))

    head_t = torch.as_tensor(head, device=device)
    relation_t = torch.as_tensor(relation.astype(np.int64), device=device)
    tail_t = torch.as_tensor(tail.astype(np.int64), device=device)
    weight_t = (torch.as_tensor(weights, device=device, dtype=torch.float32)
                if weighted else None)

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(head)

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            # type-correct corruptions, drawn per example from the endpoint's block
            corrupt_tail = _corrupt(rng, tail[batch], low, high)
            corrupt_head = _corrupt(rng, head[batch], low, high)

            index = torch.as_tensor(batch, device=device)
            loss = model.loss(
                head_t[index], relation_t[index], tail_t[index],
                torch.as_tensor(corrupt_head, device=device),
                torch.as_tensor(corrupt_tail, device=device),
                None if weight_t is None else weight_t[index],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)

        if report:
            report(f"epoch {epoch:3d}/{epochs}  loss {total / n:.4f}  "
                   f"{time.perf_counter() - started:.1f}s")

    model.eval()
    return model


def _corrupt(rng, nodes, low, high):
    """Replace each node with another of the same type."""
    return rng.integers(low[nodes], high[nodes])
