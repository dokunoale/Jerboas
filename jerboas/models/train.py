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

from ..refs import Edge


def triples(graph, keep=None):
    """(head, relation, tail) as three parallel arrays, read off the CSR --
    narrowed to `keep`, a boolean over stored edges, when there is one."""
    heads = np.repeat(np.arange(graph.n_nodes), np.diff(graph.out_indptr))
    rels, tails = np.asarray(graph.out_rels), np.asarray(graph.out_indices)
    if keep is None:
        return heads, rels, tails
    return heads[keep], rels[keep], tails[keep]


def admitted(graph, where):
    """Which stored edges a run may read, from a list of `Edge` markers.

    A marker constrains *its own* relation and says nothing about the others:
    `Edge("has_interact", score=(3, None))` drops the interactions not worth
    learning from and leaves the knowledge graph untouched. One naming no
    relation constrains every edge.

    Reuses the query's machinery rather than reimplementing it -- the same
    `graph.admits` masks the engine walks by -- so a filter means the same thing
    at training time as it does in a where()."""
    markers = [where] if isinstance(where, Edge) else list(where or ())
    if not markers:
        return None
    keep = np.ones(len(graph.out_indices), dtype=bool)
    for marker in markers:
        admits = None
        for condition in marker.filters:               # Compare(EdgeScore, op, value)
            mask = graph.admits(condition.op, condition.value, condition.left.normalized)[0]
            admits = mask if admits is None else (admits & mask)
        if admits is not None:
            # an edge outside the marker's relation is not what it speaks about,
            # so it passes rather than being judged by someone else's scale
            keep &= admits | ~_scope(graph, marker.relations)
    return keep


def _scope(graph, relations):
    """The stored edges a marker is talking about: its relations, or all of them
    when it names none."""
    if not relations:
        return np.ones(len(graph.out_rels), dtype=bool)
    codes = [code for code in (graph.relation_code(name) for name in relations)
             if code is not None]
    return np.isin(graph.out_rels, codes)


def _describe(where):
    """The filter as one line of provenance: six months on, a checkpoint should
    still say which edges it was allowed to see."""
    markers = [where] if isinstance(where, Edge) else list(where or ())
    parts = []
    for marker in markers:
        name = "|".join(marker.relations) or "*"
        for condition in marker.filters:
            scale = ".norm()" if condition.left.normalized else ""
            parts.append(f"{name}.score{scale} {condition.op} {condition.value}")
    return ", ".join(parts)


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

        where=[Edge("has_interact", score=(3, None))]   # don't learn from these
        weighted=True                                    # learn less from weak ones

    `where` is the hard reading, and it is the one that used to live in the
    loader: below a rating of 3, say, the edge is noise and the model should not
    see it. `weighted` is the soft one -- each example scaled by its edge's
    normalized score, which on a graph carrying none is 1.0 everywhere and
    changes nothing.
    """
    if not model.built:
        model.build(graph)
    device = torch.device(device)
    model.to(device)

    keep = admitted(graph, where)
    head, relation, tail = triples(graph, keep)
    if len(head) == 0:
        raise ValueError("nothing to train on: the graph has no edges"
                         if where is None else
                         f"nothing to train on: no edge satisfies {_describe(where)}")
    low, high = type_bounds(graph)

    model.meta.update(epochs=epochs, batch_size=batch_size, lr=lr,
                      device=str(device), sampler="type-aware", weighted=weighted,
                      trained_on=_describe(where) or "every edge",
                      trained_edges=len(head))

    head_t = torch.as_tensor(head, device=device)
    relation_t = torch.as_tensor(relation.astype(np.int64), device=device)
    tail_t = torch.as_tensor(tail.astype(np.int64), device=device)
    # aligned with the triples above: both are the out-CSR read in its own order
    weights = graph.weights(normalized=True)[0]
    weight_t = (torch.as_tensor(weights if keep is None else weights[keep],
                                device=device, dtype=torch.float32) if weighted else None)

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
