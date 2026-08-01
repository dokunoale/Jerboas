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
"""

import time

import numpy as np
import torch


def triples(graph):
    """(head, relation, tail) as three parallel arrays, read off the CSR."""
    heads = np.repeat(np.arange(graph.n_nodes), np.diff(graph.out_indptr))
    return heads, np.asarray(graph.out_rels), np.asarray(graph.out_indices)


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
          seed=42, report=print):
    """Fit `model` to `graph` in place, and return it.

    A batch job, not a pipeline verb: far too slow to sit inside a query's
    fit(), and run once ahead of them all.
    """
    if not model.built:
        model.build(graph)
    model.meta.update(epochs=epochs, batch_size=batch_size, lr=lr,
                      device=str(device), sampler="type-aware")
    device = torch.device(device)
    model.to(device)

    head, relation, tail = triples(graph)
    if len(head) == 0:
        raise ValueError("graph has no edges to train on")
    low, high = type_bounds(graph)

    head_t = torch.as_tensor(head, device=device)
    relation_t = torch.as_tensor(relation.astype(np.int64), device=device)
    tail_t = torch.as_tensor(tail.astype(np.int64), device=device)

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
