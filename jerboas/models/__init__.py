"""Trainable knowledge-graph embedding models.

This is the only part of the library that needs torch, and it is optional:

    pip install jerboas[torch]

Serving does not: training writes a checkpoint of plain arrays that a loaded
model reads with numpy alone, so the machine that fits and the machine that
answers need not have the same install.

Training is not a pipeline verb -- it costs orders of magnitude more than a
query can absorb -- so it is an explicit batch job, and what comes back is a
ranking strategy like any other:

    from jerboas.models import TransD, train

    model = train(TransD(factors=64), graph, epochs=50, device="mps")
    model.save("checkpoints/ml.transd.npz")

    frame.with_columns(
        score=TransD.load("checkpoints/ml.transd.npz", g, to=seeds).on("rec")
    ).top(10)

`train` is a free function because `nn.Module.train()` already means something
else in torch, and a name collision there would fail quietly.
"""

try:
    import torch as _torch
except ModuleNotFoundError as exc:      # pragma: no cover - depends on the install
    raise ModuleNotFoundError(
        "jerboas.models needs torch, which is an optional dependency.\n"
        "Install it with:  pip install 'jerboas[torch]'\n"
        "Serving a model someone else trained does not need torch -- "
        "a loaded checkpoint is read with numpy."
    ) from exc

del _torch

from .base import Translational     # noqa: E402
from .transd import TransD          # noqa: E402
from .transe import TransE          # noqa: E402
from .train import train            # noqa: E402

# Every model, by the name its checkpoints carry. A model is one class -- tables,
# arithmetic, fitting and ranking -- so there is nothing here to keep in step
# with anything else.
MODELS = {model.name: model for model in (TransD, TransE)}

__all__ = ["Translational", "TransD", "TransE", "train", "MODELS"]
