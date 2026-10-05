"""Trainable knowledge-graph embedding models.

This is the only part of the library that needs torch, and it is optional:

    pip install jerboas[torch]

Loading one needs it too: TransD and TransE are torch modules, so the machine
that serves a checkpoint installs the same extra as the one that fitted it. The
checkpoint itself is plain arrays, loaded with allow_pickle=False.

Training is not a pipeline verb -- it costs orders of magnitude more than a
query can absorb -- so it is an explicit batch job, and what comes back is a
ranking strategy like any other:

    from jerboas.learn import TransD, train

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
        "jerboas.learn needs torch, which is an optional dependency.\n"
        "Install it with:  pip install 'jerboas[torch]'"
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
