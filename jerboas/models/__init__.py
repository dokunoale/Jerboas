"""Trainable knowledge-graph embedding models.

This is the only part of the library that needs torch, and it is optional:

    pip install jerboas[torch]

Serving does not need it. Training writes a checkpoint of plain arrays, and
strategies.Embedding reads it with numpy alone -- so the machine that fits a
model and the machine that answers queries need not have the same install.

Training is not a pipeline verb. It costs orders of magnitude more than a query
can absorb -- seconds on MovieLens, minutes or more as the graph grows -- so it
is an explicit batch job, run once, outside any query:

    from jerboas.models import TransD, train

    model = train(TransD(factors=64), graph, epochs=50, device="mps")
    model.save("ml.transd.npz")

`train` is a free function rather than a method because `nn.Module.train()`
already means something else in torch -- switching to training mode -- and a
name collision there would fail quietly.
"""

try:
    import torch as _torch
except ModuleNotFoundError as exc:      # pragma: no cover - depends on the install
    raise ModuleNotFoundError(
        "jerboas.models needs torch, which is an optional dependency.\n"
        "Install it with:  pip install 'jerboas[torch]'\n"
        "Serving a model someone else trained does not need torch -- "
        "strategies.Embedding reads a checkpoint with numpy."
    ) from exc

del _torch

from .base import Translational     # noqa: E402
from .transd import TransD          # noqa: E402
from .transe import TransE          # noqa: E402
from .train import train            # noqa: E402

# Every trainable model, by the name its checkpoints carry. Nothing has to be
# kept in step here: a trainer and the ranking path reach the same kge.Model, so
# a model cannot exist without its arithmetic or vice versa.
MODELS = {model.spec.name: model for model in (TransD, TransE)}

__all__ = ["Translational", "TransD", "TransE", "train", "MODELS"]
