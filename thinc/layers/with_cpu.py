from typing import Tuple, Callable, Any, Dict, Optional

import numpy
from thinc.backends import Ops

from ..model import Model
from ..config import registry


@registry.layers("with_cpu.v1")
def with_cpu(layer: Model, ops: Ops) -> Model:
    layer.to_cpu()
    dims: Dict[str, Optional[int]] = {"nO": None}
    # set input dimension only if included layer has one - should be "False" otherwise
    if layer.has_dim("nI") is True:
        dims["nI"] = layer.get_dim("nI")
    if layer.has_dim("nI") is None:
        dims["nI"] = None
    # set output dimension according to included layer
    if layer.has_dim("nO") is True:
        dims["nO"] = layer.get_dim("nO")
    return Model(f"with_cpu({layer.name})", forward, layers=[layer], ops=ops, init=init, dims=dims)


def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
    cpu_outputs, backprop = model.layers[0].begin_update(_to_cpu(X))
    gpu_outputs = _to_device(model.ops, cpu_outputs)

    def with_cpu_backprop(d_outputs):
        cpu_d_outputs = _to_cpu(d_outputs)
        return backprop(cpu_d_outputs)

    return gpu_outputs, with_cpu_backprop


def init(model: Model, X: Any, Y: Any) -> Model:
    return model.layers[0].initialize(X, Y)


def _to_cpu(X):
    if isinstance(X, numpy.ndarray):
        return X
    elif isinstance(X, tuple):
        return tuple([_to_cpu(x) for x in X])
    elif isinstance(X, list):
        return [_to_cpu(x) for x in X]
    elif hasattr(X, "get"):
        return X.get()
    else:
        return X


def _to_device(ops, X):
    if isinstance(X, tuple):
        return tuple([_to_device(ops, x) for x in X])
    elif isinstance(X, list):
        return [_to_device(ops, x) for x in X]
    else:
        return ops.asarray(X)
