from typing import Tuple, Callable, Optional, TypeVar
import numpy

from ..model import Model
from ..config import registry
from ..types import Array


InT = TypeVar("InT", bound=Array)
OutT = TypeVar("OutT", bound=Array)


@registry.layers("uniqued.v0")
def uniqued(layer: Model, *, column: int = 0) -> Model[InT, OutT]:
    """Group inputs to a layer, so that the layer only has to compute for the
    unique values. The data is transformed back before output, and the same
    transformation is applied for the gradient. Effectively, this is a cache
    local to each minibatch.
    """
    return Model(
        f"uniqued-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        dims={"nO": None, "nI": None},
        attrs={"column": column},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    column = model.get_attr("column")
    layer = model.layers[0]
    keys = X[:, column]
    if not isinstance(keys, numpy.ndarray):
        keys = keys.get()  # pragma: no cover
    uniq_keys, ind, inv, counts = layer.ops.xp.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    counts = counts.reshape((-1, 1))
    X_uniq = X[ind]
    Y_uniq, bp_Y_uniq = layer(X_uniq, is_train)
    Y = Y_uniq[inv].reshape((X.shape[0],) + Y_uniq.shape[1:])
    uniq_shape = tuple(Y_uniq.shape)

    def backprop(dY: OutT) -> InT:
        dY_uniq = layer.ops.alloc(uniq_shape, dtype="f")
        layer.ops.scatter_add(dY_uniq, layer.ops.asarray(inv, dtype="i"), dY)
        d_uniques = bp_Y_uniq(dY_uniq)
        # This confusing bit of indexing "ununiques"
        return (d_uniques / counts)[inv]

    return Y, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    layer = model.layers[0]
    layer.initialize(X=X, Y=Y)
    if layer.has_dim("nI"):
        model.set_dim("nI", layer.get_dim("nI"))  # pragma: no cover
    if layer.has_dim("nO"):
        model.set_dim("nO", layer.get_dim("nO"))
    return model
