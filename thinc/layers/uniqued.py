from typing import Tuple, Callable, Optional
import numpy

from ..model import Model
from ..types import Array


def uniqued(layer: Model, column: int = 0) -> Model:
    """Group inputs to a layer, so that the layer only has to compute
    for the unique values. The data is transformed back before output, and the same
    transformation is applied for the gradient. Effectively, this is a cache
    local to each minibatch.

    The uniqued wrapper is useful for word inputs, because common words are
    seen often, but we may want to compute complicated features for the words,
    using e.g. character LSTM.
    """
    return Model(
        f"uniqued-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        dims={"nO": None, "nI": None},
        attrs={"column": column},
    )


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    layer = model.layers[0]
    layer.initialize(X=X, Y=Y)
    model.set_dim("nI", layer.get_dim("nI"))
    model.set_dim("nO", layer.get_dim("nO"))


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    column = model.get_attr("column")
    layer = model.layers[0]
    keys = X[:, column]
    keys = layer.ops.xp.ascontiguousarray(keys)
    if not isinstance(keys, numpy.ndarray):
        keys = keys.get()
    uniq_keys, ind, inv, counts = numpy.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    X_uniq = layer.ops.xp.ascontiguousarray(X[ind])
    Y_uniq, bp_Y_uniq = layer(X_uniq, is_train)
    Y = Y_uniq[inv].reshape((X.shape[0],) + Y_uniq.shape[1:])

    def backprop(dY: Array) -> Array:
        dY_uniq = layer.ops.allocate(Y_uniq.shape, dtype="f")
        layer.ops.scatter_add(dY_uniq, layer.ops.asarray(inv, dtype="i"), dY)
        d_uniques = bp_Y_uniq(dY_uniq)
        # This confusing bit of indexing "ununiques"
        return (d_uniques / counts)[inv]

    return Y, backprop
