from typing import Tuple
from ..model import Model


def with_getitem(idx: int, layer: Model) -> Model:
    """Transform data on the way into and out of a layer, by plucking an item
    from a tuple.
    """
    return Model(
        f"with_getitem-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        dims={"nO": None, "nI": None},
        attrs={"idx": idx},
    )


def init(model, X=None, Y=None):
    idx = model.get_attr("idx")
    X_i = X[idx] if X is not None else X
    Y_i = Y[idx] if Y is not None else Y
    model.layers[0].initialize(X=X_i, Y=Y_i)


def forward(model: Model, items: Tuple, is_train: bool):
    idx = model.get_attr("idx")
    Y_i, backprop_item = model.layers[0](items[idx], is_train)

    def backprop(d_output: Tuple):
        dY_i = backprop(d_output[idx])
        return d_output[:idx] + (dY_i,) + items[idx + 1 :]

    return items[:idx] + (Y_i,) + items[idx + 1 :], backprop
