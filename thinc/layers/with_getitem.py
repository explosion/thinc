from typing import Callable, Optional, Tuple, TypeVar

from ..model import Model


InputType = TypeVar("InputType", bound=tuple)
OutputType = TypeVar("OutputType", bound=tuple)


def with_getitem(idx: int, layer: Model) -> Model:
    """Transform data on the way into and out of a layer, by plucking an item
    from a tuple.
    """
    return Model(
        f"with_getitem-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        attrs={"idx": idx},
    )


def forward(
    model: Model, items: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    idx = model.get_attr("idx")
    Y_i, backprop_item = model.layers[0](items[idx], is_train)

    def backprop(d_output: OutputType) -> InputType:
        dY_i = backprop(d_output[idx])
        return d_output[:idx] + (dY_i,) + items[idx + 1 :]

    return items[:idx] + (Y_i,) + items[idx + 1 :], backprop


def init(model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None):
    idx = model.get_attr("idx")
    X_i = X[idx] if X is not None else X
    Y_i = Y[idx] if Y is not None else Y
    model.layers[0].initialize(X=X_i, Y=Y_i)
