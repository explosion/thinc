from typing import Tuple, Callable, Optional

from ..model import Model
from ..config import registry
from ..types import Ragged, Array2d


ValT = Array2d


@registry.layers("with_ragged2array.v0")
def with_ragged2array(layer: Model[ValT, ValT]) -> Model[Ragged, Ragged]:
    return Model(
        f"with_ragged2array-{layer.name}",
        forward,
        init=init,
        layers=[layer],
    )


def forward(model: Model[Ragged, Ragged], Xr: Ragged, is_train: bool) -> Tuple[Ragged, Callable]:
    layer: Model[ValT, ValT] = model.layers[0]
    Y, get_dX = layer(Xr.data, is_train)

    def backprop(dYr: Ragged) -> Ragged:
        return Ragged(get_dX(dYr.data), dYr.lengths)

    return Ragged(Y, Xr.lengths), backprop


def init(
    model: Model[Ragged, Ragged], X: Optional[Ragged] = None, Y: Optional[Ragged] = None
) -> None:
    layer: Model[Array2d, Array2d] = model.layers[0]
    layer.initialize(
        X=X.data if X is not None else None,
        Y=Y.data if Y is not None else None
    )
