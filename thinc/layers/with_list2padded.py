from typing import Tuple, Callable, List, Optional

from ..types import Padded, Array2d
from ..model import Model
from ..config import registry


InT = List[Array2d]


@registry.layers("with_list2padded.v0")
def with_list2padded(layer: Model[Padded, Padded]) -> Model[InT, InT]:
    return Model(f"with_list2padded-{layer.name}", forward, init=init, layers=[layer])


def forward(model: Model[InT, InT], Xs: InT, is_train: bool) -> Tuple[InT, Callable]:
    layer: Model[Padded, Padded] = model.layers[0]
    Xp = model.ops.list2padded(Xs)
    Yp, backprop_layer = layer(Xp, is_train)

    def backprop(dYs: InT) -> InT:
        dYp = model.ops.list2padded(dYs)
        dXp = backprop_layer(dYp)
        return model.ops.padded2list(dXp)

    return model.ops.padded2list(Yp), backprop


def init(
    model: Model[InT, InT], X: Optional[InT] = None, Y: Optional[InT] = None
) -> None:
    model.layers[0].initialize(
        X=model.ops.list2padded(X) if X is not None else None,
        Y=model.ops.list2padded(Y) if Y is not None else None,
    )
