from typing import Tuple, Callable

from ..model import Model
from ..config import registry
from ..types import Floats2d


InT = Floats2d
OutT = Floats2d


@registry.layers("softmax_activation.v1")
def softmax_activation() -> Model[InT, OutT]:
    return Model("softmax_activation", forward)


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.softmax(X, inplace=False)

    def backprop(dY: OutT) -> InT:
        return dY

    return Y, backprop
