from typing import Tuple, Callable

from ..model import Model
from ..config import registry
from ..types import Floats2d


InT = Floats2d
OutT = Floats2d


@registry.layers("Logistic.v1")
def Logistic() -> Model[InT, OutT]:
    return Model("logistic", forward)


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.sigmoid(X, inplace=False)

    def backprop(dY: OutT) -> InT:
        return dY * model.ops.dsigmoid(Y, inplace=False)

    return Y, backprop
