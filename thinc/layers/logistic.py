from typing import Tuple, Callable

from ..model import Model
from ..config import registry
from ..types import Array2d


InT = Array2d
OutT = Array2d


@registry.layers("Logistic.v0")
def Logistic() -> Model[InT, OutT]:
    return Model("logistic", forward)


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    Y = model.ops.sigmoid(X, inplace=False)

    def backprop(dY: OutT) -> InT:
        return dY * model.ops.dsigmoid(Y, inplace=False)

    return Y, backprop
