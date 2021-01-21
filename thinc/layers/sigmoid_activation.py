from typing import TypeVar, Tuple, Callable, cast

from ..model import Model
from ..config import registry
from ..types import FloatsXd


InT = TypeVar("InT", bound=FloatsXd)


@registry.layers("sigmoid_activation.v1")
def sigmoid_activation() -> Model[InT, InT]:
    return Model("sigmoid_activation", forward)


def forward(model: Model[InT, InT], X: InT, is_train: bool) -> Tuple[InT, Callable]:
    Y = model.ops.sigmoid(X, inplace=False)

    def backprop(dY: InT) -> InT:
        return dY * model.ops.dsigmoid(Y, inplace=False) # type: ignore

    return Y, backprop
