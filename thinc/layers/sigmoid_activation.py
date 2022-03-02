from typing import TypeVar, Tuple, Callable, cast

from ..model import Model
from ..config import registry
from ..types import FloatsXd


InT = TypeVar("InT", bound=FloatsXd)
InT_co = TypeVar("InT_co", bound=FloatsXd, covariant=True)


@registry.layers("sigmoid_activation.v1")
def sigmoid_activation() -> Model[InT_co, InT_co]:
    return Model("sigmoid_activation", forward)


def forward(
    model: Model[InT_co, InT_co], X: InT, is_train: bool
) -> Tuple[InT, Callable]:
    Y = model.ops.sigmoid(X, inplace=False)

    def backprop(dY: InT) -> InT:
        return cast(
            InT, dY * model.ops.dsigmoid(Y, inplace=False)
        )  # type:ignore[operator]

    return Y, backprop
