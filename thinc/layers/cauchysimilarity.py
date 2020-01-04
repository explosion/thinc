from typing import Tuple, Callable, TypeVar, Optional

from ..model import Model
from ..types import Array
from ..util import get_width


InputValue = TypeVar("InputValue", bound=Array)
InputType = Tuple[InputValue, InputValue]
OutputType = TypeVar("OutputType", bound=Array)


def CauchySimilarity(nI: Optional[int] = None) -> Model:
    """Compare input vectors according to the Cauchy similarity function proposed by
    Chen (2013). Primarily used within Siamese neural networks.
    """
    return Model(
        "cauchy_similarity",
        forward,
        init=init,
        dims={"nI": nI, "nO": 1},
        params={"W": None},
    )


def forward(model, X1_X2: InputType, is_train: bool = False) -> Tuple[Array, Callable]:
    X1, X2 = X1_X2
    W = model.get_param("W")
    diff = X1 - X2
    square_diff = diff ** 2
    total = (W * square_diff).sum(axis=1)
    sim, bp_sim = inverse(total)

    def backprop(d_sim: OutputType) -> InputType:
        d_total = bp_sim(d_sim)
        d_total = d_total.reshape((-1, 1))
        model.inc_grad("W", (d_total * square_diff).sum(axis=0))
        d_square_diff = W * d_total
        d_diff = 2 * d_square_diff * diff
        return (d_diff, -d_diff)

    return sim, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    if X is not None:
        model.set_dim("nI", get_width(X[0]))
    # Initialize weights to 1
    W = model.ops.allocate((model.get_dim("nI"),))
    W += 1
    model.set_param("W", W)


def inverse(total: Array) -> Tuple[Array, Callable]:
    inv = 1.0 / (1 + total)

    def backward(d_inverse: Array) -> Array:
        return d_inverse * (-1 / (total + 1) ** 2)

    return inv, backward
