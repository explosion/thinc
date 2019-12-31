from typing import Tuple
from ..types import Array
from .. import Model
from ..util import get_width


def CauchySimilarity(nI=None):
    return Model(
        "cauchy_similarity",
        forward,
        init=init,
        dims={"nI": nI, "nO": 1},
        params={"W": None},
    )


def init(model, X=None, Y=None):
    if X is not None:
        model.set_dim("nI", get_width(X))
    # Initialize weights to 1
    W = model.ops.allocate((model.get_dim("nI"),))
    W += 1
    model.set_param("W", W)


def forward(model, X1_X2: Tuple[Array, Array], is_train):
    X1, X2 = X1_X2
    W = model.get_param("W")
    diff = X1 - X2
    square_diff = diff ** 2
    total = (W * square_diff).sum(axis=1)
    sim, bp_sim = inverse(total)

    def backprop(d_sim):
        d_total = bp_sim(d_sim)
        d_total = d_total.reshape((-1, 1))
        model.inc_grad("W", (d_total * square_diff).sum(axis=0))
        d_square_diff = W * d_total
        d_diff = 2 * d_square_diff * diff
        return (d_diff, -d_diff)

    return sim, backprop


def inverse(total):
    inverse = 1.0 / (1 + total)

    def backward(d_inverse):
        return d_inverse * (-1 / (total + 1) ** 2)

    return inverse, backward
