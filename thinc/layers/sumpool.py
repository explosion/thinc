from typing import Tuple
from ..types import Array
from ..model import Model


def SumPool():
    return Model("sum_pool", forward)


def forward(model: Model, X_lengths: Tuple[Array, Array], is_train: bool):
    X, lengths = X_lengths

    Y = model.ops.sum_pool(X, lengths)

    def backprop(dY):
        return (model.ops.backprop_sum_pool(dY, lengths), lengths)

    return Y, backprop
