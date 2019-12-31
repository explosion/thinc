from typing import Tuple
from ..types import Array
from ..model import Model


def MaxPool():
    return Model("max_pool", forward)


def forward(model: Model, X_lengths: Tuple[Array, Array], is_train: bool):
    X, lengths = X_lengths

    Y, which = model.ops.max_pool(X, lengths)

    def backprop(dY):
        return model.ops.backprop_max_pool(dY, which, lengths)

    return Y, backprop
