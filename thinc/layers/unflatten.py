from typing import Tuple
from ..model import Model
from ..types import Array


def unflatten():
    """Transform sequences from a ragged format into lists."""
    return Model("unflatten", forward)


def forward(model: Model, X_lengths: Tuple[Array, Array], is_train: bool):
    X, lengths = X_lengths
    Xs = model.ops.unflatten(X, lengths)

    def backprop(dXs):
        return model.ops.flatten(dXs, pad=0)

    return Xs, backprop
