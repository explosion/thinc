from typing import List
from ..types import Array
from ..model import Model


def flatten_add_lengths():
    """Transform sequences to ragged arrays if necessary. If sequences are
    already ragged, do nothing. A ragged array is a tuple (data, lengths),
    where data is the concatenated data.
    """
    return Model("flatten_add_lengths", forward)


def forward(model: Model, seqs: List[Array], is_train: bool):
    lengths = model.ops.asarray([len(seq) for seq in seqs], dtype="i")

    def backprop(dY):
        return model.ops.unflatten(dY, lengths)

    return (model.ops.flatten(seqs), lengths), backprop
