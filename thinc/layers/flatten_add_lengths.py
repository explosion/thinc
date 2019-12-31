from typing import Tuple, List, Callable, TypeVar

from ..model import Model
from ..types import Array


InputType = TypeVar("InputType", bound=List[Array])
OutputType = TypeVar("OutputType", bound=Tuple[Array, Array])


def flatten_add_lengths() -> Model:
    """Transform sequences to ragged arrays if necessary. If sequences are
    already ragged, do nothing. A ragged array is a tuple (data, lengths),
    where data is the concatenated data.
    """
    return Model("flatten_add_lengths", forward)


def forward(
    model: Model, seqs: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    lengths = model.ops.asarray([len(seq) for seq in seqs], dtype="i")

    def backprop(dY: OutputType) -> InputType:
        return model.ops.unflatten(dY, lengths)

    return (model.ops.flatten(seqs), lengths), backprop
