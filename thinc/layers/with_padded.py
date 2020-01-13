from typing import Tuple, Callable, List, Optional, TypeVar, Union, cast

from ..types import Padded, Ragged, Array1d, Array2d, Array3d
from ..model import Model
from ..config import registry


PaddedData = Tuple[Array3d, Array1d, List[int], List[int]]
ValT = TypeVar("ValT", bound=Array2d)
SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List[Array2d], PaddedData])


@registry.layers("with_padded.v0")
def with_padded(layer: Model[Padded, Padded]) -> Model[SeqT, SeqT]:
    return Model(f"with_padded-{layer.name}", forward, init=init, layers=[layer])


def forward(
    model: Model[SeqT, SeqT], Xseq: SeqT, is_train: bool
) -> Tuple[SeqT, Callable]:
    layer: Model[Padded, Padded] = model.layers[0]
    Y: Union[Padded, Ragged, List[Array2d], PaddedData]
    if isinstance(Xseq, Padded):
        Y, backprop = layer(Xseq, is_train)
    elif isinstance(Xseq, Ragged):
        Y, backprop = _ragged_forward(layer, cast(Ragged, Xseq), is_train)
    elif _is_padded_data(Xseq):
        Y, backprop = _tuple_forward(layer, cast(PaddedData, Xseq), is_train)
    else:
        Y, backprop = _list_forward(layer, cast(List[Array2d], Xseq), is_train)
    return cast(Tuple[SeqT, Callable], (Y, backprop))


def init(
    model: Model[SeqT, SeqT], X: Optional[SeqT] = None, Y: Optional[SeqT] = None
) -> None:
    model.layers[0].initialize(
        X=_get_padded(model, X) if X is not None else None,
        Y=_get_padded(model, Y) if Y is not None else None,
    )


def _is_padded_data(seq):
    return isinstance(seq, tuple) and len(seq) == 4 and isinstance(seq[3], list)


def _get_padded(model: Model, seq: SeqT) -> Padded:
    if isinstance(seq, Padded):
        return seq
    elif isinstance(seq, Ragged):
        return model.ops.list2padded(model.ops.unflatten(seq.data, seq.lengths))
    elif _is_padded_data(seq):
        return Padded(*cast(PaddedData, seq))
    else:
        return model.ops.list2padded(cast(List[Array2d], seq))


def _tuple_forward(layer: Model[Padded, Padded], X: PaddedData, is_train: bool):
    Yp, get_dXp = layer(Padded(*X), is_train)

    def backprop(dY: PaddedData) -> PaddedData:
        dXp = get_dXp(Padded(*dY))
        return (dXp.data, dXp.size_at_t, dXp.lengths, dXp.indices)

    return (Yp.data, Yp.size_at_t, Yp.lengths, Yp.indices), backprop


def _ragged_forward(
    layer: Model[Padded, Padded], Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    # Assign these to locals, to keep code a bit shorter.
    list2padded = layer.ops.list2padded
    padded2list = layer.ops.padded2list
    unflatten = layer.ops.unflatten
    flatten = layer.ops.flatten
    # It's worth being a bit careful about memory here, as the activations
    # are potentially large on GPU. So we make nested function calls instead
    # of assigning to temporaries where possible, so memory can be reclaimed
    # sooner.
    Yp, get_dXp = layer(list2padded(unflatten(Xr.data, Xr.lengths)), is_train)

    def backprop(dYr: Ragged):
        return Ragged(
            flatten(
                padded2list(get_dXp(list2padded(unflatten(dYr.data, dYr.lengths))))
            ),
            dYr.lengths,
        )

    return Ragged(flatten(padded2list(Yp)), Xr.lengths), backprop


def _list_forward(
    layer: Model[Padded, Padded], Xs: List[Array2d], is_train: bool
) -> Tuple[List[Array2d], Callable]:
    # Assign these to locals, to keep code a bit shorter.
    list2padded = layer.ops.list2padded
    padded2list = layer.ops.padded2list

    Yp, get_dXp = layer(list2padded(Xs), is_train)

    def backprop(dYs: List[Array2d]) -> List[Array2d]:
        return padded2list(get_dXp(list2padded(dYs)))

    return padded2list(Yp), backprop
