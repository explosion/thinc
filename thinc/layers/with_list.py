from typing import Tuple, Callable, List, Optional, TypeVar, Union, cast

from ..types import Padded, Ragged, Floats2d, List2d
from ..model import Model
from ..config import registry


SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List2d])


@registry.layers("with_list.v1")
def with_list(layer: Model[List2d, List2d]) -> Model[SeqT, SeqT]:
    return Model(f"with_list({layer.name})", forward, init=init, layers=[layer])


def forward(
    model: Model[SeqT, SeqT], Xseq: SeqT, is_train: bool
) -> Tuple[SeqT, Callable]:
    layer: Model[List2d, List2d] = model.layers[0]
    Y: Union[Padded, Ragged, List2d]
    if isinstance(Xseq, Padded):
        Y, backprop = _padded_forward(layer, cast(Padded, Xseq), is_train)
    elif isinstance(Xseq, Ragged):
        Y, backprop = _ragged_forward(layer, cast(Ragged, Xseq), is_train)
    else:
        Y, backprop = layer(cast(List2d, Xseq), is_train)
    return cast(Tuple[SeqT, Callable], (Y, backprop))


def init(
    model: Model[SeqT, SeqT], X: Optional[SeqT] = None, Y: Optional[SeqT] = None
) -> Model[SeqT, SeqT]:
    model.layers[0].initialize(
        X=_get_list(model, X) if X is not None else None,
        Y=_get_list(model, Y) if Y is not None else None,
    )
    return model


def _get_list(model, seq):
    if isinstance(seq, Padded):
        return model.ops.padded2list(seq)
    elif isinstance(seq, Ragged):
        return model.ops.unflatten(seq.data, seq.lengths)
    else:
        return seq


def _ragged_forward(layer, Xr, is_train):
    # Assign these to locals, to keep code a bit shorter.
    unflatten = layer.ops.unflatten
    flatten = layer.ops.flatten
    # It's worth being a bit careful about memory here, as the activations
    # are potentially large on GPU. So we make nested function calls instead
    # of assigning to temporaries where possible, so memory can be reclaimed
    # sooner.
    Ys, get_dXs = layer(unflatten(Xr.data, Xr.lengths), is_train)

    def backprop(dYr: Ragged):
        return Ragged(flatten(get_dXs(unflatten(dYr.data, dYr.lengths))), dYr.lengths)

    return Ragged(flatten(Ys), Xr.lengths), backprop


def _padded_forward(layer, Xp, is_train):
    # Assign these to locals, to keep code a bit shorter.
    padded2list = layer.ops.padded2list
    list2padded = layer.ops.list2padded
    # It's worth being a bit careful about memory here, as the activations
    # are potentially large on GPU. So we make nested function calls instead
    # of assigning to temporaries where possible, so memory can be reclaimed
    # sooner.
    Ys, get_dXs = layer(padded2list(Xp), is_train)

    def backprop(dYp):
        return list2padded(get_dXs(padded2list(dYp)))

    return list2padded(cast(List[Floats2d], Ys)), backprop
