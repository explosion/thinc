from typing import Tuple, Callable, Optional, TypeVar, Union, cast

from ..types import Padded, Ragged, Array2d, Floats3d, Ints1d, Floats2d, List2d
from ..model import Model
from ..config import registry
from ..util import is_xp_array


PaddedData = Tuple[Floats3d, Ints1d, Ints1d, Ints1d]
SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List2d, Floats3d, PaddedData])


@registry.layers("with_padded.v1")
def with_padded(layer: Model[Padded, Padded]) -> Model[SeqT, SeqT]:
    return Model(
        f"with_padded({layer.name})",
        forward,
        init=init,
        layers=[layer],
        dims={name: layer.maybe_get_dim(name) for name in layer.dim_names},
    )


def forward(
    model: Model[SeqT, SeqT], Xseq: SeqT, is_train: bool
) -> Tuple[SeqT, Callable]:
    layer: Model[Padded, Padded] = model.layers[0]
    Y: Union[Padded, Ragged, List2d, PaddedData]
    if isinstance(Xseq, Padded):
        Y, backprop = layer(Xseq, is_train)
    elif isinstance(Xseq, Ragged):
        Y, backprop = _ragged_forward(layer, cast(Ragged, Xseq), is_train)
    elif _is_padded_data(Xseq):
        Y, backprop = _tuple_forward(layer, cast(PaddedData, Xseq), is_train)
    elif is_xp_array(Xseq):
        Y, backprop = _array_forward(layer, cast(Floats3d, Xseq), is_train)
    else:
        Y, backprop = _list_forward(layer, cast(List2d, Xseq), is_train)
    return cast(Tuple[SeqT, Callable], (Y, backprop))


def init(
    model: Model[SeqT, SeqT], X: Optional[SeqT] = None, Y: Optional[SeqT] = None
) -> Model[SeqT, SeqT]:
    model.layers[0].initialize(
        X=_get_padded(model, X) if X is not None else None,
        Y=_get_padded(model, Y) if Y is not None else None,
    )
    return model


def _is_padded_data(seq):
    return isinstance(seq, tuple) and len(seq) == 4 and all(map(is_xp_array, seq))


def _get_padded(model, seq):
    if isinstance(seq, Padded):
        return seq
    elif isinstance(seq, Ragged):
        return model.ops.list2padded(model.ops.unflatten(seq.data, seq.lengths))
    elif _is_padded_data(seq):
        return Padded(*seq)  # type: ignore
    elif is_xp_array(seq):
        size_at_t = model.ops.asarray1i([seq.shape[1]] * seq.shape[0])
        lengths = model.ops.asarray1i([seq.shape[0]] * seq.shape[1])
        indices = model.ops.xp.arange(seq.shape[1])
        return Padded(seq, size_at_t, lengths, indices)
    else:
        assert isinstance(seq, list), seq
        return model.ops.list2padded(seq)


def _array_forward(layer, X, is_train):
    # Create bogus metadata for Padded.
    Xp = _get_padded(layer, X)
    Yp, get_dXp = layer(Xp, is_train)
    size_at_t = Xp.size_at_t
    lengths = Xp.lengths
    indices = Xp.indices

    def backprop(dY: Floats3d) -> Floats3d:
        dYp = Padded(dY, size_at_t, lengths, indices)
        dXp = get_dXp(dYp)
        return dXp.data

    return Yp.data, backprop


def _tuple_forward(layer, X, is_train: bool):
    Yp, get_dXp = layer(Padded(*X), is_train)

    def backprop(dY):
        dXp = get_dXp(Padded(*dY))
        return (dXp.data, dXp.size_at_t, dXp.lengths, dXp.indices)

    return (Yp.data, Yp.size_at_t, Yp.lengths, Yp.indices), backprop


def _ragged_forward(layer, Xr, is_train):
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
        flattened = flatten(
            padded2list(get_dXp(list2padded(unflatten(dYr.data, dYr.lengths))))
        )
        return Ragged(cast(Floats2d, flattened), dYr.lengths)

    flattened = flatten(padded2list(Yp))
    return Ragged(flattened, Xr.lengths), backprop


def _list_forward(layer, Xs, is_train):
    # Assign these to locals, to keep code a bit shorter.
    list2padded = layer.ops.list2padded
    padded2list = layer.ops.padded2list

    Yp, get_dXp = layer(list2padded(Xs), is_train)  # type: ignore

    def backprop(dYs):
        return padded2list(get_dXp(list2padded(dYs)))  # type: ignore

    return padded2list(Yp), backprop
