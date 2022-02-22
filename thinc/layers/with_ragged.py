from typing import Tuple, Callable, Optional, TypeVar, Union, cast

from ..types import Padded, Ragged, Ints1d, Array2d, List2d
from ..model import Model
from ..config import registry


RaggedData = Tuple[Array2d, Ints1d]
SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List2d, RaggedData])
SeqT_co = TypeVar("SeqT_co", bound=Union[Padded, Ragged, List2d, RaggedData], covariant=True)


@registry.layers("with_ragged.v1")
def with_ragged(layer: Model[Ragged, Ragged]) -> Model[SeqT_co, SeqT_co]:
    return Model(f"with_ragged({layer.name})", forward, init=init, layers=[layer])


def forward(
    model: Model[SeqT_co, SeqT_co], Xseq: SeqT, is_train: bool
) -> Tuple[SeqT, Callable]:
    layer: Model[Ragged, Ragged] = model.layers[0]
    Y: SeqT
    if isinstance(Xseq, Ragged):
        ragged_Y, backprop = layer(Xseq, is_train)
        Y = cast(SeqT, ragged_Y)
    elif isinstance(Xseq, Padded):
        Y, backprop = _padded_forward(layer, cast(Padded, Xseq), is_train)
    elif _is_ragged_data(Xseq):
        Y, backprop = _tuple_forward(layer, cast(RaggedData, Xseq), is_train)
    else:
        Y, backprop = _list_forward(layer, cast(List2d, Xseq), is_train)
    return cast(Tuple[SeqT, Callable], (Y, backprop))


def init(
    model: Model[SeqT_co, SeqT_co], X: Optional[SeqT] = None, Y: Optional[SeqT] = None
) -> None:
    model.layers[0].initialize(
        X=_get_ragged(model, X) if X is not None else None,
        Y=_get_ragged(model, Y) if Y is not None else None,
    )


def _is_ragged_data(seq):
    return isinstance(seq, tuple) and len(seq) == 2


def _get_ragged(model: Model[SeqT_co, SeqT_co], seq: SeqT) -> Ragged:
    if isinstance(seq, Ragged):
        return seq
    elif isinstance(seq, Padded):
        lists = model.ops.padded2list(seq)
        lengths = model.ops.asarray1i([len(x) for x in lists])
        return Ragged(model.ops.flatten(lists), lengths) # type: ignore[arg-type, type-var]
    elif _is_ragged_data(seq):
        return Ragged(*seq) # type: ignore[misc]
    else:
        lengths = model.ops.asarray1i([len(x) for x in seq]) # type: ignore[union-attr]
        return Ragged(model.ops.flatten(seq), lengths) # type: ignore[arg-type]


def _tuple_forward(layer: Model[Ragged, Ragged], X: RaggedData, is_train: bool) -> Tuple[SeqT, Callable]:
    Yr, get_dXr = layer(Ragged(*X), is_train)

    def backprop(dY: RaggedData) -> RaggedData:
        dXr = get_dXr(Ragged(*dY))
        return (dXr.data, dXr.lengths)

    return cast(SeqT, (Yr.data, Yr.lengths)), backprop


def _padded_forward(layer: Model[Ragged, Ragged], Xp: Padded, is_train: bool) -> Tuple[SeqT, Callable]:
    # Assign these to locals, to keep code a bit shorter.
    list2padded = layer.ops.list2padded
    padded2list = layer.ops.padded2list
    unflatten = layer.ops.unflatten
    flatten = layer.ops.flatten
    # It's worth being a bit careful about memory here, as the activations
    # are potentially large on GPU. So we make nested function calls instead
    # of assigning to temporaries where possible, so memory can be reclaimed
    # sooner.
    Xs = padded2list(Xp)
    # Bit annoying here: padded is in a different order, so we need to make new
    # lengths.
    lengths = layer.ops.asarray1i([len(x) for x in Xs])
    Yr, get_dXr = layer(Ragged(flatten(Xs), lengths), is_train) # type: ignore[type-var]

    def backprop(dYp: Padded):
        flattened = flatten(padded2list(dYp)) # type: ignore[type-var]
        return list2padded(unflatten(get_dXr(Ragged(flattened, lengths)).data, lengths))

    return cast(SeqT, list2padded(unflatten(Yr.data, Yr.lengths))), backprop # type:ignore[arg-type]


def _list_forward(layer: Model[Ragged, Ragged], Xs: List2d, is_train: bool) -> Tuple[SeqT, Callable]:
    # Assign these to locals, to keep code a bit shorter.
    flatten = layer.ops.flatten
    unflatten = layer.ops.unflatten

    lengths = layer.ops.asarray1i([len(x) for x in Xs])
    Yr, get_dXr = layer(Ragged(flatten(Xs), lengths), is_train) # type: ignore [type-var]

    def backprop(dYs):
        flattened = flatten(dYs)
        return unflatten(get_dXr(Ragged(flattened, lengths)).data, lengths)

    return cast(SeqT, unflatten(Yr.data, Yr.lengths)), backprop # type:ignore[arg-type]
