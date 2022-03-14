from typing import Tuple, Callable, Optional, TypeVar, cast, List, Union

from ..types import Padded, Ragged, Array2d, ListXd, FloatsXd, IntsXd, Floats1d
from ..types import Floats2d, Floats3d, Floats4d, Ints1d, Ints2d, Ints3d, Ints4d
from ..model import Model
from ..config import registry

RaggedData = Tuple[Array2d, Ints1d]
SeqT = TypeVar(
    "SeqT",
    bound=Union[
        Padded,
        Ragged,
        ListXd,
        List[Floats1d],
        List[Floats2d],
        List[Floats3d],
        List[Floats4d],
        List[FloatsXd],
        List[Ints1d],
        List[Ints2d],
        List[Ints3d],
        List[Ints4d],
        List[IntsXd],
        RaggedData,
    ],
)
SeqT_co = TypeVar(
    "SeqT_co",
    bound=Union[
        Padded,
        Ragged,
        ListXd,
        List[Floats1d],
        List[Floats2d],
        List[Floats3d],
        List[Floats4d],
        List[FloatsXd],
        List[Ints1d],
        List[Ints2d],
        List[Ints3d],
        List[Ints4d],
        List[IntsXd],
        RaggedData,
    ],
    covariant=True,
)


@registry.layers("with_ragged.v1")
def with_ragged(layer: Model[Ragged, Ragged]) -> Model[SeqT_co, SeqT_co]:
    return Model(f"with_ragged({layer.name})", forward, init=init, layers=[layer])


def forward(
    model: Model[SeqT_co, SeqT_co], Xseq: SeqT, is_train: bool
) -> Tuple[SeqT, Callable]:
    layer: Model[Ragged, Ragged] = model.layers[0]
    Y: SeqT_co
    if isinstance(Xseq, Ragged):
        ragged_Y, backprop = layer(Xseq, is_train)
        Y = cast(SeqT_co, ragged_Y)
    elif isinstance(Xseq, Padded):
        Y, backprop = _padded_forward(layer, cast(Padded, Xseq), is_train)
    elif _is_ragged_data(Xseq):
        Y, backprop = _tuple_forward(layer, cast(RaggedData, Xseq), is_train)
    else:
        Y, backprop = _list_forward(layer, cast(List, Xseq), is_train)
    return cast(Tuple[SeqT, Callable], (Y, backprop))


def init(
    model: Model[SeqT_co, SeqT_co],
    X: Optional[SeqT_co] = None,
    Y: Optional[SeqT_co] = None,
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
        k = model.ops.flatten(lists)
        return Ragged(model.ops.flatten(lists), lengths)
    elif _is_ragged_data(seq):
        return Ragged(*seq)  # type: ignore[misc]
    else:
        list2d_seq = cast(List[Array2d], seq)
        lengths = model.ops.asarray1i([len(x) for x in list2d_seq])
        return Ragged(model.ops.flatten(list2d_seq), lengths)


def _tuple_forward(
    layer: Model[Ragged, Ragged], X: RaggedData, is_train: bool
) -> Tuple[SeqT, Callable]:
    Yr, get_dXr = layer(Ragged(*X), is_train)

    def backprop(dY: RaggedData) -> RaggedData:
        dXr = get_dXr(Ragged(*dY))
        return (dXr.data, dXr.lengths)

    return cast(SeqT, (Yr.data, Yr.lengths)), backprop


def _padded_forward(
    layer: Model[Ragged, Ragged], Xp: Padded, is_train: bool
) -> Tuple[SeqT, Callable]:
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
    Yr, get_dXr = layer(Ragged(flatten(Xs), lengths), is_train)

    def backprop(dYp: Padded):
        flattened = flatten(padded2list(dYp))
        dXr = get_dXr(Ragged(flattened, lengths))
        return list2padded(unflatten(dXr.data, lengths))

    return (
        cast(SeqT, list2padded(unflatten(Yr.data, Yr.lengths))),
        backprop,
    )


def _list_forward(
    layer: Model[Ragged, Ragged], Xs: List, is_train: bool
) -> Tuple[SeqT, Callable]:
    # Assign these to locals, to keep code a bit shorter.
    flatten = layer.ops.flatten
    unflatten = layer.ops.unflatten

    lengths = layer.ops.asarray1i([len(x) for x in Xs])
    Yr, get_dXr = layer(Ragged(flatten(Xs), lengths), is_train)

    def backprop(dYs):
        flattened = flatten(dYs)
        return unflatten(get_dXr(Ragged(flattened, lengths)).data, lengths)

    return cast(SeqT, unflatten(Yr.data, Yr.lengths)), backprop
