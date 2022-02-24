from typing import Tuple, Callable, List, Optional, TypeVar, Union, cast, Sequence

from ..types import Padded, Ragged, Floats2d, List2d, Array2d
from ..model import Model
from ..config import registry

List2d_co = TypeVar("List2d_co", bound=Union[List2d, List[Array2d]], covariant=True)
SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List2d, List[Array2d]])
SeqT_co = TypeVar(
    "SeqT_co", bound=Union[Padded, Ragged, List2d, List[Array2d]], covariant=True
)


@registry.layers("with_list.v1")
def with_list(layer: Model[List2d_co, List2d_co]) -> Model[SeqT_co, SeqT_co]:
    return Model(
        f"with_list({layer.name})",
        forward,
        init=init,
        layers=[layer],
        dims={name: layer.maybe_get_dim(name) for name in layer.dim_names},
    )


def forward(
    model: Model[SeqT_co, SeqT_co], Xseq: SeqT, is_train: bool
) -> Tuple[SeqT, Callable]:
    layer: Model[List[Array2d], List[Array2d]] = model.layers[0]
    Y: SeqT
    if isinstance(Xseq, Padded):
        Y, backprop = _padded_forward(layer, cast(Padded, Xseq), is_train)
    elif isinstance(Xseq, Ragged):
        Y, backprop = _ragged_forward(layer, cast(Ragged, Xseq), is_train)
    else:
        concrete_Y, backprop = layer(cast(List[Array2d], Xseq), is_train)
        Y = cast(SeqT, concrete_Y)
    return Y, backprop


def init(
    model: Model[SeqT_co, SeqT_co], X: Optional[SeqT] = None, Y: Optional[SeqT] = None
) -> Model[SeqT_co, SeqT_co]:
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


def _ragged_forward(
    layer: Model[List[Array2d], List[Array2d]], Xr: Ragged, is_train: bool
) -> Tuple[SeqT, Callable]:
    # Assign these to locals, to keep code a bit shorter.
    unflatten = layer.ops.unflatten
    flatten = layer.ops.flatten
    # It's worth being a bit careful about memory here, as the activations
    # are potentially large on GPU. So we make nested function calls instead
    # of assigning to temporaries where possible, so memory can be reclaimed
    # sooner.
    Ys, get_dXs = layer(cast(List[Array2d], unflatten(Xr.data, Xr.lengths)), is_train)

    def backprop(dYr: Ragged):
        return Ragged(
            flatten(get_dXs(unflatten(dYr.data, dYr.lengths))),
            dYr.lengths,
        )

    return (
        cast(SeqT, Ragged(flatten(Ys), Xr.lengths)),
        backprop,
    )


def _padded_forward(
    layer: Model[List[Array2d], List[Array2d]], Xp: Padded, is_train: bool
) -> Tuple[SeqT, Callable]:
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

    return cast(SeqT, list2padded(cast(List[Floats2d], Ys))), backprop
