from typing import Tuple, Callable, List, Optional

from ..types import Array, Padded
from ..model import Model
from ..backends import Ops
from ..config import registry


InT = List[Array]


@registry.layers("with_list2padded.v0")
def with_list2padded(layer: Model[Padded, Padded]) -> Model[InT, InT]:
    return Model(f"with_list2padded-{layer.name}", forward, init=init, layers=[layer])


def forward(model: Model[InT, InT], Xs: InT, is_train: bool) -> Tuple[InT, Callable]:
    # Pad out batches, and sort by decreasing length. The size_at_t array records
    # the number of batch items that are still active at timestep t.
    # We undo this transformation
    X_data, size_at_t, unpad = model.ops.square_sequences(Xs)
    Yp, backprop_layer = model.layers[0](Padded(X_data, size_at_t), is_train)

    def backprop(dYs: InT) -> InT:
        dY_data, size_at_t, unpad = model.ops.square_sequences(dYs)
        dYp = backprop_layer(Padded(dY_data, size_at_t))
        return unpad(dYp.data)

    return unpad(Yp.data), backprop


def init(
    model: Model[InT, InT], X: Optional[InT] = None, Y: Optional[InT] = None
) -> None:
    model.layers[0].initialize(
        X=_maybe_get_padded(model.ops, X), Y=_maybe_get_padded(model.ops, Y)
    )


def _maybe_get_padded(ops: Ops, seqs: Optional[InT]) -> Optional[Padded]:
    if seqs is None:
        return None
    flat, size_at_t, _ = ops.square_sequences(seqs)
    return Padded(flat, size_at_t)
