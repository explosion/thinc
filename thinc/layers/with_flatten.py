from typing import Tuple, Callable, Sequence, Any, List, TypeVar

from ..model import Model
from ..config import registry
from ..types import Array2d


ItemT = TypeVar("ItemT")
InT = Sequence[Sequence[ItemT]]
OutT = List[Array2d]


@registry.layers("with_flatten.v0")
def with_flatten(layer: Model) -> Model[InT, OutT]:
    return Model(f"with_flatten-{layer.name}", forward, layers=[layer], init=init)


def forward(
    model: Model[InT, OutT], Xnest: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    layer: Model[Sequence[Any], Array2d] = model.layers[0]
    Xflat: Sequence[Any] = _flatten(Xnest)
    Yflat, backprop_layer = layer(Xflat, is_train)
    # Get the split points. We want n-1 splits for n items.
    splits = layer.ops.asarray([len(x) for x in Xnest[:-1]], dtype="i").cumsum()
    Ynest = layer.ops.xp.split(Yflat, splits, axis=-1)

    def backprop(dYnest: OutT) -> InT:
        dYflat: List[Array2d] = []
        for d_item in dYnest:
            dYflat.extend(d_item)
        dXflat = backprop_layer(dYflat)
        dXnest = layer.ops.xp.split(dXflat, splits, axis=-1)
        return dXnest

    return Ynest, backprop


def _flatten(nested: InT) -> List[ItemT]:
    flat: List[ItemT] = []
    for item in nested:
        flat.extend(item)
    return flat


def init(model, X=None, Y=None):
    model.layers[0].initialize(
        _flatten(X) if X is not None else None,
        model.layers[0].ops.xp.hstack(Y) if Y is not None else None,
    )
