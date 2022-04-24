from typing import Tuple, Callable, Sequence, Any, List, TypeVar

from ..model import Model
from ..config import registry
from ..types import Array2d, List2d


ItemT = TypeVar("ItemT")
InT = Sequence[Sequence[ItemT]]
OutT = List2d


@registry.layers("with_flatten.v1")
def with_flatten(layer: Model) -> Model[InT, OutT]:
    return Model(f"with_flatten({layer.name})", forward, layers=[layer], init=init)


def forward(
    model: Model[InT, OutT], Xnest: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    layer: Model[Sequence[Any], Array2d] = model.layers[0]
    Xflat: Sequence[Any] = _flatten(Xnest)
    Yflat, backprop_layer = layer(Xflat, is_train)
    # Get the split points. We want n-1 splits for n items.
    arr = layer.ops.asarray1i([len(x) for x in Xnest[:-1]])
    splits = arr.cumsum()
    Ynest = layer.ops.xp.split(Yflat, splits, axis=0)

    def backprop(dYnest: OutT) -> InT:
        # I think the input/output types might be wrong here?
        dYflat = model.ops.flatten(dYnest)  # type: ignore
        dXflat = backprop_layer(dYflat)
        dXnest = layer.ops.xp.split(dXflat, splits, axis=-1)
        return dXnest

    return Ynest, backprop


def _flatten(nested: InT) -> List[ItemT]:
    flat: List[ItemT] = []
    for item in nested:
        flat.extend(item)
    return flat


def init(model, X=None, Y=None) -> None:
    model.layers[0].initialize(
        _flatten(X) if X is not None else None,
        model.layers[0].ops.xp.hstack(Y) if Y is not None else None,
    )
