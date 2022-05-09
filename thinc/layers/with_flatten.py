from typing import Tuple, Callable, Sequence, Any, cast, TypeVar, Optional, List

from ..model import Model
from ..config import registry
from ..types import ListXd


ItemT = TypeVar("ItemT")
InT = Sequence[Sequence[ItemT]]
OutT = TypeVar("OutT", bound=ListXd)


@registry.layers("with_flatten.v1")
def with_flatten(layer: Model[InT, InT]) -> Model[OutT, OutT]:
    return Model(f"with_flatten({layer.name})", forward, layers=[layer], init=init)


def forward(
    model: Model[OutT, OutT], Xnest: OutT, is_train: bool
) -> Tuple[OutT, Callable]:
    layer: Model[InT, InT] = model.layers[0]
    Xflat: Sequence[Sequence[Any]] = _flatten(Xnest)
    Yflat, backprop_layer = layer(Xflat, is_train)
    # Get the split points. We want n-1 splits for n items.
    arr = layer.ops.asarray1i([len(x) for x in Xnest[:-1]])
    splits = arr.cumsum()
    Ynest = layer.ops.xp.split(Yflat, splits, axis=0)

    def backprop(dYnest: OutT) -> OutT:
        dYflat = model.ops.flatten(dYnest)  # type: ignore[arg-type, var-annotated]
        # type ignore necessary for older versions of Mypy/Pydantic
        dXflat = backprop_layer(dYflat)
        dXnest = layer.ops.xp.split(dXflat, splits, axis=-1)
        return dXnest

    return Ynest, backprop


def _flatten(nested: OutT) -> InT:
    flat: List = []
    for item in nested:
        flat.extend(item)
    return cast(InT, flat)


def init(
    model: Model[OutT, OutT], X: Optional[OutT] = None, Y: Optional[OutT] = None
) -> None:
    model.layers[0].initialize(
        _flatten(X) if X is not None else None,
        model.layers[0].ops.xp.hstack(Y) if Y is not None else None,
    )
