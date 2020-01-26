from typing import TypeVar, cast, List

from .noop import noop
from .chain import chain
from ..model import Model
from ..config import registry


InT = TypeVar("InT")
OutT = TypeVar("OutT")


@registry.layers("clone.v1")
def clone(orig: Model[InT, OutT], n: int) -> Model[InT, OutT]:
    """Construct `n` copies of a layer, with distinct weights.  i.e.
    `clone(f, 3)(x)` computes f(f'(f''(x))).
    """
    if n == 0:
        return cast(Model[InT, OutT], noop())
    elif n == 1:
        return orig
    layers: List[Model] = [orig]
    for i in range(n - 1):
        layers.append(orig.copy())
    return cast(Model[InT, OutT], chain(*layers))
