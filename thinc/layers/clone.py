from typing import TypeVar

from .noop import noop
from .chain import chain
from ..model import Model


# TODO: input / output types for model?
InT = TypeVar("InT")
OutT = TypeVar("OutT")


def clone(orig: Model[InT, OutT], n: int) -> Model[InT, OutT]:
    """Construct `n` copies of a layer, with distinct weights.  i.e.
    `clone(f, 3)(x)` computes f(f'(f''(x))).
    """
    if n == 0:
        return noop()
    layers = [orig]
    for i in range(n - 1):
        layers.append(orig.copy())
    return chain(*layers)
