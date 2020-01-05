from .noop import noop
from .chain import chain
from ..model import Model


# TODO: input / output types for model?


def clone(orig: Model, n: int) -> Model:
    """Construct `n` copies of a layer, with distinct weights.  i.e.
    `clone(f, 3)(x)` computes f(f'(f''(x))).
    """
    if n == 0:
        return noop()
    layers = [orig]
    for i in range(n - 1):
        layers.append(orig.copy())
    return chain(*layers)
