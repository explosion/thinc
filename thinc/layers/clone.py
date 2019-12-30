import copy


from .noop import noop
from .chain import chain

# TODO: fix


def clone(orig, n):
    """Construct `n` copies of a layer, with distinct weights.

    i.e. `clone(f, 3)(x)` computes `f(f'(f''(x)))`.
    """
    if n == 0:
        return noop()
    layers = [orig]
    for i in range(n - 1):
        layers.append(copy.deepcopy(orig))
        layers[-1].set_id()
    return chain(*layers)
