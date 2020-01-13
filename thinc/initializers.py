from typing import Callable

from .config import registry
from .types import Array
from .util import get_array_module, copy_array, partial


def xavier_uniform_init(data: Array, *, inplace: bool = False) -> Array:
    xp = get_array_module(data)
    scale = xp.sqrt(6.0 / (data.shape[0] + data.shape[1]))
    if inplace:
        copy_array(data, xp.random.uniform(-scale, scale, data.shape))
        return data
    else:
        return xp.random.uniform(-scale, scale, data.shape)


@registry.initializers("xavier_uniform_init.v0")
def configure_xavier_uniform_init(*, inplace: bool = False) -> Callable[[Array], Array]:
    return partial(xavier_uniform_init, inplace=inplace)


def zero_init(data: Array, *, inplace: bool = False) -> Array:
    if inplace:
        data.fill(0.0)
        return data
    else:
        xp = get_array_module(data)
        return xp.zeros_like(data)


@registry.initializers("zero_init.v0")
def configure_zero_init(*, inplace: bool = False) -> Callable[[Array], Array]:
    return partial(zero_init, inplace=inplace)


def uniform_init(
    data: Array, *, lo: float = -0.1, hi: float = 0.1, inplace: bool = False
) -> Array:
    xp = get_array_module(data)
    values = xp.random.uniform(lo, hi, data.shape)
    if inplace:
        copy_array(data, values)
        return data
    else:
        return values.astype(data.dtype)


@registry.initializers("uniform_init.v0")
def configure_uniform_init(
    *, lo: float = -0.1, hi: float = 0.1, inplace: bool = False
) -> Callable[[Array], Array]:
    return partial(uniform_init, lo=lo, hi=hi, inplace=inplace)


def normal_init(data: Array, *, fan_in: int = -1, inplace: bool = False) -> Array:
    if fan_in == -1:
        fan_in = data.shape[1]
    xp = get_array_module(data)
    scale = xp.sqrt(1.0 / fan_in)
    size = int(xp.prod(data.shape))
    inits = xp.random.normal(scale=scale, size=size)
    inits = inits.reshape(data.shape)
    if inplace:
        copy_array(data, inits)
        return data
    else:
        return inits


@registry.initializers("normal_init.v0")
def configure_normal_init(
    *, fan_in: int = -1, inplace: bool = False
) -> Callable[[Array], Array]:
    return partial(normal_init, fan_in=fan_in, inplace=inplace)


__all__ = ["normal_init", "uniform_init", "xavier_uniform_init", "zero_init"]
