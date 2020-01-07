from .types import Array
from .util import get_array_module, copy_array


def xavier_uniform_init(data: Array, *, inplace: bool = False) -> Array:
    xp = get_array_module(data)
    scale = xp.sqrt(6.0 / (data.shape[0] + data.shape[1]))
    if inplace:
        copy_array(data, xp.random.uniform(-scale, scale, data.shape))
        return data
    else:
        return xp.random.uniform(-scale, scale, data.shape)


def zero_init(data: Array, *, inplace: bool = False) -> Array:
    if inplace:
        data.fill(0.0)
        return data
    else:
        xp = get_array_module(data)
        return xp.zeroslike(data)


def uniform_init(
    data: Array, lo: float = -0.1, hi: float = 0.1, *, inplace: bool = False
) -> Array:
    xp = get_array_module(data)
    values = xp.random.uniform(lo, hi, data.shape)
    if inplace:
        copy_array(data, values)
        return data
    else:
        return values.astype(data.dtype)


def normal_init(data: Array, fan_in: int, *, inplace: bool = True) -> Array:
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


__all__ = ["normal_init", "uniform_init", "xavier_uniform_init", "zero_init"]
