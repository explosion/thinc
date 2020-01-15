from typing import Callable

from .backends import Ops
from .config import registry
from .types import Array, Shape
from .util import copy_array, partial


def xavier_uniform_init(ops: Ops, shape: Shape) -> Array:
    scale = ops.xp.sqrt(6.0 / (shape[0] + shape[1]))
    return ops.xp.random.uniform(-scale, scale, shape).astype("float32")


@registry.initializers("xavier_uniform_init.v0")
def configure_xavier_uniform_init(*, inplace: bool = False) -> Callable[[Shape], Array]:
    return partial(xavier_uniform_init, inplace=inplace)


def zero_init(ops: Ops, shape: Shape) -> Array:
    return ops.alloc(*shape)


@registry.initializers("zero_init.v0")
def configure_zero_init(*, inplace: bool = False) -> Callable[[Array], Array]:
    return partial(zero_init, inplace=inplace)


def uniform_init(
        ops: Ops, shape: Shape, *, lo: float = -0.1, hi: float = 0.1
) -> Array:
    values = ops.xp.random.uniform(lo, hi, shape)
    return values.astype(data.dtype)


@registry.initializers("uniform_init.v0")
def configure_uniform_init(
    *, lo: float = -0.1, hi: float = 0.1, inplace: bool = False
) -> Callable[[Array], Array]:
    return partial(uniform_init, lo=lo, hi=hi, inplace=inplace)


def normal_init(ops: Ops, shape: Shape, *, fan_in: int = -1, inplace: bool = False) -> Array:
    if fan_in == -1:
        fan_in = shape[1]
    scale = ops.xp.sqrt(1.0 / fan_in)
    size = int(ops.xp.prod(data.shape))
    inits = ops.xp.random.normal(scale=scale, size=size).astype("float32")
    inits = inits.reshape(data.shape)
    return inits


@registry.initializers("normal_init.v0")
def configure_normal_init(
    *, fan_in: int = -1, inplace: bool = False
) -> Callable[[Array], Array]:
    return partial(normal_init, fan_in=fan_in, inplace=inplace)


__all__ = ["normal_init", "uniform_init", "xavier_uniform_init", "zero_init"]
