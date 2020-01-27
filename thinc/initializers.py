from typing import Callable
import numpy

from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial

# TODO: Harmonize naming with Keras, and fill in missing entries
# https://keras.io/initializers/ We should also have He normal/uniform
# and probably lecun normal/uniform.

# Initialize via numpy, before copying to ops. This makes it easier to work with
# the different backends, because the backend won't affect the randomization.
# It's especially helpful for JAX, which has a pretty intrincate PRNG scheme I
# haven't figured out yet.


def glorot_uniform_init(ops: Ops, shape: Shape) -> FloatsXd:
    scale = numpy.sqrt(6.0 / (shape[0] + shape[1]))
    return ops.asarray_f(numpy.random.uniform(-scale, scale, shape))


@registry.initializers("glorot_uniform_init.v1")
def configure_glorot_uniform_init() -> Callable[[Shape], FloatsXd]:
    return partial(glorot_uniform_init)


def zero_init(ops: Ops, shape: Shape) -> FloatsXd:
    return ops.alloc(shape)


@registry.initializers("zero_init.v1")
def configure_zero_init() -> Callable[[FloatsXd], FloatsXd]:
    return partial(zero_init)


def uniform_init(
    ops: Ops, shape: Shape, *, lo: float = -0.1, hi: float = 0.1
) -> FloatsXd:
    values = numpy.random.uniform(lo, hi, shape)
    return ops.asarray_f(values.astype("float32"))


@registry.initializers("uniform_init.v1")
def configure_uniform_init(
    *, lo: float = -0.1, hi: float = 0.1
) -> Callable[[FloatsXd], FloatsXd]:
    return partial(uniform_init, lo=lo, hi=hi)


def normal_init(ops: Ops, shape: Shape, *, fan_in: int = -1) -> FloatsXd:
    if fan_in == -1:
        fan_in = shape[1]
    scale = ops.xp.sqrt(1.0 / fan_in)
    size = int(ops.xp.prod(ops.xp.asarray(shape)))
    inits = numpy.random.normal(scale=scale, size=size).astype("float32")
    inits = ops.reshape_f(inits, shape)
    return ops.asarray_f(inits)


@registry.initializers("normal_init.v1")
def configure_normal_init(*, fan_in: int = -1) -> Callable[[FloatsXd], FloatsXd]:
    return partial(normal_init, fan_in=fan_in)


__all__ = ["normal_init", "uniform_init", "glorot_uniform_init", "zero_init"]
