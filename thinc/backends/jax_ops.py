from .ops import Ops

try:
    import jax
    import jax.random
except ImportError:
    pass


class JaxRandom:
    """Perform randomization functions for Jax."""
    def shuffle(self, array):
        key = jax.random.PRNGKey(0)
        return jax.random.shuffle(key, array)

    def uniform(self, minval, maxval, shape):
        key = jax.random.PRNGKey(0)
        return jax.random.uniform(key, minval=0.0, maxval=1.0, shape=shape, dtype="f")

    def normal(self, scale, size):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape=(size,)).astype("float32")


class JaxOps(Ops):
    xp = jax.numpy


JaxOps.xp.random = JaxRandom()
