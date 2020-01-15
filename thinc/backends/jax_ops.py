from .ops import Ops
import numpy
from ..types import Array, Array2d, Array1d, ArrayT, DTypes, Array3d
from typing import Sequence, Optional, List

try:
    import jax
    import jax.ops
    import jax.random
    import jax.tree_util
    from jax.ops import index_update, index

    has_jax = True
except ImportError:
    has_jax = False


class JaxOps(Ops):
    xp = jax.numpy

    def seq2col(self, seq: ArrayT, nW: int) -> ArrayT:
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1))
        sequence. The new sequence is constructed by concatenating nW preceding
        and succeeding vectors onto each column in the sequence, to extract a
        window of features.
        """
        if nW == 1:
            return seq2col_one(seq)
        else:
            raise ValueError("Currently only nW=1 supported.")

    def backprop_seq2col(self, dY: ArrayT, nW: int) -> Array:
        if nW == 1:
            return backprop_seq2col_one(dY)
        else:
            raise ValueError("Currently only nW=1 supported.")

    def affine(self, X, W, b):
        return affine(X, W, b)

    def flatten(
        self,
        X: Sequence[ArrayT],
        dtype: Optional[DTypes] = None,
        pad: int = 0,
        ndim_if_empty: int = 2,
    ) -> ArrayT:
        if X is None or len(X) == 0:
            return self.alloc((0,) * ndim_if_empty, dtype=dtype or "f")
        X = [x for x in X if x.size != 0]
        if int(pad) >= 1:
            return flatten_with_padding(X, pad)
        else:
            result = self.xp.concatenate(X)

        result = self.xp.concatenate(X)
        if dtype is not None:
            result = self.xp.asarray(result, dtype=dtype)
        return result

    def unflatten(self, X: ArrayT, lengths: Array1d, pad: int = 0) -> List[ArrayT]:
        return unflatten(X, lengths, pad)

    def maxout(self, X):
        return maxout(X)

    def backprop_maxout(self, dY, which, P):
        return backprop_maxout(dY, which, P)

    def relu(self, X, inplace=False):
        if not inplace:
            return X * (X > 0)
        else:
            X *= X > 0
            return X

    def backprop_relu(self, delta_, signal_out, inplace=False):
        if not inplace:
            return delta_ * (signal_out > 0)
        delta_ *= signal_out > 0
        return delta_

    def mish(
        self, X: Array2d, threshold: float = 20.0, out: Optional[Array2d] = None
    ) -> Array2d:
        Xsoft = self.softplus(X, threshold=threshold, out=out)
        Y = self.xp.tanh(Xsoft, out=out)
        Y *= X
        return Y

    def backprop_mish(
        self,
        dY: Array2d,
        X: Array2d,
        threshold: float = 20.0,
        out: Optional[Array2d] = None,
    ):
        xp = self.xp
        indices = X < threshold
        Xsub = X[indices]
        dYsub = dY[indices]
        omega = 4.0 * (Xsub + 1.0)
        omega += 4.0 * xp.exp(2.0 * Xsub)
        omega += xp.exp(Xsub) * ((4.0 * Xsub) + 6.0)
        delta = 2.0 * xp.exp(Xsub)
        delta += xp.exp(2.0 * Xsub)
        delta += 2.0
        dXsub = dYsub * ((xp.exp(Xsub) * omega) / (delta ** 2))
        # Gradient when above threshold will ignore softplus.
        out = dY + dY * self.dtanh(X)
        out = index_update(out, index[indices], dXsub)
        return out

    def update_averages(
        self, ema: Array, weights: Array, t: int, max_decay: float = 0.9999
    ) -> None:
        decay = (1.0 + t) / (10.0 + t)
        if decay > max_decay:
            decay = max_decay
        ema -= (1 - decay) * (ema - weights)

    # TODO: types
    def adam(
        self,
        weights: Array1d,
        gradient: Array1d,
        mom1: Array1d,
        mom2: Array1d,
        beta1: float,
        beta2: float,
        eps: float,
        learn_rate: float,
        mod_rate: float = 1.0,
    ) -> None:
        mom1 *= beta1
        mom2 *= beta2
        mom1 += gradient * (1.0 - beta1)
        mom2 += gradient * gradient * (1.0 - beta2)
        # Here we assume learn rate is calculated by the caller.
        # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
        weights -= learn_rate * (mom1 / (mod_rate * self.xp.sqrt(mom2) + eps))
        gradient.fill(0)

    def clip_gradient(self, gradient: Array, threshold: float) -> Array:
        xp = self.xp
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm
        return gradient

    def logloss(self, y_true: Array, y_pred: Array):
        log_yp = self.xp.log(y_pred + 1e-8)
        loss = (y_true * log_yp) + (1 - y_true) * self.xp.log((1 - y_pred) + 1e-8)
        return -loss

    def sum_pool(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].sum(axis=0)
            start += length
        return Y

    def mean_pool(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].mean(axis=0)
            start += length
        return Y

    def max_pool(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].max(axis=0)
            start += length
        return Y

    def backprop_sum_pool(self, d_sums: Array2d, lengths: Array1d) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_sums.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_sums[i]
            start += length
        return dX

    def backprop_mean_pool(self, d_means: Array2d, lengths: Array1d) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_means.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_means[i] / length
            start += length
        return dX

    def backprop_max_pool(
        self, d_maxes: Array2d, which: Array2d, lengths: Array1d
    ) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_maxes.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length, which[i]] = d_maxes[i]
            start += length
        return dX


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


def jit_static_argnums(*nums):
    def wrapper(func):
        return jax.jit(func, static_argnums=nums)

    return wrapper


@jax.jit
def seq2col_one(seq):
    # This is a test implementation that only supports nW=1
    nW = 1
    B = seq.shape[0]
    I = seq.shape[1]
    cols: Array3d = jax.numpy.zeros((B, (nW * 2 + 1), I))
    # Copy left contexts. The last words aren't the left-context for anything.
    cols = index_update(cols, index[nW:, :nW], seq[:-nW].reshape((-1, nW, I)))
    cols = index_update(cols, index[:, nW], seq)
    cols = index_update(cols, index[:-nW, nW + 1 :], seq[nW:].reshape((-1, nW, I)))
    return cols.reshape((B, I * (2 * nW + 1)))


@jax.jit
def backprop_seq2col_one(dY):
    xp = jax.numpy
    nW = 1
    nF = nW * 2 + 1
    B = dY.shape[0]
    I = dY.shape[1] // nF
    dX = xp.zeros((B, I), dtype="f")
    dY = dY.reshape((B, nF, I))
    dX = index_update(dX, index[:-nW], dX[:-nW] + dY[nW:, :nW].reshape((-1, I)))
    dX += dY[:, nW]
    dX = index_update(dX, index[nW:], dX[nW:] + dY[:-nW, nW + 1 :].reshape((-1, I)))
    return dX


@jax.jit
def affine(X, W, b):
    return X @ W.T + b


@jit_static_argnums(1)
def flatten_with_padding(X, pad):
    xp = jax.numpy
    padded = []
    for x in X:
        padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
        padded.append(x)
    padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
    return xp.concatenate(padded)


@jit_static_argnums(1)
def unflatten(X, lengths, pad):
    unflat = []
    for length in lengths:
        if pad >= 1:
            X = X[pad:]
        unflat.append(X[:length])
        X = X[length:]
    if pad >= 1:
        X = X[pad:]
    return unflat


@jax.jit
def maxout(X):
    which = X.argmax(axis=-1)
    return X.max(axis=-1), which


@jit_static_argnums(2)
def backprop_maxout(dY, which, P):
    dX = jax.numpy.zeros((dY.shape[0], dY.shape[1], P), dtype="float32")
    for b in range(dY.shape[0]):
        for o in range(dY.shape[1]):
            dX[b, o, which[b, o]] = dY[b, o]
    return dX


JaxOps.xp.random = JaxRandom()
JaxOps.xp.testing = numpy.testing

if has_jax:
    jax.tree_util.register_pytree_node(
        JaxOps, lambda ops: ([], None), lambda info, values: JaxOps()
    )
