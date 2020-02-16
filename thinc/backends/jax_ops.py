from typing import Sequence, Optional, List, Tuple, Callable, cast, TypeVar, Union
from typing import overload
import numpy

from .ops import Ops
from ..types import Floats1d, Floats2d, Floats3d, Ints1d, Ints2d, Ints3d
from ..types import ArrayXd, DTypes, Array3d, DeviceTypes, Padded, List2d, _Floats


try:  # pragma: no cover
    import jax
    import jax.ops
    import jax.random
    import jax.tree_util
    from jax.ops import index_update, index

    has_jax = True
except ImportError:  # pragma: no cover
    has_jax = False


ArrayT = TypeVar("ArrayT", bound=ArrayXd)
FloatsT = TypeVar("FloatsT", bound=_Floats)
_W = TypeVar("_W")
Wrapper = Callable[[_W], _W]


class JaxOps(Ops):
    name = "jax"
    xp = jax.numpy if has_jax else None

    def __init__(
        self, device_type: DeviceTypes = "gpu", device_id: int = 0, **kwargs
    ) -> None:
        self.device_type = device_type
        self.device_id = device_id

    def as_contig(self, data: ArrayT, dtype: Optional[DTypes] = None) -> ArrayT:
        arr = data if dtype is None else data.astype(dtype)
        return cast(ArrayT, arr)

    def to_numpy(self, data):  # pragma: no cover
        if isinstance(data, numpy.ndarray):
            return data
        else:
            return jax.device_get(data)

    def seq2col(self, seq: Floats2d, nW: int) -> Floats2d:
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1))
        sequence. The new sequence is constructed by concatenating nW preceding
        and succeeding vectors onto each column in the sequence, to extract a
        window of features.
        """
        if nW == 1:
            return seq2col_one(seq)
        else:  # pragma: no cover
            raise ValueError("Currently only nW=1 supported.")

    def backprop_seq2col(self, dY: Floats2d, nW: int) -> Floats2d:
        if nW == 1:
            return backprop_seq2col_one(dY)
        else:  # pragma: no cover
            raise ValueError("Currently only nW=1 supported.")

    def gemm(
        self,
        x: Floats2d,
        y: Floats2d,
        out: Optional[Floats2d] = None,
        trans1: bool = False,
        trans2: bool = False,
    ) -> Floats2d:
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        return self.xp.dot(x, y)

    def affine(self, X: Floats2d, W: Floats2d, b: Floats1d) -> Floats2d:
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

    def unflatten(self, X: Floats2d, lengths: Ints1d, pad: int = 0) -> List[Floats2d]:
        if not len(lengths):
            return []
        elif not X.size:
            empty_shape = (0,) + tuple(X.shape[1:])
            return [self.alloc(empty_shape) for _ in lengths]
        elif pad == 0:
            return unflatten_no_padding(X, self.asarray(lengths))
        else:
            return unflatten_with_padding(X, self.asarray(lengths), pad)

    def maxout(self, X):
        return maxout(X)

    def backprop_maxout(self, dY, which, P):
        return backprop_maxout(dY, which, P)

    def mish(self, X: Floats2d, threshold: float = 20.0) -> Floats2d:
        return mish(X, threshold)

    def backprop_mish(
        self,
        dY: Floats2d,
        X: Floats2d,
        threshold: float = 20.0,
        out: Optional[Floats2d] = None,
    ) -> Floats2d:
        return backprop_mish(dY, X, threshold)

    def relu(self, X: Floats2d, inplace: bool = False) -> Floats2d:
        return relu(X)

    def backprop_relu(
        self, dY: Floats2d, Y: Floats2d, inplace: bool = False
    ) -> Floats2d:
        return backprop_relu(dY, Y)

    def update_averages(
        self, ema: FloatsT, weights: FloatsT, t: int, max_decay: float = 0.9999
    ) -> None:
        decay = (1.0 + t) / (10.0 + t)
        if decay > max_decay:
            decay = max_decay
        return update_averages(ema, weights, decay)

    def adam(
        self,
        weights: Floats1d,
        gradient: Floats1d,
        mom1: Floats1d,
        mom2: Floats1d,
        beta1: float,
        beta2: float,
        eps: float,
        learn_rate: float,
        mod_rate: float = 1.0,
    ) -> Tuple[Floats1d, Floats1d, Floats1d, Floats1d]:
        return adam(
            weights, gradient, mom1, mom2, beta1, beta2, eps, learn_rate * mod_rate
        )

    def clip_gradient(self, gradient: FloatsT, threshold: float) -> FloatsT:
        xp = self.xp
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient = gradient * (threshold / grad_norm)
        return gradient

    def logloss(self, y_true: FloatsT, y_pred: FloatsT):
        return logloss

    def reduce_sum(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        return reduce_sum(X, lengths)

    def reduce_mean(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        return reduce_mean(X, lengths)

    def reduce_max(self, X: Floats2d, lengths: Ints1d) -> Tuple[Floats2d, Ints2d]:
        return reduce_max(X, lengths)

    def backprop_reduce_sum(self, d_sums: Floats2d, lengths: Ints1d) -> Floats2d:
        return backprop_reduce_sum(d_sums, lengths)

    def backprop_reduce_mean(self, d_means: Floats2d, lengths: Ints1d) -> Floats2d:
        return backprop_reduce_mean(d_means, lengths)

    def backprop_reduce_max(
        self, d_maxes: Floats2d, which: Ints2d, lengths: Ints1d
    ) -> Floats2d:
        return backprop_reduce_max(d_maxes, which, lengths)

    @overload
    def pad(self, seqs: List[Ints2d], round_to=1) -> Ints3d:
        ...

    @overload  # noqa: F811
    def pad(self, seqs: List[Floats2d], round_to=1) -> Floats3d:
        ...

    def pad(  # noqa: F811
        self, seqs: Union[List[Ints2d], List[Floats2d]], round_to=1
    ) -> Array3d:
        if not seqs:
            raise ValueError("Cannot pad empty sequence")
        if len(set(seq.ndim for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences with different ndims")
        if len(set(seq.dtype for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences with different dtypes")
        if len(set(seq.shape[1:] for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences that differ on other dimensions")
        # Find the maximum dimension along each axis. That's what we'll pad to.
        length = max(len(seq) for seq in seqs)
        # Round the length
        length = (length + (round_to - 1)) // round_to * round_to
        final_shape = (len(seqs), length) + seqs[0].shape[1:]
        output: Array3d = self.alloc(final_shape, dtype=seqs[0].dtype)
        for i, arr in enumerate(seqs):
            output[i, : arr.shape[0]] = arr  # type: ignore
        return output

    def list2padded(self, seqs: List[Floats2d]) -> Padded:
        """Pack a sequence of 2d arrays into a Padded datatype."""
        # I don't know why this is so slow, but it's *terrible*. Try going
        # via numpy?
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        numpy_seqs = [numpy_ops.asarray(seq) for seq in seqs]
        numpy_padded = numpy_ops.list2padded(numpy_seqs)
        return Padded(
            numpy_padded.data,
            self.asarray1i(numpy_padded.size_at_t),
            self.asarray1i(numpy_padded.lengths),
            self.asarray1i(numpy_padded.indices),
        )

    def padded2list(self, padded: Padded) -> List2d:
        indices = padded.indices
        data = padded.data
        lengths = padded.lengths
        unpadded = [None] * len(lengths)
        data = self.as_contig(data.transpose((1, 0, 2)))
        for i in range(data.shape[0]):
            index_update(unpadded, index[indices[i]], data[i, : lengths[i]])
        return cast(List2d, unpadded)

    def sigmoid(self, X: FloatsT, *, inplace: bool = False) -> FloatsT:
        return sigmoid(X)

    def dsigmoid(self, Y: FloatsT, *, inplace: bool = False) -> FloatsT:
        return Y * (1.0 - Y)

    def dtanh(self, Y: FloatsT, *, inplace: bool = False) -> FloatsT:
        if inplace:
            Y **= 2
            Y *= -1.0
            Y += 1.0
            return Y
        else:
            return 1 - Y ** 2

    def softmax(self, x: FloatsT, *, inplace: bool = False, axis: int = -1) -> FloatsT:
        maxes = self.xp.max(x, axis=axis, keepdims=True)
        shifted = x - maxes
        new_x = self.xp.exp(shifted)
        new_x /= new_x.sum(axis=axis, keepdims=True)
        return new_x

    def softmax_sequences(
        self, Xs: Floats2d, lengths: Ints1d, *, inplace: bool = False, axis: int = -1
    ) -> Floats2d:
        if Xs.ndim >= 3:
            err = f"Softmax currently only supports 2d. Got: {Xs.ndim}"
            raise NotImplementedError(err)
        # This loses almost no fidelity, and helps the numerical stability.
        Xs = self.xp.clip(Xs, -20.0, 20.0)
        new_x = self.xp.exp(Xs)
        summed = self.backprop_reduce_sum(self.reduce_sum(new_x, lengths), lengths)
        new_x /= summed
        return new_x

    def backprop_softmax(self, Y: FloatsT, dY: FloatsT, *, axis: int = -1) -> FloatsT:
        dX = Y * dY
        dX -= Y * dX.sum(axis=axis, keepdims=True)
        return dX

    def backprop_softmax_sequences(
        self, dY: Floats2d, Y: Floats2d, lengths: Ints1d
    ) -> Floats2d:
        dX = Y * dY
        sum_dX = self.backprop_reduce_sum(self.reduce_sum(dX, lengths), lengths)
        dX -= Y * sum_dX
        return dX

    def insert_into(self, shape, Xs):
        output = self.alloc(shape, dtype=Xs[0].dtype)
        for i, x in enumerate(Xs):
            output = index_update(output, index[i, : x.shape[0]], x)
        return output


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


def jax_jit(*static_args: int) -> Wrapper:
    """Apply jax.jit to the decorated function, if Jax is installed. Otherwise,
    do nothing. The decorator takes a variable-length sequence of positional
    arguments, which are passed as a tuple to jax.jit as the 'static_argnums'
    keyword argument.
    """

    def wrapper(func: Callable) -> Callable:
        return jax.jit(func, static_argnums=static_args) if has_jax else func

    return wrapper


@jax_jit()
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


@jax_jit()
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


@jax_jit()
def affine(X, W, b):
    return X @ W.T + b


@jax_jit()
def relu(X):
    return X * (X > 0)


@jax_jit()
def backprop_relu(delta, signal_out):
    return delta * (signal_out > 0)


@jax_jit(1)
def flatten_with_padding(X, pad):
    xp = jax.numpy
    padded = []
    for x in X:
        padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
        padded.append(x)
    padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
    return xp.concatenate(padded)


def unflatten_no_padding(X, lengths):
    # Couldn't get the JIT version right here yet.
    start = 0
    unflat = []
    for length in lengths:
        unflat.append(X[start : start + length])
        start += length
    return unflat


def unflatten_with_padding(X, lengths, pad):
    # Couldn't get the JIT version right here yet.
    unflat = []
    for length in lengths:
        X = X[pad:]
        unflat.append(X[:length])
        X = X[length:]
    X = X[pad:]
    return unflat


@jax_jit()
def maxout(X):
    which = X.argmax(axis=-1)
    return X.max(axis=-1), which


@jax_jit(2)
def backprop_maxout(dY, which, P):
    dX = jax.numpy.zeros((dY.shape[0], dY.shape[1], P), dtype="float32")
    for b in range(dY.shape[0]):
        for o in range(dY.shape[1]):
            dX = index_update(dX, index[b, o, which[b, o]], dY[b, o])
    return dX


@jax_jit()
def adam(
    weights: Floats1d,
    gradient: Floats1d,
    mom1: Floats1d,
    mom2: Floats1d,
    beta1: float,
    beta2: float,
    eps: float,
    learn_rate: float,
) -> Tuple[Floats1d, Floats1d, Floats1d, Floats1d]:
    mom1 *= beta1
    mom2 *= beta2
    mom1 += gradient * (1.0 - beta1)
    mom2 += gradient * gradient * (1.0 - beta2)
    # Here we assume learn rate is calculated by the caller.
    # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
    weights -= learn_rate * mom1 / (1.0 + eps)
    return weights, gradient, mom1, mom2


@jax_jit()
def update_averages(ema, weights, decay):
    return ema - (1 - decay) * (ema - weights)


@jax_jit()
def logloss(y_true: ArrayXd, y_pred: ArrayXd):
    log_yp = jax.numpy.log(y_pred + 1e-8)
    loss = (y_true * log_yp) + (1 - y_true) * jax.numpy.log((1 - y_pred) + 1e-8)
    return -loss


@jax_jit()
def reduce_sum(X: Floats2d, lengths: Ints1d) -> Floats2d:
    Y = jax.numpy.zeros((lengths.shape[0], X.shape[1]), dtype="f")
    start = 0
    for i, length in enumerate(lengths):
        Y = jax.ops.index_update(
            Y, jax.ops.index[i], X[start : start + length].sum(axis=0)
        )
        start += length
    return Y


@jax_jit()
def reduce_mean(X: Floats2d, lengths: Ints1d) -> Floats2d:
    Y = jax.numpy.zeros((lengths.shape[0], X.shape[1]), dtype="f")
    start = 0
    for i, length in enumerate(lengths):
        Y = jax.ops.index_update(
            Y, jax.ops.index[i], X[start : start + length].mean(axis=0)
        )
        start += length
    return Y


@jax_jit()
def reduce_max(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
    Y = jax.numpy.zeros((lengths.shape[0], X.shape[1]), dtype="f")
    start = 0
    for i, length in enumerate(lengths):
        Y = jax.ops.index_update(
            Y, jax.ops.index[i], X[start : start + length].max(axis=0)
        )
        start += length
    return Y


@jax_jit()
def backprop_reduce_sum(self, d_sums: Floats2d, lengths: Ints1d) -> Floats2d:
    dX = self.alloc2f(lengths.sum(), d_sums.shape[1])
    start = 0
    for i, length in enumerate(lengths):
        dX[start : start + length] = d_sums[i]
        start += length
    return dX


@jax_jit()
def backprop_reduce_mean(self, d_means: Floats2d, lengths: Ints1d) -> Floats2d:
    dX = self.alloc2f(lengths.sum(), d_means.shape[1])
    start = 0
    for i, length in enumerate(lengths):
        dX[start : start + length] = d_means[i] / length
        start += length
    return dX


@jax_jit()
def backprop_reduce_max(d_maxes: Floats2d, which: Ints2d, lengths: Ints1d) -> Floats2d:
    dX = numpy.jax.zeros((lengths.sum(), d_maxes.shape[1]))
    start = 0
    for i, length in enumerate(lengths):
        dX = index_update(dX, index[start : start + length, which[i]], d_maxes[i])
        start += length
    return dX


@jax_jit(1)
def mish(X: Floats2d, threshold: float = 20.0) -> Floats2d:
    Y = X * jax.numpy.tanh(jax.numpy.log(1.0 + jax.numpy.exp(X)))
    return jax.numpy.where(X >= threshold, X, Y)


@jax_jit(2)
def backprop_mish(X: Floats2d, dY: Floats2d, threshold: float = 20.0) -> Floats2d:
    xp = jax.numpy
    exp_x = xp.exp(X)
    exp_2x = xp.exp(2 * X)
    exp_3x = xp.exp(3 * X)
    omega = (4.0 * (X + 1)) + (4 * exp_2x) + exp_3x + exp_x * (4.0 * X + 6)
    delta = 2.0 * exp_x + exp_2x + 2.0
    dX = dY * ((exp_x * omega) / (delta * delta))
    # Gradient when above threshold will ignore softplus.
    return jax.numpy.where(X >= threshold, dY, dX)


@jax_jit()
def sigmoid(X: ArrayT) -> ArrayT:
    return 1.0 / (1.0 + jax.numpy.exp(-X))


@jax_jit()
def dsigmoid(Y: ArrayT) -> ArrayT:
    return Y * (1.0 - Y)


@jax_jit()
def dtanh(Y: ArrayT) -> ArrayT:
    return 1 - Y ** 2


@jax_jit(1)
def softmax(X: ArrayT, axis: int) -> ArrayT:
    xp = jax.numpy
    maxes = xp.max(X, axis=axis, keepdims=True)
    shifted = X - maxes
    new_x = xp.exp(shifted)
    new_x /= new_x.sum(axis=axis, keepdims=True)
    return new_x


@jax_jit(2)
def softmax_sequences(Xs: Floats2d, lengths: Ints1d, axis: int) -> Floats2d:
    xp = jax.numpy
    # This loses almost no fidelity, and helps the numerical stability.
    Xs = xp.clip(Xs, -20.0, 20.0)
    new_x = xp.exp(Xs)
    summed = backprop_reduce_sum(reduce_sum(new_x, lengths), lengths)
    new_x /= summed
    return new_x


@jax_jit(2)
def backprop_softmax(Y: ArrayXd, dY: ArrayXd, axis: int) -> ArrayXd:
    dX = Y * dY
    dX -= Y * dX.sum(axis=axis, keepdims=True)
    return dX


@jax_jit(2)
def backprop_softmax_sequences(dY: Floats2d, Y: Floats2d, lengths: Ints1d) -> Floats2d:
    dX = Y * dY
    sum_dX = backprop_reduce_sum(reduce_sum(dX, lengths), lengths)
    dX -= Y * sum_dX
    return dX


"""
LSTM Notation (kind of involved, but made it a lot easier to write)

X: Inputs
Y: Outputs (aka hiddens)
C: Cells
G: Gates (Output of non-linearity, i.e. lstm_gates(X @ W.T)
A: Activations (X @ W.T, before non-linearity)

Imagine we have the input:
batch = [
    ["apple", "banana", "cantaloupe", "date", "elderberry"],
    ["aardvark", "bat", "capybara", "dingo", "elephant"]
]

The input variable X will have one vector per word, so X[0, 1] will be banana's
vector, X[0, 1, 0] will be a float, the first element of that vector.

We're computing an output variable Y of shape (nL, nB, nO), so that Y[0, 1] is
the output variable of banana.

A problem with variables for RNNs is keeping the timesteps straight. It's hard
to distinguish the current, previous, and next timesteps. To solve this problem,
we follow the convention that **we are at timestep 3**.

Additionally, the variables for Y and C are offset by one, as the 0th elements
have the initial hiddens and initial cells. So:

    t=3
    Xt3: The input vectors for 'dingo' and 'date', i.e. X[t]
    Yt3: The output vectors for 'dingo' and 'date', i.e. Y[t+1] (Y is offset.)
    Ct2: The cells calculated at 'c...', that are the input for 'd...'
    Ct3: The cells calculated at 'd...', that are the input for 'e...'
    At3: The activations at 'd...'
    Gt3: The gates at 'd...'
"""


def lstm_forward_training(
    params: Floats1d, c_init: Floats2d, h_init: Floats2d, X: Floats2d, lengths: Ints1d
) -> Tuple[Floats2d, Tuple]:
    # TODO: bidirectional
    xp = jax.numpy
    depth, nO = c_init.shape
    nI: int = X.shape[1]
    batch_size = lengths[0]
    # Preallocate these so we can pass them through for loop.
    G = cast(Floats3d, xp.zeros((depth, X.shape[0], nO * 4), dtype="f"))
    # The Y and C are shifted by 1 step
    offset = batch_size
    Y = cast(Floats3d, xp.zeros((depth, X.shape[0] + offset, nO), dtype="f"))
    C = cast(Floats3d, xp.zeros((depth, X.shape[0] + offset, nO), dtype="f"))
    # The inits are shaped (n_layers, nO). We add the internal dimension to make
    # them set correctly.
    Y[:, :batch_size, :] = cast(Floats3d, h_init.reshape((depth, 1, nO)))
    C[:, :batch_size, :] = cast(Floats3d, c_init.reshape((depth, 1, nO)))
    params_i = 0
    seq_i = 0
    Xt3: Floats2d
    Yt3: Floats2d
    Ct3: Floats2d
    Yt2: Floats2d
    Ct2: Floats2d
    orig_X = X
    for i in range(depth):
        layer_params, params_i = _split_weights(params, i, nO, nI, params_i)
        Yt3 = Y[i, :batch_size, :]
        Ct3 = C[i, :batch_size, :]
        for t, batch_size in enumerate(lengths):
            # Prepare the inputs
            Xt3 = X[seq_i : seq_i + batch_size]
            Yt2 = Yt3[:batch_size]
            Ct2 = Ct3[:batch_size]
            # Now do the actual calculation
            At3 = lstm_weights_forward(layer_params, Xt3, Yt2)
            Yt3, Ct3, Gt3 = lstm_gates_forward(At3, Ct2)
            # Store the outputs
            G[i, seq_i : seq_i + batch_size] = Gt3
            Y[i, seq_i + offset : seq_i + offset + batch_size] = Yt3
            C[i, seq_i + offset : seq_i + offset + batch_size] = Ct3
            seq_i += batch_size
        X = Y[i, batch_size:]
    return Y[-1, offset:], (Y, G, C, orig_X)


def backprop_lstm(dY: Floats2d, lengths: Ints1d, params: Floats1d, fwd_state: Tuple):
    xp = jax.numpy
    Y, G, C, X = fwd_state
    depth, N, nO = C.shape
    nI = X.shape[1]
    batch_size = lengths[0]
    dX = cast(Floats2d, xp.zeros(X.shape, dtype=X.dtype))
    # We don't need to store all the cells for all the layers.
    dC = cast(Floats2d, xp.zeros((N, nO), dtype=C.dtype))
    # We just need these as gradients for the initial states.
    dC0 = cast(Floats2d, xp.zeros((depth, nO), dtype="f"))
    dH0 = cast(Floats2d, xp.zeros((depth, nO), dtype="f"))

    d_params = cast(Floats1d, xp.zeros(params.shape, dtype=params.dtype))
    # Collect the params and slices. It makes it a bit easier to get the indexing
    # right, when we're iterating backwards.
    params_i = 0
    all_layer_params = []
    for i in range(depth):
        layer_params, params_i = _split_weights(params, i, nO, nI, params_i)
        all_layer_params.append((layer_params, params_i))
    # Similarly, we want to compute all our indices first, and do the reversal.
    offset = lengths[0]
    indices = [slice(0, offset)]
    seq_i = offset
    for batch_size in lengths:
        indices.append(slice(seq_i, seq_i + batch_size))
        seq_i += batch_size
    Xs = [X] + [Y[i, offset:] for i in range(1, depth)]
    # Okay, now do the actual looping
    for i in reversed(range(depth)):
        for t in range(len(indices) - 1, 0, -1):
            # Y and C are offset, but dY isn't
            dAt3, dCt2 = backprop_lstm_gates(
                dY[indices[t - 1]],
                dC[indices[t]],
                G[i, indices[t - 1]],
                C[i, indices[t]],
                C[i, indices[t - 1]],
            )
            dXt3, dYt2, d_params = backprop_lstm_weights(
                dAt3,
                d_params,
                Xs[i][indices[t - 1]],
                Y[i, indices[t]],
                all_layer_params[i],
            )
            # Store the outputs
            dC[indices[t - 1]] = dCt2
            dX[indices[t - 1]] = dXt3
            if t >= 2:
                dY[indices[t - 2]] += dYt2
        dH0[i] = dYt2.sum(axis=0)
        dC0[i] = dCt2.sum(axis=0)
        dY = dX
    assert dX.shape[1] == X.shape[-1]
    return dX, d_params, dH0, dC0


def _split_weights(params: Floats1d, i: int, nO: int, nI: int, params_i: int):
    Wx_size = 4 * nO * nI
    bx_size = 4 * nO
    Wh_size = 4 * nO * nO
    bh_size = 4 * nO
    Wx = params[params_i : params_i + Wx_size].reshape((4 * nO, nI))
    params_i += Wx_size
    bx = params[params_i : params_i + bx_size].reshape((4 * nO,))
    params_i += bx_size
    Wh = params[params_i : params_i + Wh_size].reshape((4 * nO, nO))
    params_i += Wh_size
    bh = params[params_i : params_i + bh_size].reshape((4 * nO,))
    params_i += bh_size
    return ((Wx, bx), (Wh, bh)), params_i


def lstm_weights_forward(params, Xt3: Floats2d, Yt2: Floats2d) -> Floats2d:
    (Wx, bx), (Wh, bh) = params
    At3 = Xt3 @ Wx.T
    At3 += bx
    At3 += Yt2 @ Wh.T
    At3 += bh
    return At3


def backprop_lstm_weights(
    dAt3: Floats2d,
    d_params: Floats1d,
    Xt3: Floats2d,
    Yt2: Floats2d,
    params: Tuple[Tuple, int],
) -> Tuple[Floats2d, Floats2d, Floats1d]:
    ((Wx, bx), (Wh, bh)), i = params
    db = dAt3.sum(axis=0)
    size = db.shape[0]
    i -= size
    d_params[i : i + size] += db  # Grad of bh
    size = dAt3.shape[1] * Yt2.shape[1]
    i -= size
    d_params[i : i + size] += (dAt3.T @ Yt2).ravel()  # Grad of Wh
    size = db.shape[0]
    i -= size
    d_params[i : i + size] += db  # Grad of bx
    size = dAt3.shape[1] * Xt3.shape[1]
    i -= size
    d_params[i : i + size] += (dAt3.T @ Xt3).ravel()  # Grad of Wx
    dXt3 = dAt3 @ Wx
    dYt2 = dAt3 @ Wh
    return dXt3, dYt2, d_params


def lstm_gates_forward(
    At3: Floats2d, Ct2: Floats2d
) -> Tuple[Floats2d, Floats2d, Floats2d]:
    xp = jax.numpy
    # hf, hi, ho, hc: Forget, input, output, cell gates.
    At3_hf, At3_hi, At3_ho, At3_hc = xp.split(At3, 4, axis=-1)
    # Number the steps here, to refer back for backward pass.
    # 1. Activations
    hf = sigmoid(At3_hf)  # 1a
    hi = sigmoid(At3_hi)  # 1b
    ho = sigmoid(At3_ho)  # 1c
    hc = xp.tanh(At3_hc)  # 1d

    Ct3 = hf * Ct2  # 2a
    Ct3 += hi * hc  # 2b
    tanhCt3 = xp.tanh(Ct3)  # 3a
    Yt3 = tanhCt3 * ho  # 3b
    # We don't need the gradient for this, it's just for backprop calculation.
    Gt3 = xp.concatenate((hf, hi, ho, hc), axis=-1)
    return Yt3, Ct3, Gt3


def backprop_lstm_gates(
    dYt3: Floats2d, dCt3: Floats2d, Gt3: Floats2d, Ct3: Floats2d, Ct2: Floats2d
) -> Tuple[Floats2d, Floats2d]:
    # See above for notation. Step numbering refers to forward_lstm_gates
    xp = jax.numpy
    hf, hi, ho, hc = xp.split(Gt3, 4, axis=-1)
    assert hf.shape[0] == hi.shape[0] == ho.shape[0] == hc.shape[0]
    assert hf.shape[0] == dYt3.shape[0] == dCt3.shape[0] == Ct3.shape[0] == Ct2.shape[0]
    tanhCt3 = xp.tanh(Ct3)
    # 3b: Yt3 = tanhCt3 * ho
    d_ho = dYt3 * tanhCt3
    d_tanhCt3 = dYt3 * ho
    # 3a: tanhCt3 = tanh(Ct3)
    dCt3 += d_tanhCt3 * dtanh(tanhCt3)
    # 2b: Ct3 += hi * hc
    d_hi = dCt3 * hc
    d_hc = dCt3 * hi
    # 2a: Ct3 = hf * Ct2
    d_hf = dCt3 * Ct2
    dCt2 = dCt3 * hf
    d_At3_hc = d_hc * dtanh(hc)  # 1d
    d_At3_ho = d_ho * dsigmoid(ho)  # 1c
    d_At3_hi = d_hi * dsigmoid(hi)  # 1b
    d_At3_hf = d_hf * dsigmoid(hf)  # 1a
    dAt3 = xp.concatenate((d_At3_hf, d_At3_hi, d_At3_ho, d_At3_hc), axis=-1)
    return dAt3, dCt2


if has_jax:
    JaxOps.xp.random = JaxRandom()
    JaxOps.xp.testing = numpy.testing
    jax.tree_util.register_pytree_node(
        JaxOps, lambda ops: ([], None), lambda info, values: JaxOps()
    )

__all__ = ["JaxOps", "has_jax", "jax_jit"]
