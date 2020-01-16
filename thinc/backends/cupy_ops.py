try:
    import cupy
    import cupy.cuda
    from cupy.cuda.compiler import compile_with_cache  # noqa: F401
    has_cupy = True

    # We no longer have to set up the memory pool, fortunately.
except ImportError:
    cupy = None
    has_cupy = False


from ..types import Array2d, Array3d
from .ops import Ops
from .numpy_ops import NumpyOps
from . import _custom_kernels
from ..util import get_array_module, copy_array


class CupyOps(Ops):
    device = "gpu"
    xp = cupy

    def gemm(self, x, y, out=None, trans1=False, trans2=False):
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        if out is None:
            return self.xp.dot(x, y)
        else:
            self.xp.dot(x, y, out=out)
            return out

    def asarray(self, data, dtype=None):
        # This is sort of frustrating, but we can't easily otherwise pass
        # forward "unset".
        dtype = {"dtype": dtype} if dtype is not None else {}
        if isinstance(data, cupy.ndarray):
            return self.xp.asarray(data, dtype=dtype)
        elif hasattr(data, "data_ptr"):
            # Handles PyTorch Tensors
            pointer = cupy.cuda.MemoryPointer(data.data_ptr())
            shape = data.stride()
            array = self.xp.ndarray(shape, memptr=pointer, **dtype)
            return array
        else:
            return self.xp.array(data, **dtype)

    def maxout(self, X):
        return _custom_kernels.maxout(X)

    def backprop_maxout(self, dY, which, P):
        return _custom_kernels.backprop_maxout(dY, which, P)

    def relu(self, X, inplace=False):
        if not inplace:
            return X * (X > 0)
        else:
            X *= X > 0
            return X

    def backprop_relu(self, dY, Y, inplace=False):
        if not inplace:
            return dY * (Y > 0)
        dY *= Y > 0
        return dY

    def mish(self, X, threshold=20.0, out=None):
        return _custom_kernels.mish(X, threshold=threshold, out=out)

    def backprop_mish(self, dY, X, threshold=20.0, out=None):
        return _custom_kernels.backprop_mish(dY, X, threshold=threshold, out=out)

    def clip_gradient(self, gradient, threshold):
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm

    def seq2col(self, seq, nW):
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1)) sequence.
        The new sequence is constructed by concatenating nW preceding and succeeding
        vectors onto each column in the sequence, to extract a window of features.
        """
        return _custom_kernels.seq2col(seq, nW)

    def backprop_seq2col(self, dY, nW):
        return _custom_kernels.backprop_seq2col(dY, nW)

    def mean_pool(self, X, lengths):
        return _custom_kernels.mean_pool(X, lengths)

    def backprop_mean_pool(self, d_means, lengths):
        return _custom_kernels.backprop_mean_pool(d_means, lengths)

    def max_pool(self, X, lengths):
        return _custom_kernels.max_pool(X, lengths)

    def backprop_max_pool(self, d_maxes, which, lengths):
        return _custom_kernels.backprop_max_pool(d_maxes, which, lengths)

    def sum_pool(self, X, lengths):
        return _custom_kernels.sum_pool(X, lengths)

    def backprop_sum_pool(self, d_sums, lengths):
        return _custom_kernels.backprop_sum_pool(d_sums, lengths)

    def hash(self, ids, seed):
        return _custom_kernels.hash(ids, seed)

    def scatter_add(self, out, ids, inputs):
        self.xp.scatter_add(out, ids, inputs)

    def adam(
        self, weights, gradient, mom1, mom2, beta1, beta2, eps, learn_rate, mod_rate=1.0
    ):
        cupy.ElementwiseKernel(
            "T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps",
            "T param, T m, T v",
            """m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (sqrt(v) + eps);""",
            "adam",
        )(gradient, learn_rate, 1 - beta1, 1 - beta2, eps, weights, mom1, mom2)
        gradient.fill(0)
        return weights, gradient, mom1, mom2

    def position_encode(self, N, D, period=10000, out=None):
        positions = NumpyOps().position_encode(N, D, period=period, out=out)
        return self.asarray(positions)

    def backprop_lstm(
        self,
        d_cells: Array2d,
        d_prev: Array2d,
        d_gates: Array3d,
        d_output: Array2d,
        gates: Array3d,
        cells: Array2d,
        prev: Array2d,
    ) -> None:
        (hf, hi, ho, hc) = (0, 1, 2, 3)
        cells_tanh = self.xp.tanh(cells)
        # Gradient for ho and c in h = sigmoid(ho) * tanh(c)
        d_gates[ho] = cells_tanh
        d_gates[ho] *= d_output * self.dsigmoid(gates[ho])
        d_prevcells = gates[ho] * d_output * self.dtanh(cells_tanh)
        d_prevcells += d_cells  # Carry gradient from timestep
        # Gradient for hf, hi, hc, prev[i]
        # in c = sigmoid(hf) * prev[i] + sigmoid(hi) * tanh(hc)
        d_gates[hf] = self.dsigmoid(gates[hf]) * d_prevcells * prev
        d_gates[hi] = self.dsigmoid(gates[1]) * d_prevcells * gates[hc]
        d_gates[hc] = self.dtanh(gates[hc]) * d_prevcells * gates[hi]
        d_prev[:] = d_prevcells * gates[hf]
        copy_array(d_cells, d_prevcells)
