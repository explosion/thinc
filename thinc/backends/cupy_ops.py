import numpy

try:
    import cupy
    import cupyx
    import cupy.cuda
    from cupy.cuda.compiler import compile_with_cache  # noqa: F401

    has_cupy = True

    # We no longer have to set up the memory pool, fortunately.
except ImportError:
    cupy = None
    cupyx = None
    has_cupy = False

from .. import registry
from .ops import Ops
from .numpy_ops import NumpyOps
from . import _custom_kernels
from ..types import DeviceTypes


@registry.ops("CupyOps")
class CupyOps(Ops):
    name = "cupy"
    xp = cupy
    _xp2 = cupyx

    def __init__(
        self, device_type: DeviceTypes = "gpu", device_id: int = 0, **kwargs
    ) -> None:
        self.device_type = device_type
        self.device_id = device_id

    def to_numpy(self, data, *, byte_order=None):
        if not isinstance(data, numpy.ndarray):
            data = data.get()
        if byte_order:
            dtype = data.dtype.newbyteorder(byte_order)
            data = numpy.asarray(data, dtype=dtype)
        return data

    def gelu(self, X, inplace=False):
        if X.dtype == "float32":
            return _custom_kernels.gelu(X, inplace=inplace, threshold=6.0)
        else:
            return super().gelu(X, inplace=inplace)

    def backprop_gelu(self, dY, X, inplace=False):
        if X.dtype == "float32" and dY.dtype == "float32":
            return _custom_kernels.backprop_gelu(dY, X, inplace=inplace, threshold=6.0)
        else:
            return super().backprop_gelu(dY, X, inplace=inplace)

    def gemm(self, x, y, out=None, trans1=False, trans2=False):
        if isinstance(x, numpy.ndarray) or isinstance(y, numpy.ndarray):
            raise ValueError(
                "Encountered a numpy array when processing with cupy. "
                "Did you call model.ops.asarray on your data?"
            )
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
            return self.xp.asarray(data, **dtype)
        elif hasattr(data, "data_ptr"):
            # Handles PyTorch Tensors
            pointer = cupy.cuda.MemoryPointer(data.data_ptr())
            shape = data.stride()
            array = self.xp.ndarray(shape, memptr=pointer, **dtype)
            return array
        else:
            result = self.xp.array(data, **dtype)
            return result

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

    def clipped_linear(
        self,
        X,
        slope: float = 1.0,
        offset: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        inplace: bool = False,
    ):
        if X.dtype == "float32":
            return _custom_kernels.clipped_linear(
                X,
                inplace=inplace,
                slope=slope,
                offset=offset,
                min_val=min_val,
                max_val=max_val,
            )
        else:
            return super().clipped_linear(
                X,
                inplace=inplace,
                slope=slope,
                offset=offset,
                min_val=min_val,
                max_val=max_val,
            )

    def backprop_clipped_linear(
        self,
        dY,
        X,
        slope: float = 1.0,
        offset: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        inplace: bool = False,
    ):
        if X.dtype == "float32" and dY.dtype == "float32":
            return _custom_kernels.backprop_clipped_linear(
                dY=dY,
                X=X,
                slope=slope,
                offset=offset,
                min_val=min_val,
                max_val=max_val,
                inplace=inplace,
            )
        else:
            return super().backprop_clipped_linear(
                dY=dY,
                X=X,
                slope=slope,
                offset=offset,
                min_val=min_val,
                max_val=max_val,
                inplace=inplace,
            )

    def backprop_hard_swish(self, dY, X, inplace: bool = False):
        if X.dtype == "float32" and dY.dtype == "float32":
            return _custom_kernels.backprop_hard_swish(dY, X, inplace=inplace)
        else:
            return super().backprop_hard_swish(dY, X, inplace=inplace)

    def backprop_hard_swish_mobilenet(self, dY, X, inplace: bool = False):
        if X.dtype == "float32" and dY.dtype == "float32":
            return _custom_kernels.backprop_hard_swish_mobilenet(dY, X, inplace=inplace)
        else:
            return super().backprop_hard_swish_mobilenet(dY, X, inplace=inplace)

    def mish(self, X, threshold=20.0, inplace=False):
        if X.dtype == "float32" and not inplace:
            return _custom_kernels.mish(X, threshold=threshold, out=None)
        else:
            return super().mish(X, threshold, inplace)

    def backprop_mish(self, dY, X, threshold=20.0, inplace=False):
        if dY.dtype == "float32" and X.dtype == "float32" and not inplace:
            return _custom_kernels.backprop_mish(dY, X, threshold=threshold)
        else:
            return super().backprop_mish(dY, X, threshold, inplace)

    def swish(self, X, inplace=False):
        if X.dtype == "float32":
            return _custom_kernels.swish(X, inplace=inplace, threshold=17.0)
        else:
            return super().swish(X, inplace=inplace)

    def backprop_swish(self, dY, X, Y, inplace=False):
        if X.dtype == "float32" and dY.dtype == "float32":
            return _custom_kernels.backprop_swish(
                dY, X, Y, inplace=inplace, threshold=17.0
            )
        else:
            return super().backprop_swish(dY, X, Y, inplace=inplace)

    def clip_gradient(self, gradient, threshold):
        # We do not use CuPy's linalg.norm, since it uses scalar reductions
        # using one CUDA block. This is a lot slower than the cuBLAS
        # implementation.
        def frobenius_norm(X):
            X_vec = X.reshape(-1)
            return cupy.cublas.nrm2(X_vec)

        grad_norm = cupy.maximum(frobenius_norm(gradient), 1e-12)
        gradient *= cupy.minimum(threshold, grad_norm) / grad_norm
        return gradient

    def seq2col(self, seq, nW, *, lengths=None):
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1)) sequence.
        The new sequence is constructed by concatenating nW preceding and succeeding
        vectors onto each column in the sequence, to extract a window of features.
        """
        return _custom_kernels.seq2col(seq, nW, lengths=lengths)

    def backprop_seq2col(self, dY, nW, *, lengths=None):
        return _custom_kernels.backprop_seq2col(dY, nW, lengths=lengths)

    def reduce_mean(self, X, lengths):
        return _custom_kernels.reduce_mean(X, lengths)

    def backprop_reduce_mean(self, d_means, lengths):
        return _custom_kernels.backprop_reduce_mean(d_means, lengths)

    def reduce_max(self, X, lengths):
        return _custom_kernels.reduce_max(X, lengths)

    def backprop_reduce_max(self, d_maxes, which, lengths):
        return _custom_kernels.backprop_reduce_max(d_maxes, which, lengths)

    def reduce_sum(self, X, lengths):
        return _custom_kernels.reduce_sum(X, lengths)

    def backprop_reduce_sum(self, d_sums, lengths):
        return _custom_kernels.backprop_reduce_sum(d_sums, lengths)

    def hash(self, ids, seed):
        return _custom_kernels.hash(ids, seed)

    def scatter_add(self, table, indices, values):
        self._xp2.scatter_add(table, indices, values)

    def adam(
        self, weights, gradient, mom1, mom2, beta1, beta2, eps, learn_rate, mod_rate=1.0
    ):
        adam_kernel(
            gradient, learn_rate, 1 - beta1, 1 - beta2, eps, weights, mom1, mom2
        )
        gradient.fill(0)
        return weights, gradient, mom1, mom2

    def position_encode(self, N, D, period=10000, out=None):
        positions = NumpyOps().position_encode(N, D, period=period, out=out)
        return self.asarray(positions)


if cupy is not None:
    adam_kernel = cupy.ElementwiseKernel(
        "T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps",
        "T param, T m, T v",
        """m += one_minus_beta1 * (grad - m);
        v += one_minus_beta2 * (grad * grad - v);
        param -= lr * m / (sqrt(v) + eps);""",
        "adam",
    )
else:
    adam_kernel = None
