from typing import Optional, Tuple
import re
from pathlib import Path
from collections import defaultdict
from ..compat import cupy, has_cupy_gpu


PWD = Path(__file__).parent
KERNELS_SRC = (PWD / "_custom_kernels.cu").read_text(encoding="utf8")
KERNELS_LIST = [
    "backprop_clipped_linear<double>",
    "backprop_clipped_linear<float>",
    "backprop_gelu<double>",
    "backprop_gelu<float>",
    "backprop_hard_swish<double>",
    "backprop_hard_swish<float>",
    "backprop_hard_swish_mobilenet<double>",
    "backprop_hard_swish_mobilenet<float>",
    "backprop_maxout<double>",
    "backprop_maxout<float>",
    "backprop_mish<double>",
    "backprop_mish<float>",
    "backprop_reduce_max<double>",
    "backprop_reduce_max<float>",
    "backprop_reduce_mean<double>",
    "backprop_reduce_mean<float>",
    "backprop_reduce_sum<double>",
    "backprop_reduce_sum<float>",
    "backprop_seq2col<double>",
    "backprop_seq2col<float>",
    "backprop_swish<double>",
    "backprop_swish<float>",
    "clipped_linear<double>",
    "clipped_linear<float>",
    "gather_add<double>",
    "gather_add<float>",
    "gelu<double>",
    "gelu<float>",
    "maxout<double>",
    "maxout<float>",
    "mish<double>",
    "mish<float>",
    "reduce_max<double>",
    "reduce_max<float>",
    "reduce_sum<double>",
    "reduce_sum<float>",
    "seq2col<double>",
    "seq2col<float>",
    "swish<double>",
    "swish<float>",
]
KERNELS = (
    cupy.RawModule(
        code=KERNELS_SRC, options=("--std=c++11",), name_expressions=KERNELS_LIST
    )
    if has_cupy_gpu
    else None
)


def _get_kernel(name):
    """A small wrapper around KERNELS.get_function that verifies first that
    compiler kernels are available (cupy is installed)."""
    if KERNELS is None:
        return None
    else:
        return KERNELS.get_function(name)


def compile_mmh(src):
    if not has_cupy_gpu:
        return None
    return cupy.RawKernel(src, "hash_data")


MMH_SRC = (PWD / "_murmur3.cu").read_text(encoding="utf8")


clipped_linear_kernel_float = _get_kernel("clipped_linear<float>")
clipped_linear_kernel_double = _get_kernel("clipped_linear<double>")
gather_add_kernel_float = _get_kernel("gather_add<float>")
gather_add_kernel_double = _get_kernel("gather_add<double>")
gelu_kernel_float = _get_kernel("gelu<float>")
gelu_kernel_double = _get_kernel("gelu<double>")
hash_data_kernel = compile_mmh(MMH_SRC)
maxout_kernel_float = _get_kernel("maxout<float>")
maxout_kernel_double = _get_kernel("maxout<double>")
mish_kernel_float = _get_kernel("mish<float>")
mish_kernel_double = _get_kernel("mish<double>")
reduce_max_kernel_float = _get_kernel("reduce_max<float>")
reduce_max_kernel_double = _get_kernel("reduce_max<double>")
reduce_sum_kernel_float = _get_kernel("reduce_sum<float>")
reduce_sum_kernel_double = _get_kernel("reduce_sum<double>")
seq2col_kernel_float = _get_kernel("seq2col<float>")
seq2col_kernel_double = _get_kernel("seq2col<double>")
swish_kernel_float = _get_kernel("swish<float>")
swish_kernel_double = _get_kernel("swish<double>")

backprop_clipped_linear_kernel_double = _get_kernel("backprop_clipped_linear<double>")
backprop_clipped_linear_kernel_float = _get_kernel("backprop_clipped_linear<float>")
backprop_gelu_kernel_double = _get_kernel("backprop_gelu<double>")
backprop_gelu_kernel_float = _get_kernel("backprop_gelu<float>")
backprop_hard_swish_kernel_double = _get_kernel("backprop_hard_swish<double>")
backprop_hard_swish_kernel_float = _get_kernel("backprop_hard_swish<float>")
backprop_hard_swish_mobilenet_kernel_double = _get_kernel(
    "backprop_hard_swish_mobilenet<double>"
)
backprop_hard_swish_mobilenet_kernel_float = _get_kernel(
    "backprop_hard_swish_mobilenet<float>"
)
backprop_maxout_kernel_double = _get_kernel("backprop_maxout<double>")
backprop_maxout_kernel_float = _get_kernel("backprop_maxout<float>")
backprop_mish_kernel_double = _get_kernel("backprop_mish<double>")
backprop_mish_kernel_float = _get_kernel("backprop_mish<float>")
backprop_reduce_max_kernel_double = _get_kernel("backprop_reduce_max<double>")
backprop_reduce_max_kernel_float = _get_kernel("backprop_reduce_max<float>")
backprop_reduce_mean_kernel_double = _get_kernel("backprop_reduce_mean<double>")
backprop_reduce_mean_kernel_float = _get_kernel("backprop_reduce_mean<float>")
backprop_reduce_sum_kernel_double = _get_kernel("backprop_reduce_sum<double>")
backprop_reduce_sum_kernel_float = _get_kernel("backprop_reduce_sum<float>")
backprop_seq2col_kernel_double = _get_kernel("backprop_seq2col<double>")
backprop_seq2col_kernel_float = _get_kernel("backprop_seq2col<float>")
backprop_swish_kernel_double = _get_kernel("backprop_swish<double>")
backprop_swish_kernel_float = _get_kernel("backprop_swish<float>")


def _alloc(shape, dtype, *, zeros: bool = True):
    if zeros:
        return cupy.zeros(shape, dtype)
    else:
        return cupy.empty(shape, dtype)


def _alloc_like(array, zeros: bool = True):
    if zeros:
        return cupy.zeros_like(array)
    else:
        return cupy.empty_like(array)


def clipped_linear(
    X,
    *,
    inplace=False,
    slope=1.0,
    offset=0.0,
    min_val=0.0,
    max_val=1.0,
    threads_per_block=128,
    num_blocks=128,
):
    _is_float_array(X)

    out = X
    if not inplace:
        out = _alloc_like(X, zeros=False)
    if X.dtype == "float32":
        clipped_linear_kernel_float(
            (num_blocks,),
            (threads_per_block,),
            (out, X, slope, offset, min_val, max_val, X.size),
        )
    else:
        clipped_linear_kernel_double(
            (num_blocks,),
            (threads_per_block,),
            (out, X, slope, offset, min_val, max_val, X.size),
        )
    return out


def gather_add(table, indices, *, threads_per_block=128, num_blocks=128):
    if table.ndim != 2:
        raise ValueError(
            f"gather_add expects table with dimensionality 2, was: {table.ndim}"
        )
    if indices.ndim != 2:
        raise ValueError(
            f"gather_add expects indices with dimensionality 2, was: {indices.ndim}"
        )
    _is_float_array(table)
    indices = indices.astype("int32")
    _check_indices(indices, table.shape[0])

    B = indices.shape[0]
    K = indices.shape[1]
    T = table.shape[0]
    O = table.shape[1]

    out = _alloc((B, O), dtype=table.dtype, zeros=True)
    if table.dtype == "float32":
        gather_add_kernel_float(
            (num_blocks,), (threads_per_block,), (out, table, indices, T, O, B, K)
        )
    else:
        gather_add_kernel_double(
            (num_blocks,), (threads_per_block,), (out, table, indices, T, O, B, K)
        )
    return out


def gelu(X, *, inplace=False, threshold=6.0, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    out = X
    if not inplace:
        out = _alloc_like(X, zeros=False)
    if X.dtype == "float32":
        gelu_kernel_float(
            (num_blocks,), (threads_per_block,), (out, X, threshold, X.size)
        )
    else:
        gelu_kernel_double(
            (num_blocks,), (threads_per_block,), (out, X, threshold, X.size)
        )
    return out


def check_seq2col_lengths(lengths, B):
    if lengths is None:
        lengths = cupy.array([B], dtype="int32")
    else:
        _check_lengths(lengths, B)
    return lengths


def seq2col(seq, nW, *, lengths=None, threads_per_block=128, num_blocks=128):
    _is_float_array(seq)

    B = seq.shape[0]
    nF = nW * 2 + 1
    I = seq.shape[1]

    lengths = check_seq2col_lengths(lengths, B)
    nL = lengths.shape[0]

    out = _alloc((B, I * nF), dtype=seq.dtype, zeros=True)

    if seq.size != 0 and lengths.size != 0:
        if seq.dtype == "float32":
            seq2col_kernel_float(
                (num_blocks,), (threads_per_block,), (out, seq, lengths, nW, B, I, nL)
            )
        else:
            seq2col_kernel_double(
                (num_blocks,), (threads_per_block,), (out, seq, lengths, nW, B, I, nL)
            )

    return out


def maxout(X, *, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    B, I, P = X.shape

    out_shape = (B, I)
    best = _alloc(out_shape, dtype=X.dtype, zeros=False)
    which = _alloc(out_shape, dtype="i", zeros=False)

    if X.dtype == "float32":
        maxout_kernel_float(
            (num_blocks,), (threads_per_block,), (best, which, X, B, I, P)
        )
    else:
        maxout_kernel_double(
            (num_blocks,), (threads_per_block,), (best, which, X, B, I, P)
        )

    return best, which


def mish(X, *, inplace=False, threshold=5, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    out = X
    if not inplace:
        out = _alloc_like(X, zeros=False)

    if X.dtype == "float32":
        mish_kernel_float(
            (num_blocks,), (threads_per_block,), (out, X, threshold, X.size)
        )
    else:
        mish_kernel_double(
            (num_blocks,), (threads_per_block,), (out, X, threshold, X.size)
        )

    return out


def reduce_sum(X, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]

    _check_lengths(lengths, T)

    out = _alloc((B, O), dtype=X.dtype, zeros=True)

    if X.dtype == "float32":
        reduce_sum_kernel_float(
            (num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O)
        )
    else:
        reduce_sum_kernel_double(
            (num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O)
        )

    return out


def reduce_mean(X, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]

    _check_lengths(lengths, T)

    out = _alloc((B, O), dtype=X.dtype, zeros=True)

    if X.dtype == "float32":
        reduce_sum_kernel_float(
            (num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O)
        )
    else:
        reduce_sum_kernel_double(
            (num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O)
        )

    # Avoid divide by zero
    out /= lengths.reshape((-1, 1)) + 1e-10
    return out


def reduce_max(X, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]

    _check_lengths(lengths, T, min_length=1)

    out_shape = (B, O)
    maxes = _alloc(out_shape, dtype=X.dtype, zeros=False)
    which = _alloc(out_shape, dtype="i", zeros=False)

    if X.dtype == "float32":
        reduce_max_kernel_float(
            (num_blocks,), (threads_per_block,), (maxes, which, X, lengths, B, T, O)
        )
    else:
        reduce_max_kernel_double(
            (num_blocks,), (threads_per_block,), (maxes, which, X, lengths, B, T, O)
        )

    return maxes, which


def swish(X, *, inplace=False, threshold=17.0, threads_per_block=128, num_blocks=128):
    _is_float_array(X)

    out = X
    if not inplace:
        out = _alloc_like(X, zeros=False)
    if X.dtype == "float32":
        swish_kernel_float(
            (num_blocks,), (threads_per_block,), (out, X, threshold, X.size)
        )
    else:
        swish_kernel_double(
            (num_blocks,), (threads_per_block,), (out, X, threshold, X.size)
        )
    return out


def backprop_seq2col(dY, nW, *, lengths=None, threads_per_block=128, num_blocks=128):
    _is_float_array(dY)

    B = dY.shape[0]
    nF = nW * 2 + 1
    I = dY.shape[1] // nF

    lengths = check_seq2col_lengths(lengths, B)
    nL = lengths.shape[0]

    out = _alloc((B, I), dtype=dY.dtype, zeros=True)

    if dY.size != 0 and lengths.size != 0:
        if dY.dtype == "float32":
            backprop_seq2col_kernel_float(
                (num_blocks,), (threads_per_block,), (out, dY, lengths, nW, B, I, nL)
            )
        else:
            backprop_seq2col_kernel_double(
                (num_blocks,), (threads_per_block,), (out, dY, lengths, nW, B, I, nL)
            )

    return out


def backprop_clipped_linear(
    dY,
    X,
    *,
    slope: float = 1.0,
    offset: float = 0.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    inplace: bool = False,
    threads_per_block=128,
    num_blocks=128,
):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)

    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)

    if dY.dtype == "float32":
        backprop_clipped_linear_kernel_float(
            (num_blocks,),
            (threads_per_block,),
            (out, dY, X, slope, offset, min_val, max_val, out.size),
        )
    else:
        backprop_clipped_linear_kernel_double(
            (num_blocks,),
            (threads_per_block,),
            (out, dY, X, slope, offset, min_val, max_val, out.size),
        )

    return out


def backprop_hard_swish(
    dY, X, *, inplace: bool = False, threads_per_block=128, num_blocks=128
):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)

    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)

    if dY.dtype == "float32":
        backprop_hard_swish_kernel_float(
            (num_blocks,), (threads_per_block,), (out, dY, X, out.size)
        )
    else:
        backprop_hard_swish_kernel_double(
            (num_blocks,), (threads_per_block,), (out, dY, X, out.size)
        )

    return out


def backprop_hard_swish_mobilenet(
    dY, X, *, inplace: bool = False, threads_per_block=128, num_blocks=128
):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)

    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)

    if dY.dtype == "float32":
        backprop_hard_swish_mobilenet_kernel_float(
            (num_blocks,), (threads_per_block,), (out, dY, X, out.size)
        )
    else:
        backprop_hard_swish_mobilenet_kernel_double(
            (num_blocks,), (threads_per_block,), (out, dY, X, out.size)
        )

    return out


def backprop_gelu(
    dY,
    X,
    *,
    inplace: bool = False,
    threshold=6.0,
    threads_per_block=128,
    num_blocks=128,
):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)

    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)

    if dY.dtype == "float32":
        backprop_gelu_kernel_float(
            (num_blocks,), (threads_per_block,), (out, dY, X, threshold, out.size)
        )
    else:
        backprop_gelu_kernel_double(
            (num_blocks,), (threads_per_block,), (out, dY, X, threshold, out.size)
        )

    return out


def backprop_maxout(dY, which, P, *, threads_per_block=128, num_blocks=128):
    _is_float_array(dY)

    B = dY.shape[0]
    I = dY.shape[1]

    out = _alloc((B, I, P), dtype=dY.dtype, zeros=True)

    _check_which_maxout(which, B, I, P)

    if dY.dtype == "float32":
        backprop_maxout_kernel_float(
            (num_blocks,), (threads_per_block,), (out, dY, which, B, I, P)
        )
    else:
        backprop_maxout_kernel_double(
            (num_blocks,), (threads_per_block,), (out, dY, which, B, I, P)
        )

    return out


def backprop_mish(
    dY, X, *, inplace: bool = False, threshold=5, threads_per_block=128, num_blocks=128
):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)

    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)

    if dY.dtype == "float32":
        backprop_mish_kernel_float(
            (num_blocks,), (threads_per_block,), (out, dY, X, threshold, dY.size)
        )
    else:
        backprop_mish_kernel_double(
            (num_blocks,), (threads_per_block,), (out, dY, X, threshold, dY.size)
        )

    return out


def backprop_reduce_sum(d_sums, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(d_sums)

    B = len(lengths)
    T = int(lengths.sum())
    O = d_sums.shape[1]
    _check_lengths(lengths, T)

    out = _alloc((T, O), dtype=d_sums.dtype, zeros=False)

    if d_sums.dtype == "float32":
        backprop_reduce_sum_kernel_float(
            (num_blocks,), (threads_per_block,), (out, d_sums, lengths, B, T, O)
        )
    else:
        backprop_reduce_sum_kernel_double(
            (num_blocks,), (threads_per_block,), (out, d_sums, lengths, B, T, O)
        )

    return out


def backprop_reduce_mean(d_means, lengths, *, threads_per_block=128, num_blocks=128):
    _is_float_array(d_means)

    B = len(lengths)
    T = int(lengths.sum())
    O = d_means.shape[1]
    _check_lengths(lengths, T)

    out = _alloc((T, O), dtype=d_means.dtype, zeros=False)

    if d_means.dtype == "float32":
        backprop_reduce_mean_kernel_float(
            (num_blocks,), (threads_per_block,), (out, d_means, lengths, B, T, O)
        )
    else:
        backprop_reduce_mean_kernel_double(
            (num_blocks,), (threads_per_block,), (out, d_means, lengths, B, T, O)
        )

    return out


def backprop_reduce_max(
    d_maxes, which, lengths, *, threads_per_block=128, num_blocks=128
):
    _is_float_array(d_maxes)

    B = len(lengths)
    T = int(lengths.sum())
    O = d_maxes.shape[1]
    _check_lengths(lengths, T, min_length=1)

    out = _alloc((T, O), dtype=d_maxes.dtype, zeros=True)

    _check_which_reduce_max(which, (B, O), lengths)

    if d_maxes.dtype == "float32":
        backprop_reduce_max_kernel_float(
            (num_blocks,), (threads_per_block,), (out, d_maxes, which, lengths, B, T, O)
        )
    else:
        backprop_reduce_max_kernel_double(
            (num_blocks,), (threads_per_block,), (out, d_maxes, which, lengths, B, T, O)
        )

    return out


def backprop_swish(
    dY, X, Y, *, inplace=False, threshold=17.0, threads_per_block=128, num_blocks=128
):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)
    _is_float_array(Y, shape=dY.shape)

    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)

    if dY.dtype == "float32":
        backprop_swish_kernel_float(
            (num_blocks,), (threads_per_block,), (out, dY, X, Y, threshold, out.size)
        )
    else:
        backprop_swish_kernel_double(
            (num_blocks,), (threads_per_block,), (out, dY, X, Y, threshold, out.size)
        )

    return out


def hash(ids, seed, *, threads_per_block=128, num_blocks=128):
    out = _alloc((ids.shape[0], 4), dtype="uint32", zeros=True)

    # sizeof(uint32_t) * 4
    out_size = 4 * 4
    in_size = 8  # sizeof(uint64_t)
    # T = ids.shape[0]
    hash_data_kernel(
        (num_blocks,),
        (threads_per_block,),
        (out, ids, out_size, in_size, ids.shape[0], seed),
    )
    return out


def _is_float_array(out, *, shape: Optional[Tuple] = None):
    assert out.dtype in (
        "float32",
        "float64",
    ), "CUDA kernel can only handle float32 and float64"
    if shape is not None and out.shape != shape:
        msg = f"array has incorrect shape, expected: {shape}, was: {out.shape}"
        raise ValueError(msg)


def _check_lengths(lengths, n_elems: int, *, min_length=0):
    assert lengths.dtype == "int32", "lengths should be encoded as 32-bit integers"
    if not cupy.all(lengths >= min_length):
        raise ValueError(f"all sequence lengths must be >= {min_length}")
    if cupy.sum(lengths) != n_elems:
        raise IndexError("lengths must sum up to the batch size")


def _check_indices(indices, n: int):
    assert indices.dtype == "int32", "indices should be encoded as 32-bit integers"

    if not _values_within_range(indices, 0, n):
        raise IndexError(f"index out of bounds, must be >= 0 && < {n}")


def _check_which_maxout(which, B: int, I: int, P: int):
    shape = (B, I)
    msg = "maximum index (which) should be encoded as 32-bit integers"
    assert which.dtype == "int32", msg
    if which.shape != shape:
        msg = f"maximum index (which) has incorrect shape, expected: {shape}, was: {which.shape}"
        raise ValueError(msg)
    if not _values_within_range(which, 0, P):
        raise IndexError("maximum index (which) value out of bounds")


_values_within_range = (
    cupy.ReductionKernel(
        "T x, T lower, T upper",
        "bool r",
        "x >= lower && x < upper",
        "a & b",
        "r = a",
        "true",
        "within_range",
    )
    if has_cupy_gpu
    else None
)


def _check_which_reduce_max(which, shape: Tuple, lengths):
    msg = "maximum index (which) should be encoded as 32-bit integers"
    assert which.dtype == "int32", msg
    if which.shape != shape:
        msg = f"maximum index (which) has incorrect shape, expected: {shape}, was: {which.shape}"
        raise ValueError(msg)
    if not cupy.all((which >= 0) & (which < cupy.expand_dims(lengths, -1))):
        raise IndexError("maximum index (which) value out of bounds")
