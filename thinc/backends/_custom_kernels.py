from typing import Tuple
import re
from pathlib import Path
from collections import defaultdict

try:
    import cupy
except ImportError:
    cupy = None


kernel_re = re.compile(r"extern \"C\" __global__.+?(?=extern|$)", re.DOTALL)
name_re = re.compile(r"(?<=void )\w+(?=\()")


def parse_kernels(src):
    kernels = {}
    for kernel in kernel_re.findall(src):
        name = name_re.search(kernel).group()
        kernels[name] = kernel
    return kernels


def compile_kernels(src):
    if cupy is None:
        return defaultdict(lambda: None)
    kernels = parse_kernels(src)
    return {name: cupy.RawKernel(src, name) for name, src in kernels.items()}


def compile_mmh(src):
    if cupy is None:
        return None
    return cupy.RawKernel(src, "hash_data")


PWD = Path(__file__).parent
SRC = (PWD / "_custom_kernels.cu").read_text(encoding="utf8")
KERNELS = compile_kernels(SRC)

MMH_SRC = (PWD / "_murmur3.cu").read_text(encoding="utf8")
KERNELS["hash"] = compile_mmh(MMH_SRC)

clipped_linear_kernel = KERNELS["clipped_linear"]
gelu_kernel = KERNELS["gelu"]
hash_data_kernel = compile_mmh(MMH_SRC)
maxout_kernel = KERNELS["maxout"]
mish_kernel = KERNELS["mish"]
reduce_max_kernel = KERNELS["reduce_max"]
reduce_sum_kernel = KERNELS["reduce_sum"]
seq2col_kernel = KERNELS["seq2col"]
swish_kernel = KERNELS["swish"]

backprop_clipped_linear_kernel = KERNELS["backprop_clipped_linear"]
backprop_gelu_kernel = KERNELS["backprop_gelu"]
backprop_hard_swish_kernel = KERNELS["backprop_hard_swish"]
backprop_hard_swish_mobilenet_kernel = KERNELS["backprop_hard_swish_mobilenet"]
backprop_maxout_kernel = KERNELS["backprop_maxout"]
backprop_mish_kernel = KERNELS["backprop_mish"]
backprop_reduce_max_kernel = KERNELS["backprop_reduce_max"]
backprop_reduce_mean_kernel = KERNELS["backprop_reduce_mean"]
backprop_reduce_sum_kernel = KERNELS["backprop_reduce_sum"]
backprop_seq2col_kernel = KERNELS["backprop_seq2col"]
backprop_swish_kernel = KERNELS["backprop_swish"]


def clipped_linear(
    X,
    inplace=False,
    slope=1.0,
    offset=0.0,
    min_val=0.0,
    max_val=1.0,
    threads_per_block=128,
    num_blocks=128,
):
    assert X.dtype == "float32", "CUDA clipped_linear kernel can only handle float32"

    out = X
    if not inplace:
        out = cupy.zeros_like(X, dtype="f")
    clipped_linear_kernel(
        (num_blocks,),
        (threads_per_block,),
        (out, X, slope, offset, min_val, max_val, X.size),
    )
    return out


def gelu(X, inplace=False, threshold=6.0, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA gelu kernel can only handle float32"

    out = X
    if not inplace:
        out = cupy.zeros_like(X, dtype="f")
    gelu_kernel((num_blocks,), (threads_per_block,), (out, X, threshold, X.size))
    return out


def check_seq2col_lengths(lengths, B):
    if lengths is None:
        lengths = cupy.array([B], dtype="int32")
    else:
        _check_lengths(lengths, B)
    return lengths


def seq2col(X, nW, *, lengths=None, out=None, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA seq2col kernel can only handle float32"

    B = X.shape[0]
    nF = nW * 2 + 1
    I = X.shape[1]

    lengths = check_seq2col_lengths(lengths, B)
    nL = lengths.shape[0]

    out_shape = (B, I * nF)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    if X.size != 0 and lengths.size != 0:
        seq2col_kernel(
            (num_blocks,), (threads_per_block,), (out, X, lengths, nW, B, I, nL)
        )

    return out


def maxout(X, out=None, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA maxout kernel can only handle float32"

    B, I, P = X.shape

    out_shape = (B, I)
    if out is None:
        best = cupy.zeros(out_shape, dtype="f")
        which = cupy.zeros(out_shape, dtype="i")
    else:
        best, which = out
        _check_array(best, out_shape)
        _check_which(which, B, I, P)

    maxout_kernel((num_blocks,), (threads_per_block,), (best, which, X, B, I, P))
    return best, which


def mish(X, inplace=False, threshold=5, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA mish kernel can only handle float32"

    out = X
    if not inplace:
        out = cupy.zeros_like(X, dtype="f")
    mish_kernel((num_blocks,), (threads_per_block,), (out, X, threshold, X.size))
    return out


def reduce_sum(X, lengths, out=None, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA reduce_sum kernel can only handle float32"

    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]

    _check_lengths(lengths, T)

    out_shape = (B, O)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    reduce_sum_kernel((num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O))
    return out


def reduce_mean(X, lengths, out=None, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA reduce_mean kernel can only handle float32"

    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]

    _check_lengths(lengths, T)

    out_shape = (B, O)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    reduce_sum_kernel((num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O))
    # Avoid divide by zero
    out /= lengths.reshape((-1, 1)) + 1e-10
    return out


def reduce_max(X, lengths, out=None, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA reduce_max kernel can only handle float32"

    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]

    _check_lengths(lengths, T)

    out_shape = (B, O)
    if out is None:
        maxes = cupy.zeros(out_shape, dtype="f")
        which = cupy.zeros(out_shape, dtype="i")
    else:
        maxes, which = out
        _check_array(maxes, out_shape)
        _check_which(which, B, I, P)

    reduce_max_kernel(
        (num_blocks,), (threads_per_block,), (maxes, which, X, lengths, B, T, O)
    )
    return maxes, which


def swish(X, inplace=False, threshold=17.0, threads_per_block=128, num_blocks=128):
    assert X.dtype == "float32", "CUDA swish kernel can only handle float32"

    out = X
    if not inplace:
        out = cupy.zeros_like(X, dtype="f")
    swish_kernel((num_blocks,), (threads_per_block,), (out, X, threshold, X.size))
    return out


def backprop_seq2col(
    dY, nW, *, lengths=None, out=None, threads_per_block=128, num_blocks=128
):
    assert dY.dtype == "float32", "CUDA backprop_seq2col kernel can only handle float32"

    B = dY.shape[0]
    nF = nW * 2 + 1
    I = dY.shape[1] // nF

    lengths = check_seq2col_lengths(lengths, B)
    nL = lengths.shape[0]

    out_shape = (B, I)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    if dY.size != 0 and lengths.size != 0:
        backprop_seq2col_kernel(
            (num_blocks,), (threads_per_block,), (out, dY, lengths, nW, B, I, nL)
        )

    return out


def backprop_clipped_linear(
    dY,
    X,
    slope: float = 1.0,
    offset: float = 0.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    inplace: bool = False,
    threads_per_block=128,
    num_blocks=128,
):
    assert (
        dY.dtype == "float32"
    ), "CUDA backprop_clipped_linear kernel can only handle float32"
    _check_array(X, dY.shape)

    out = dY
    if not inplace:
        out = cupy.zeros_like(dY, dtype="f")
    backprop_clipped_linear_kernel(
        (num_blocks,),
        (threads_per_block,),
        (out, dY, X, slope, offset, min_val, max_val, out.size),
    )
    return out


def backprop_hard_swish(
    dY, X, inplace: bool = False, threads_per_block=128, num_blocks=128
):
    assert (
        dY.dtype == "float32"
    ), "CUDA backprop_hard_swish kernel can only handle float32"
    _check_array(X, dY.shape)

    out = dY
    if not inplace:
        out = cupy.zeros_like(dY, dtype="f")
    backprop_hard_swish_kernel(
        (num_blocks,), (threads_per_block,), (out, dY, X, out.size)
    )
    return out


def backprop_hard_swish_mobilenet(
    dY, X, inplace: bool = False, threads_per_block=128, num_blocks=128
):
    assert (
        dY.dtype == "float32"
    ), "CUDA backprop_hard_swish_mobilenet kernel can only handle float32"
    _check_array(X, dY.shape)

    out = dY
    if not inplace:
        out = cupy.zeros_like(dY, dtype="f")
    backprop_hard_swish_mobilenet_kernel(
        (num_blocks,), (threads_per_block,), (out, dY, X, out.size)
    )
    return out


def backprop_gelu(
    dY, X, inplace: bool = False, threshold=6.0, threads_per_block=128, num_blocks=128
):
    assert dY.dtype == "float32", "CUDA backprop_gelu kernel can only handle float32"
    _check_array(X, dY.shape)

    out = dY
    if not inplace:
        out = cupy.zeros_like(dY, dtype="f")
    backprop_gelu_kernel(
        (num_blocks,), (threads_per_block,), (out, dY, X, threshold, out.size)
    )
    return out


def backprop_maxout(dY, which, P, out=None, threads_per_block=128, num_blocks=128):
    assert dY.dtype == "float32", "CUDA backprop_maxout kernel can only handle float32"

    B = dY.shape[0]
    I = dY.shape[1]

    out_shape = (B, I, P)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    _check_which(which, B, I, P, check_values=True)

    backprop_maxout_kernel(
        (num_blocks,), (threads_per_block,), (out, dY, which, B, I, P)
    )
    return out


def backprop_mish(
    dY, X, inplace: bool = False, threshold=5, threads_per_block=128, num_blocks=128
):
    assert dY.dtype == "float32", "CUDA backprop_mish kernel can only handle float32"
    _check_array(X, dY.shape)

    out = dY
    if not inplace:
        out = cupy.zeros_like(dY, dtype="f")
    backprop_mish_kernel(
        (num_blocks,), (threads_per_block,), (out, dY, X, threshold, dY.size)
    )
    return out


def backprop_reduce_sum(
    d_sum, lengths, out=None, threads_per_block=128, num_blocks=128
):
    assert (
        d_sum.dtype == "float32"
    ), "CUDA backprop_reduce_sum kernel can only handle float32"

    B = len(lengths)
    T = int(lengths.sum())
    O = d_sum.shape[1]
    _check_lengths(lengths, T)

    out_shape = (T, O)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    backprop_reduce_sum_kernel(
        (num_blocks,), (threads_per_block,), (out, d_sum, lengths, B, T, O)
    )
    return out


def backprop_reduce_mean(
    d_mean, lengths, out=None, threads_per_block=128, num_blocks=128
):
    assert (
        d_mean.dtype == "float32"
    ), "CUDA backprop_reduce_mean kernel can only handle float32"

    B = len(lengths)
    T = int(lengths.sum())
    O = d_mean.shape[1]
    _check_lengths(lengths, T)

    out_shape = (T, O)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    backprop_reduce_mean_kernel(
        (num_blocks,), (threads_per_block,), (out, d_mean, lengths, B, T, O)
    )
    return out


def backprop_reduce_max(
    d_maxes, which, lengths, out=None, threads_per_block=128, num_blocks=128
):
    assert (
        d_maxes.dtype == "float32"
    ), "CUDA backprop_reduce_max kernel can only handle float32"

    B = len(lengths)
    T = int(lengths.sum())
    O = d_maxes.shape[1]
    _check_lengths(lengths, T)

    out_shape = (T, O)
    if out is None:
        out = cupy.zeros(out_shape, dtype="f")
    else:
        _check_array(out, out_shape)

    _check_which(which, B, T, O, check_values=True)

    backprop_reduce_max_kernel(
        (num_blocks,), (threads_per_block,), (out, d_maxes, which, lengths, B, T, O)
    )
    return out


def backprop_swish(
    dY, X, Y, inplace=False, threshold=17.0, threads_per_block=128, num_blocks=128
):
    assert dY.dtype == "float32", "CUDA backprop_swish kernel can only handle float32"
    _check_array(X, dY.shape)
    _check_array(Y, dY.shape)

    out = dY
    if not inplace:
        out = cupy.zeros_like(dY, dtype="f")
    backprop_swish_kernel(
        (num_blocks,), (threads_per_block,), (out, dY, X, Y, threshold, out.size)
    )
    return out


def hash(ids, seed, out=None, threads_per_block=128, num_blocks=128):
    out_shape = (ids.shape[0], 4)
    if out is None:
        out = cupy.zeros(out_shape, dtype="uint32")
    else:
        assert out.shape == out_shape, "out has incorrect shape"

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


def _check_array(out, shape: Tuple):
    assert out.dtype == "float32", "CUDA kernel can only handle float32"
    assert out.shape == shape, "array has incorrect shape"


def _check_lengths(lengths, n_elems: int):
    assert lengths.dtype == "int32", "lengths should be encoded as 32-bit integers"
    assert cupy.all(lengths >= 0), "all sequence lengths must be >= 0"
    assert cupy.sum(lengths) == n_elems, "the lengths must sum up to the batch size"


def _check_which(which, B: int, I: int, P: int, check_values: bool = False):
    assert (
        which.dtype == "int32"
    ), "maximum index (which) should be encoded as 32-bit integers"
    assert which.shape == (B, I), "maximum index (which) has incorrect shape"
    if check_values:
        assert cupy.all(
            (which >= 0) & (which < P)
        ), "maximum index (which) value out of bounds"
