from __future__ import unicode_literals
from numpy.testing import assert_allclose
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
SRC = (PWD / "_custom_kernels.cu").open("r", encoding="utf8").read()
KERNELS = compile_kernels(SRC)

MMH_SRC = (PWD / "_murmur3.cu").open("r", encoding="utf8").read()
KERNELS["hash"] = compile_mmh(MMH_SRC)

seq2col_kernel = KERNELS["seq2col"]
maxout_kernel = KERNELS["maxout"]
mish_kernel = KERNELS["mish"]
sum_pool_kernel = KERNELS["sum_pool"]
max_pool_kernel = KERNELS["max_pool"]

backprop_seq2col_kernel = KERNELS["backprop_seq2col"]
backprop_maxout_kernel = KERNELS["backprop_maxout"]
backprop_mish_kernel = KERNELS["backprop_mish"]
backprop_sum_pool_kernel = KERNELS["backprop_sum_pool"]
backprop_mean_pool_kernel = KERNELS["backprop_mean_pool"]
backprop_max_pool_kernel = KERNELS["backprop_max_pool"]
hash_data_kernel = compile_mmh(MMH_SRC)


def seq2col(X, nW, out=None, threads_per_block=128, num_blocks=128):
    if out is None:
        out = cupy.zeros((X.shape[0], X.shape[1] * ((nW*2)+1)), dtype="f")
    B = X.shape[0]
    I = X.shape[1]
    seq2col_kernel((num_blocks,), (threads_per_block,), (out, X, nW, B, I))
    return out

def maxout(X, out=None, threads_per_block=128, num_blocks=128):
    B, I, P = X.shape
    if out is None:
        best = cupy.zeros((B, I), dtype="f")
        which = cupy.zeros((B, I), dtype="i")
    else:
        best, which = None
    maxout_kernel((num_blocks,), (threads_per_block,),
        (best, which, X, B, I, P))
    return best, which


def mish(X, out=None, threshold=5, threads_per_block=128, num_blocks=128):
    N = X.size
    if out is None:
        out = cupy.zeros(X.shape, dtype="f")
    mish_kernel((num_blocks,), (threads_per_block,),
        (out, X, threshold, N))
    return out


def sum_pool(X, lengths, out=None, threads_per_block=128, num_blocks=128):
    if out is None:
        out = cupy.zeros((len(lengths), X.shape[1]), dtype="f")
    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]
    sum_pool_kernel((num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O))
    return out


def mean_pool(X, lengths, out=None, threads_per_block=128, num_blocks=128):
    if out is None:
        out = cupy.zeros((len(lengths), X.shape[1]), dtype="f")
    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]
    sum_pool_kernel((num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O))
    out /= lengths.reshape((-1, 1))
    return out


def max_pool(X, lengths, out=None, threads_per_block=128, num_blocks=128):
    if out is None:
        maxes = cupy.zeros((len(lengths), X.shape[1]), dtype="f")
        which = cupy.zeros((len(lengths), X.shape[1]), dtype="i")
    else:
        maxes, which = out
    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]
    max_pool_kernel((num_blocks,), (threads_per_block,),
        (maxes, which, X, lengths, B, T, O))
    return maxes, which


def backprop_seq2col(dY, nW, out=None, threads_per_block=128, num_blocks=128):
    B = dY.shape[0]
    nF = nW*2+1
    I = dY.shape[1] // nF
    if out is None:
        out = cupy.zeros((B, I), dtype="f")
    backprop_seq2col_kernel((num_blocks,), (threads_per_block,),
        (out, dY, nW, B, I))
    return out


def backprop_maxout(dY, which, P, out=None, threads_per_block=128, num_blocks=128):
    B = dY.shape[0]
    I = dY.shape[1]
    if out is None:
        out = cupy.zeros((B, I, P), dtype="f")
    backprop_maxout_kernel((num_blocks,), (threads_per_block,),
        (out, dY, which, B, I, P))
    return out

def backprop_mish(dY, X, out=None, threshold=5, threads_per_block=128, num_blocks=128):
    B = dY.shape[0]
    I = dY.shape[1]
    if out is None:
        out = cupy.zeros((B, I), dtype="f")
    backprop_mish_kernel((num_blocks,), (threads_per_block,),
        (out, dY, X, threshold, B*I))
    return out


def backprop_sum_pool(d_sum, lengths, out=None, threads_per_block=128, num_blocks=128):
    B = len(lengths)
    T = int(lengths.sum())
    O = d_sum.shape[1]
    if out is None:
        out = cupy.zeros((T, O), dtype="f")

    backprop_sum_pool_kernel((num_blocks,), (threads_per_block,),
        (out, d_sum, lengths, B, T, O))
    return out


def backprop_mean_pool(d_mean, lengths, out=None, threads_per_block=128, num_blocks=128):
    B = len(lengths)
    T = int(lengths.sum())
    O = d_mean.shape[1]
    if out is None:
        out = cupy.zeros((T, O), dtype="f")

    backprop_mean_pool_kernel((num_blocks,), (threads_per_block,),
        (out, d_mean, lengths, B, T, O))
    return out


def backprop_max_pool(d_maxes, which, lengths, out=None,
        threads_per_block=128, num_blocks=128):
    B = len(lengths)
    T = int(lengths.sum())
    O = d_maxes.shape[1]
    if out is None:
        out = cupy.zeros((T, O), dtype="f")

    backprop_max_pool_kernel((num_blocks,), (threads_per_block,),
        (out, d_maxes, which, lengths, B, T, O))
    return out


def hash(ids, seed, out=None, threads_per_block=128, num_blocks=128):
    if out is None:
        out = cupy.zeros((ids.shape[0], 4), dtype="uint32")
    # sizeof(uint32_t) * 4
    out_size = 4 * 4
    in_size = 8 # sizeof(uint64_t)
    T = ids.shape[0]
    hash_data_kernel((num_blocks,), (threads_per_block,),
        (out, ids, out_size, in_size, ids.shape[0], seed))
    return out
