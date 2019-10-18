from numpy.testing import assert_allclose
import re
from pathlib import Path

try:
    import cupy
except ImportError:
    cupy = None


kernel_re = re.compile(r"extern \"C\" __global__.+(?=extern|$)", re.DOTALL)
name_re = re.compile(r"(?<=void )\w+(?=\()")
def parse_kernels(src):
    kernels = {}
    for kernel in kernel_re.findall(src):
        name = name_re.search(kernel).group()
        kernels[name] = kernel
    return kernels
    
SRC = (Path(__file__).parent / "_custom_kernels.cu").open().read()
kernels = parse_kernels(SRC)
for name, src in kernels.items():
    print(name)
    print(src)


#if cupy is not None:
#    sum_pool_kernel = cupy.RawKernel(SUM_POOL_STR, "sum_pool")
#    max_pool_kernel = cupy.RawKernel(MAX_POOL_STR, "max_pool")
#

def sum_pool(X, lengths, out=None, threads_per_block=128):
    if out is None:
        out = cupy.zeros((len(lengths), X.shape[1]), dtype="f")
    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]
    num_blocks = min(1, B // threads_per_block)
    sum_pool_kernel((num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O))
    return out


def mean_pool(X, lengths, out=None, threads_per_block=128):
    if out is None:
        out = cupy.zeros((len(lengths), X.shape[1]), dtype="f")
    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]
    num_blocks = min(1, B // threads_per_block)
    sum_pool_kernel((num_blocks,), (threads_per_block,), (out, X, lengths, B, T, O))
    out /= lengths
    return out


def max_pool(X, lengths, out=None, threads_per_block=128):
    if out is None:
        maxes = cupy.zeros((len(lengths), X.shape[1]), dtype="f")
        which = cupy.zeros((len(lengths), X.shape[1]), dtype="i")
    else:
        maxes, which = out
    B = len(lengths)
    T = X.shape[0]
    O = X.shape[1]
    num_blocks = min(1, B // threads_per_block)
    max_pool_kernel((num_blocks,), (threads_per_block,),
        (maxes, which, X, lengths, B, T, O))
    return maxes, which


def test_sum_pool():
    m = cupy.zeros((19, 5), dtype="f")
    m += 1
    lengths = cupy.array([5,5,3,6], dtype="i")
    output = sum_pool(m, lengths)
    assert output.sum() == m.sum(), (output.sum(), m.sum())


def test_max_pool():
    m = cupy.zeros((19, 5), dtype="f")
    m += cupy.random.uniform(-1, 1, m.shape)
    lengths = cupy.array([5,5,3,6], dtype="i")
    m[4, 0] = 1
    m[0, 1] = 2
    m[1, 3] = 3
    maxes, which = max_pool(m, lengths)
    start = 0
    for i, length in enumerate(lengths):
        truth = m[start:start+length].max(axis=0)
        assert_allclose(maxes[i].get(), truth.get())
        start += length


#test_sum_pool()
#test_max_pool()
