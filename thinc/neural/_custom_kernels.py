from numpy.testing import assert_allclose

try:
    import cupy
except ImportError:
    cupy = None


SUM_POOL_STR = """
extern "C" __global__
void sum_pool(float* output,
    const float* X, const int* lengths, int B, int T, int O)
{
    // Compute sums of a batch of concatenated sequences
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch-item we're working on
    if (b >= B) return;

    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        output += O;
	X += lengths[i] * O;
    }
    int length = lengths[b];
    // Each invocation of the kernel sums one batch.
    for (int _=0; _ < length; ++_) // Iterate over rows
    {
        for (int i=0; i < O; ++i) 
        {
          output[i] += X[i];
        }
        X += O;
    }
}
"""


MAXOUT_STR = """
extern "C" __global__
void maxout(float* best, int* which,
        const float* cands, int B, int O, int P)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; 
    if (b >= B) return;

    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        best += O;
        which += O;
        cands += O * P;
    }

    for (int i=0; i < O; ++i)
    {
        which[i] = 0
        best[i] = cands[0];
        for (int p=1; p < P; ++p)
	{
            if (cands[i+p] > best[0])
	    {
                which[i] = p;
                best[i] = cands[i+p];
	    }
	}
    }
}
"""


MAX_POOL_STR = """
extern "C" __global__
void max_pool(float* maxes, int* which,
    const float* X, const int* lengths, int B, int T, int O)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch-item we're working on
    if (b >= B) return;

    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        maxes += O;
        which += O;
	X += lengths[i] * O;
    }
 
    // Each invocation of the kernel maxes one batch.
    // Start by assuming maxes are at i=0
    for (int j=0; j < O; ++j) {
        maxes[j] = X[j];
	which[j] = 0;
    }
    X += O;
    
    int length = lengths[b];
    for (int i=1; i < length; ++i) // Iterate over rows
    {
        for (int j=0; j < O; ++j)
	{
            if (X[j] > maxes[j])
            {
                maxes[j] = X[j];
                which[j] = i;
	    }
	}
	X += O;
    }
}
"""


if cupy is not None:
    sum_pool_kernel = cupy.RawKernel(SUM_POOL_STR, "sum_pool")
    max_pool_kernel = cupy.RawKernel(MAX_POOL_STR, "max_pool")


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


test_sum_pool()
test_max_pool()
