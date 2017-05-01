#include <stdio.h>


void __device__
saxpy(float* X, const float* Y, float scale, int n)
{
    for (int i=0; i < n; ++i) // Iterate over cols
        X[i] += Y[i] * scale;
}


void __global__ kernel_add_one(int* a, int length) {
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    while(gid < length) {
    	a[gid] += 1;
        gid += blockDim.x*gridDim.x;
    }
}


void __global__
mean_pool(float* means__bo,
    const float* X__to, const int* lengths__b, int B, int T, int O)
{
    // Compute means of a batch of concatenated sequences, using the lengths.'''
    int b = blockIdx.x; // Batch-item we're averaging
    if (b >= B) return;

    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        means__bo += O;
	X__to += lengths__b[i] * O;
    }

    int length = lengths__b[b];
    // Each invocation of the kernel averages one batch.
    float scale = 1. / length;
    for (int _=0; _ < length; ++_) // Iterate over rows
    {
        saxpy(means__bo, X__to, scale, O);
        X__to += O;
    }
}

void __global__
backprop_mean_pool(float* dX__to, const float* d_means__bo, const int* lengths__b,
    int B, int T, int O)
{
    int b = blockIdx.x; // Batch-item we're averaging
    if (b >= B) return;

    int length = lengths__b[b];
    float scale = 1./ length;
    
    for (int _=0; _ < length; _++)
    {
        saxpy(dX__to, d_means__bo, scale, O);
        dX__to += O;
        d_means__bo += O;
    }
}


