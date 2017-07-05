#include <stdio.h>


// Replace this with the saxpy from cuBlas or whatever?
// I doubt it matters, but it's definitely weird to have this
void __device__
saxpy(float* X, const float* Y, float scale, int n)
{
    for (int i=0; i < n; ++i) 
        X[i] += Y[i] * scale;
}


void __global__
maxout(float* best__bo, int* which__bo,
        const float* cands__bop, int B, int O, int P)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; 
    if (b >= B) return;

    for (int o=0; o < O; ++o)
    {
        which__bo[0] = 0;
        best__bo[0] = cands__bop[0];
        cands__bop += 1;
        for (int p=1; p < P; ++p)
	{
            if (cands__bop[0] > best__bo[0])
	    {
                which__bo[0] = p;
                best__bo[0] = cands__bop[0];
	    }
            cands__bop += 1;
	}
        best__bo += 1;
        which__bo += 1;
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
sum_pool(float* sums__bo,
    const float* X__to, const int* lengths__b, int B, int T, int O)
{
    // Compute sums of a batch of concatenated sequences, using the lengths.'''
    int b = blockIdx.x; // Batch-item we're summing
    if (b >= B) return;

    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        sums__bo += O;
	X__to += lengths__b[i] * O;
    }

    int length = lengths__b[b];
    // Each invocation of the kernel sums one batch.
    for (int _=0; _ < length; ++_) // Iterate over rows
    {
        saxpy(sums__bo, X__to, 1.0, O);
        X__to += O;
    }
}


void __global__
max_pool(float* maxes__bo, int* which__bo,
    const float* X__to, const int* lengths__b, int B, int T, int O)
{
    // Compute means of a batch of concatenated sequences, using the lengths.'''
    int b = blockIdx.x; // Batch-item we're averaging
    if (b >= B) return;

    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        maxes__bo += O;
	which__bo += O;
	X__to += lengths__b[i] * O;
    }

    // Each invocation of the kernel maxes one batch.
    // Start by assuming maxes are at i=0
    for (int j=0; j < O; ++j) {
        maxes__bo[j] = X__to[j];
	which__bo[j] = 0;
    }
    X__to += O;
    
    int length = lengths__b[b];
    for (int i=1; i < length; ++i) // Iterate over rows
    {
        for (int j=0; j < O; ++j)
	{
            if (X__to[j] > maxes__bo[j])
            {
                maxes__bo[j] = X__to[j];
                which__bo[j] = i;
	    }
	}
	X__to += O;
    }
}


void __global__
backprop_mean_pool(float* dX__to, const float* d_means__bo, const int* lengths__b,
    int B, int T, int O)
{
    int b = blockIdx.x; // Batch-item we're averaging
    if (b >= B) return;
    
    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        d_means__bo += O;
	dX__to += lengths__b[i] * O;
    }

    int length = lengths__b[b];
    float scale = 1./ length;
    
    for (int _=0; _ < length; _++)
    {
        saxpy(dX__to, d_means__bo, scale, O);
        dX__to += O;
    }
}


void __global__
backprop_sum_pool(float* dX__to, const float* d_sum__bo, const int* lengths__b,
    int B, int T, int O)
{
    int b = blockIdx.x; // Batch-item we're averaging
    if (b >= B) return;
    
    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        d_sum__bo += O;
	dX__to += lengths__b[i] * O;
    }

    int length = lengths__b[b];
    
    for (int _=0; _ < length; _++)
    {
        saxpy(dX__to, d_sum__bo, 1.0, O);
        dX__to += O;
    }
}


void __global__
backprop_max_pool(float* dX__to,
    const float* d_maxes__bo, const int* which__bo, const int* lengths__b, int B, int T, int O)
{
    int b = blockIdx.x; // Batch-item we're averaging
    if (b >= B) return;
    
    // Go to the regions we're working on
    for (int i=0; i < b; ++i) {
        d_maxes__bo += O;
	which__bo += O;
	dX__to += lengths__b[i] * O;
    }

    int length = lengths__b[b];
 
    for (int i=0; i < length; ++i)
    {
       for (int j=0; j < O; ++j)
       {
         if (which__bo[j] == i)
           dX__to[j] += d_maxes__bo[j];
       }
       dX__to += O;
    }
}
