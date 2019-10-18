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
