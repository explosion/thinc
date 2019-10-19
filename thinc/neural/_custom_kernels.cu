extern "C" __global__
void seq2col(float* output,
    const float* X, int nW, int B, int I)
{
    // Let's say nW is 1 (it usually is). Then we want to take:

    // 1a 1b 1c
    // 2a 2b 2c
    // 3a 3b 3c

    // And make

    // __ __ __ 1a 1b 1c 2a 2b 2c
    // 1a 1b 1c 2a 2b 2c 3a 3b 3c
    // 2a 2b 2c 3a 3b 3c __ __ __

    // Where __ is padding.

    // Now let's say nW is 2. Then we want to take:

    // 1a 1b 1c
    // 2a 2b 2c
    // 3a 3b 3c

    // And make

    // __ __ __ __ __ __ 1a 1b 1c 2a 2b 2c 3a 3b 3c
    // __ __ __ 1a 1b 1c 2a 2b 2c 3a 3b 3c __ __ __
    // 1a 1b 1c 2a 2b 2c 3a 3b 3c __ __ __ __ __ __
    
    // * x_start=-6, x_end=9 : (0-2) * 3, (0+2+1) * 3
    // * x_start=-3, x_end=13 : (1-2) * 3, (1+2+1) * 3
    // * x_start=0, x_end=16 : (2-2) * 3, (2+2+1) * 3
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch-item we're working on
    if (b >= B) return;
    int nF = nW * 2 + 1;

    int o_start = b * I * nF;
    int x_start = (b-nW) * I;
    int x_end = (b+nW+1) * I;
    if (x_start < 0)
    {
        o_start += -x_start;
        x_start = 0;
    }
    if (x_end >= B * I)
    {
        x_end = B * I;
    }
    // Unsure which memcpy function to use on CUDA..
    // Shrug, just write the loop...
    int cpy_length = x_end - x_start;
    for (int i=0; i<cpy_length; ++i)
    {
        output[o_start+i] = X[x_start+i];
    }
}




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

extern "C" __global__
void backprop_seq2col(float* d_seqs,
    const float* d_cols, int B, int I, int nW)
{
    // Here's what we're doing, if we had 2d indexing.
    //for i in range(B):
    //    d_seq[i] += d_cols[i-2, 4]
    //    d_seq[i] += d_cols[i-1, 3]
    //    d_seq[i] += d_cols[i, 2]
    //    d_seq[i] += d_cols[i+1, 1]
    //    d_seq[i] += d_cols[i+2, 0]
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Batch-item we're working on
    if (b >= B) return;

    int nF = nW * 2 + 1;
    int seq_row = b * I;
    int col_feat = nF * I;
    for (int f=-nW; f < (nW+1); ++f)
    {
	int col_row = (b+f) * (I * nF);
	col_feat -= I;
	if ((col_row >= 0) && (col_row < (B*I*nF)))
	{
            int start = col_row + col_feat;
	    if ((start >= 0) && ((start+I) < (B*I*nF)))
	    {
	        for (int i=0; i < I; ++i)
		    d_seqs[seq_row+i] += d_cols[start+i];
	    }
	}
    }
}


extern "C" __global__
void backprop_sum_pool(float* dX, const float* d_sum, const int* lengths,
    int B, int T, int O)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // row we're working on
    if (row >= T) return;
    
    // Find the sequence item we're working on
    int seq_start = 0;
    int b = 0;
    while ((b < B) && (seq_start+lengths[b]) < row)
    {
       seq_start += lengths[b];
       b += 1;
    }
        
    dX = &dX[row * O];
    d_sum = &d_sum[b * O];

    for (int i=0; i < O; ++i) 
    {
	dX[i] = d_sum[i];
    }
}


extern "C" __global__
void backprop_max_pool(float* dX,
    const float* d_maxes, const int* which, const int* lengths, int B, int T, int O)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Batch-item we're working on
    if (row >= T) return;
    
    // Find the sequence item we're working on
    int seq_start = 0;
    int b = 0;
    while ((b < B) && (seq_start+lengths[b]) < row)
    {
       seq_start += lengths[b];
       b += 1;
    }

    dX = &dX[row];
    which = &which[b];
    d_maxes = &d_maxes[b];
 
    for (int j=0; j < O; ++j)
    {
	// If we used the value for this cell,
	// pass the gradient
        if (which[j] == (row-seq_start))
           dX[j] = d_maxes[j];
    }
}
