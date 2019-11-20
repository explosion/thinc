// Use grid strided loops, descriped here:
// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
// This pattern ensures that all of the loop values are visited once, no matter
// what grid parameters are used for the function.
	
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
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int nF = nW * 2 + 1;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
        int o_start = b * I * nF;
        // Let's say b=0, nW=1, I=10, B=20
        // x_start = (0-1) * 10 : -10
        // x_end = (0+1+1)*10 : 20
        // o_start = (0*0*3) = 0
        int x_start = (b-nW) * I;
        int x_end = (b+nW+1) * I;
        if (x_start < 0)
        {
            // Adjust o_start to 10, because we're skipping
            // the first feature
            o_start += -x_start;
            x_start = 0;
        }
        if (x_end >= (B * I))
        {
            x_end = B * I;
        }
        // cpy_length = 20-0 : 20
        // Unsure which memcpy function to use on CUDA..
        // Shrug, just write the loop...
        int cpy_length = x_end - x_start;
        for (int i=0; i<cpy_length; ++i)
        {
            // Write the region output[10:30] = X[0:20]
            output[o_start+i] = X[x_start+i];
        }
    }
}


extern "C" __global__
void maxout(float* best, int* which,
        const float* cands, int B, int O, int P)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
        // Go to the regions we're working on
        float* best_b = &best[b*O];
        int* which_b = &which[b*O];

        for (int i=0; i < O; ++i)
        {
            const float* cands_bi = &cands[b*O*P+(i*P)];
            which_b[i] = 0;
            best_b[i] = cands_bi[0];
            for (int p=1; p < P; ++p)
            {
                if (cands_bi[p] > best_b[i])
                {
                    which_b[i] = p;
                    best_b[i] = cands_bi[p];
                }
            }
        }
    }
}

extern "C" __global__
void mish(float* Y, const float* X, float threshold, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    float one = 1.;
    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        if (X[i] >= threshold)
	    Y[i] = X[i];
	else
            Y[i] = X[i] * tanhf(logf(one + expf(X[i])));
    }
} 


extern "C" __global__
void sum_pool(float* output,
    const float* X, const int* lengths, int B, int T, int O)
{
    // Compute sums of a batch of concatenated sequences
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
        // Go to the regions we're working on
	float* output_b = &output[b*O];
        // Find the sequence item we're working on
	int t = 0;
        for (int i=0; i < b; ++i) {
	    t += lengths[i];
        }
        int length = lengths[b];
        // Each invocation of the kernel sums one batch.
        for (int i=0; i < length; ++i) // Iterate over rows
        {
	    const float* X_t = &X[(t+i)*O];
            for (int j=0; j < O; ++j) 
            {
              output_b[j] += X_t[j];
            }
        }
    }
}


extern "C" __global__
void max_pool(float* maxes, int* which,
    const float* X, const int* lengths, int B, int T, int O)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {

        // Go to the regions we're working on
        float* maxes_b = &maxes[b*O];
        int* which_b = &which[b*O];
        // Find the sequence item we're working on
        const float* X_t = X;
        for (int i=0; i < b; ++i) {
	    X_t += lengths[i] * O;
        }
        // Each invocation of the kernel maxes one sequence.
        // Start by assuming maxes are the first element.
        for (int i=0; i < O; ++i) {
            maxes_b[i] = X_t[i];
            which_b[i] = 0;
        }
        int length = lengths[b];
        for (int i=1; i < length; ++i) // Iterate over rows
        {
            X_t += O;
            for (int j=0; j < O; ++j)
            {
                if (X_t[j] > maxes_b[j])
                {
                    maxes_b[j] = X_t[j];
                    which_b[j] = i;
                }
            }
        }

    }
}

extern "C" __global__
void backprop_seq2col(float* d_seqs,
    const float* d_cols, int nW, int B, int I)
{
    // Here's what we're doing, if we had 2d indexing.
    //for i in range(B):
    //    d_seq[i] += d_cols[i-2, 4]
    //    d_seq[i] += d_cols[i-1, 3]
    //    d_seq[i] += d_cols[i, 2]
    //    d_seq[i] += d_cols[i+1, 1]
    //    d_seq[i] += d_cols[i+2, 0]

    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int nF = nW * 2 + 1;
    int end_d_cols = B * I * nF;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
        float* d_seqs_b = &d_seqs[b*I];
        int col_feat = nF * I;
        for (int f=-nW; f < (nW+1); ++f)
        {
            int col_row = (b+f) * (I*nF);
            col_feat -= I;
            if ((col_row >= 0) && (col_row < end_d_cols))
            {
                int start = col_row + col_feat;
                if ((start >= 0) && ((start+I) < end_d_cols))
                {
                    for (int i=0; i < I; ++i)
                        d_seqs_b[i] += d_cols[start+i];
                }
            }
        }
    }
}


extern "C" __global__
void backprop_maxout(float* dX,
    const float* dY, const int* which, int B, int O, int P)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
	// Go to the regions we're working on
	float* dX_b = &dX[b*O*P];
	const float* dY_b = &dY[b*O];
	const int* which_b = &which[b*O];
        for (int i=0; i < O; ++i)
            dX_b[(i*P)+which_b[i]] = dY_b[i];
    }
}
 

extern "C" __global__
void backprop_mish(float* dX,
    const float* dY, const float* X, float threshold, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    float two = 2.;
    for (int i = _loop_start; i < N; i += _loop_stride)
    {
	float x = X[i];
	if (x >= threshold)
        {
	    dX[i] = dY[i];
	} else
	{
	    float exp_x = exp(x);
	    float exp_2x = exp(2*x);
	    float exp_3x = exp(3*x);

	    float omega = (4. * (x+1)) + (4 * exp_2x) + exp_3x + exp_x * (4.*x+6);
	    float delta = 2 * exp_x + exp_2x + 2;
	    dX[i] = dY[i] * ((exp_x * omega) / pow(delta, two));
	}
    }
}
 
extern "C" __global__
void backprop_sum_pool(float* dX, const float* d_sum, const int* lengths,
    int B, int T, int O)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int seq_start = 0;
    int b = 0;
    for (int t = _loop_start; t < T; t += _loop_stride)
    { 
        // Find the sequence item we're working on
        while ((b < B) && (seq_start+lengths[b]) < t)
        {
           seq_start += lengths[b];
           b += 1;
        }
            
        float* dX_t = &dX[t * O];
        const float* d_sum_b = &d_sum[b * O];

        for (int i=0; i < O; ++i) 
        {
            dX_t[i] = d_sum_b[i];
        }
    }
}


extern "C" __global__
void backprop_mean_pool(float* dX, const float* d_mean, const int* lengths,
    int B, int T, int O)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int seq_start = 0;
    int b = 0;
    for (int t = _loop_start; t < T; t += _loop_stride)
    { 
        // Find the sequence item we're working on
        while ((b < B) && (seq_start+lengths[b]) < t)
        {
           seq_start += lengths[b];
           b += 1;
        }
            
        float* dX_t = &dX[t * O];
        const float* d_mean_b = &d_mean[b * O];
        int lengths_b = lengths[b];

        for (int i=0; i < O; ++i) 
        {
            dX_t[i] = d_mean_b[i] / lengths_b;
        }
    }
}


extern "C" __global__
void backprop_max_pool(float* dX,
    const float* d_maxes, const int* which, const int* lengths, int B, int T, int O)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int seq_start = 0;
    int b = 0;
    for (int t = _loop_start; t < T; t += _loop_stride)
    {
        // We're calculating the gradient of the unpooled sequences, from
        // the gradient of the maxes. In this loop, we're getting the gradient
        // of a single sequence item, t. We need to know the sequence index,
        // b.
        while ((b < B) && (seq_start+lengths[b]) < t)
        {
           seq_start += lengths[b];
           b += 1;
        }
        // The "which" array tells us which rows were selected as the max.
        // So we need to find the index of our t in the sequence.
        int index_of_t = t-seq_start;
        // Get the rows we're dealing with, to avoid cluttering the loop
        // with the index math.
        float* dX_t = &dX[t*O];
        const float* d_maxes_b = &d_maxes[b*O];
        const int* which_b = &which[b*O];
        // Now loop over our row.
        for (int i=0; i < O; ++i)
        {
            // If we used the value for this cell,
            // pass the gradient
            if (which_b[i] == index_of_t)
               dX_t[i] = d_maxes_b[i];
        }
    }
}
