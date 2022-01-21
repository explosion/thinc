// Use grid strided loops, described here:
// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
// This pattern ensures that all of the loop values are visited once, no matter
// what grid parameters are used for the function.

extern "C" __global__
void seq2col(float* output, const float* X, const int* lengths,
        int nW, int B, int I, int nL)
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
    //
    // If lengths > 1, then the sequence lengths dictate
    // the boundaries/padding rather than the begin/end
    // of X.
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    int nF = nW * 2 + 1;

    int seq = 0;
    int seq_start = 0;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
        // Find sequence sequence in which b lies.
        for (; seq < nL; ++seq) {
            if (b < seq_start + lengths[seq]) {
                break;
            }
            seq_start += lengths[seq];
        }

        // Calculate the bounds of the sequence wherein b lies.
        int seq_end = seq_start + lengths[seq];

        // Find the unconstrained window around b, which
        // may be out of the sequence bounds.
        int window_start = b - nW;
        int window_end = b + nW + 1;

        // Find the sequence-constrained window around b.
        int x_start = max(seq_start, window_start);
        int x_end = min(seq_end, window_end);
        int n_elems = x_end - x_start;

        // If the left window is cut short, we want to start by
        // the same amount in the output.
        int out_offset = x_start - window_start;

        for (int i = 0; i < n_elems * I; i++) {
            output[(b * I * nF) + (out_offset * I) + i] =
                X[(x_start * I) + i];
        }
    }
}


extern "C" __global__
void maxout(float* best, int* which,
        const float* cands, int B, int O, int P)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    for (int bo = _loop_start; bo < B * O; bo += _loop_stride)
    {
        // Go to the candidates at the output we're working on
        const float* cands_bo = &cands[bo * P];

        int best_idx = 0;
        float best_val = cands_bo[0];
        for (int p = 1; p < P; ++p)
        {
            if (cands_bo[p] > best_val) {
                best_idx = p;
                best_val = cands_bo[p];
            }
        }

        which[bo] = best_idx;
        best[bo] = best_val;
    }
}

extern "C" __global__
void clipped_linear(float* Y, const float* X, double slope, double offset, double min_val, double max_val, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        float y = X[i] * slope + offset;
        Y[i] = min(max(y, min_val), max_val);
    }
}

extern "C" __global__
void gelu(float* Y, const float* X, double threshold, int N)
{
    const float INV_SQRT_2 = 0.7071067811865475;

    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        float x = X[i];
        if (x >= threshold) {
            Y[i] = x;
        } else if (x <= -threshold) {
            Y[i] = 0.0;
        } else {
            float cdf = 0.5 * (1.0 + erff(INV_SQRT_2 * x));
            Y[i] = x * cdf;
        }
    }
}

extern "C" __global__
void mish(float* Y, const float* X, double threshold, int N)
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
void swish(float* Y, const float* X, double threshold, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        if (X[i] >= threshold) {
            Y[i] = X[i];
        } else if (X[i] <= -threshold) {
            Y[i] = 0.0;
        } else {
            float logistic_cdf = 1.0 / (1.0 + expf(-X[i]));
            Y[i] = X[i] * logistic_cdf;
        }
    }
}

extern "C" __global__
void reduce_sum(float* output,
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
void reduce_max(float* maxes, int* which,
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
void backprop_seq2col(float* d_seqs, const float* d_cols, const int* lengths,
        int nW, int B, int I, int nL)
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
    int seq = 0;
    int seq_start = 0;
    for (int b = _loop_start; b < B; b += _loop_stride)
    {
        // Find sequence offset in which b lies.
        // Fixme: do not restart offset search for every b.
        for (; seq < nL; ++seq) {
            if (b < seq_start + lengths[seq]) {
                break;
            }
            seq_start += lengths[seq];
        }

        // Calculate the bounds of the sequence wherein b lies.
        int seq_end = seq_start + lengths[seq];

        // Find the unconstrained window around b, which
        // may be out of the sequence bounds.
        int window_start = b - nW;
        int window_end = b + nW + 1;

        // Find the sequence-constrained window around b.
        int d_seqs_start = max(seq_start, window_start);
        int d_seqs_end = min(seq_end, window_end);


        // The here update proceeds differently than the other seq2col
        // implementations. We have to do all the updates for the b in this loop
        // iteration, otherwise we get data races due to parallelism in CUDA.
        //
        // A batch item b occurs, given nw=1, in:
        //
        // position 0 in b - 1 (if present) <- window_start
        // position 1 in b
        // position 2 in b + 1 (if present) <- window_end
        //
        // The following loop sums the gradients for those occurrences.
        // b_w loops over [b - 1, b, b + 1] and computes the position
        // of b within the column gradients of [b - 1 ... b + 1].
        for (int b_w = d_seqs_start; b_w < d_seqs_end; ++b_w) {
            int position = (2 * nW) - (b_w - window_start);
            int start = (b_w * I * nF) + (position * I);
            for (int i = 0; i < I; ++i) {
                d_seqs[(b*I + i)] += d_cols[start + i];
            }
        }
    }
}

extern "C" __global__
void backprop_clipped_linear(float* dX, const float* dY, const float* X, double slope, double offset, double min_val, double max_val, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    float low = (min_val - offset) / slope;
    float high = (max_val - offset) / slope;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        float x = X[i];

        if (low < x && x < high) {
            dX[i] = dY[i] * slope;
        } else {
            dX[i] = 0;
        }
    }
}

extern "C" __global__
void backprop_hard_swish(float* dX, const float* dY, const float* X, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        if (X[i] > 2.5) {
            dX[i] = dY[i];
        } else if (X[i] < -2.5) {
            dX[i] = 0;
        } else {
            dX[i] = dY[i] * (X[i] * 0.4 + 0.5);
        }
    }
}

extern "C" __global__
void backprop_hard_swish_mobilenet(float* dX, const float* dY, const float* X, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        if (X[i] > 3.0) {
            dX[i] = dY[i];
        } else if (X[i] < -3.0) {
            dX[i] = 0;
        } else {
            dX[i] = dY[i] * ((X[i] * 2.0 + 3.0) / 6.0);
        }
    }
}

extern "C" __global__
void backprop_gelu(float* dX, const float* dY, const float* X,
    double threshold, int N)
{
    const float INV_SQRT_2PI = 0.3989422804014327;
    const float INV_SQRT_2 = 0.7071067811865475;

    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        float x = X[i];

        if (x >= threshold) {
            dX[i] = dY[i];
        } else if (x <= -threshold) {
            dX[i] = 0.0;
        } else {
            float cdf = 0.5 * (1.0 + erff(INV_SQRT_2 * x));
            float pdf = INV_SQRT_2PI * expf(-0.5 * x * x);
            dX[i] = dY[i] * (cdf + x * pdf);
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
    const float* dY, const float* X, double threshold, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
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
            dX[i] = dY[i] * ((exp_x * omega) / (delta * delta));
        }
    }
}


extern "C" __global__
void backprop_swish(float* dX, const float* dY, const float* X,
    const float* Y, double threshold, int N)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;

    for (int i = _loop_start; i < N; i += _loop_stride)
    {
        float x = X[i];
        float y = Y[i];

        if (x >= threshold) {
            dX[i] = dY[i];
        } else if (x <= -threshold) {
            dX[i] = 0.0;
        } else {
            float cdf = 1.0 / (1 + exp(-x));
            float d = y + cdf * (1 - y);
            dX[i] = dY[i] * d;
        }
    }
}

extern "C" __global__
void backprop_reduce_sum(float* dX, const float* d_sum, const int* lengths,
    int B, int T, int O)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int seq_start = 0;
    int b = 0;
    for (int t = _loop_start; t < T; t += _loop_stride)
    {
        // Find the sequence item we're working on
        while ((b < B) && (seq_start+lengths[b]) <= t)
        {
           seq_start += lengths[b];
           b += 1;
        }
        if (lengths[b] == 0)
            continue;

        for (int i=0; i < O; ++i)
        {
            dX[t * O + i] = d_sum[b * O + i];
        }
    }
}


extern "C" __global__
void backprop_reduce_mean(float* dX, const float* d_mean, const int* lengths,
    int B, int T, int O)
{
    int _loop_start = blockIdx.x * blockDim.x + threadIdx.x;
    int _loop_stride = blockDim.x * gridDim.x;
    int seq_start = 0;
    int b = 0;
    for (int t = _loop_start; t < T; t += _loop_stride)
    {
        // Find the sequence item we're working on
        while ((b < B) && (seq_start+lengths[b]) <= t)
        {
           seq_start += lengths[b];
           b += 1;
        }
        if (lengths[b] == 0)
            continue;

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
void backprop_reduce_max(float* dX,
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
        while ((b < B) && (seq_start+lengths[b]) <= t)
        {
           seq_start += lengths[b];
           b += 1;
        }
        if (lengths[b] == 0)
            continue;

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
