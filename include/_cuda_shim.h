
void gpu_maxout(float* best__bo, int* which__bo,
        const float* cands__bop, int B, int O, int P);

void gpu_mean_pool(float* means,
        const float* X, const int* lengths, int B, int T, int O);

void gpu_sum_pool(float* sums,
        const float* X, const int* lengths, int B, int T, int O);

void gpu_max_pool(float* maxes, int* which,
        const float* X, const int* lengths, int B, int T, int O);

void gpu_backprop_mean_pool(float* dX,
        const float* d_means, const int* lengths, int B, int T, int O);

void gpu_backprop_sum_pool(float* dX,
        const float* d_sum, const int* lengths, int B, int T, int O);

void gpu_backprop_max_pool(float* dX,
        const float* d_maxes, const int* which, const int* lengths, int B, int T, int O);

void gpu_hash_data(char* dest,
    const char* src, size_t out_size, size_t in_size, size_t n_items, uint32_t seed);

