#include <_kernels.cu>
#include <_murmur3.cu>


void
gpu_maxout(float* best, int* which,
        const float* cands, int B, int O, int P)
{
    maxout<<<B/16,16>>>(best, which, cands, B, O, P);
}


void
gpu_mean_pool(float* means,
    const float* X, const int* lengths, int B, int T, int O)
{
  mean_pool<<<B, 1>>>(means, X, lengths, B, T, O);
}


void
gpu_max_pool(float* maxes, int* which,
    const float* X, const int* lengths, int B, int T, int O)
{
  max_pool<<<B, 1>>>(maxes, which, X, lengths, B, T, O);
}


void
gpu_sum_pool(float* sums, 
    const float* X, const int* lengths, int B, int T, int O)
{
  sum_pool<<<B, 1>>>(sums, X, lengths, B, T, O);
}


void
gpu_backprop_mean_pool(float* dX, const float* d_means, const int* lengths, int B, int T, int O)
{
  backprop_mean_pool<<<B, 1>>>(dX, d_means, lengths, B, T, O);
}


void
gpu_backprop_sum_pool(float* dX, const float* d_sums, const int* lengths, int B, int T, int O)
{
  backprop_sum_pool<<<B, 1>>>(dX, d_sums, lengths, B, T, O);
}


void
gpu_backprop_max_pool(float* dX, const float* d_maxes, const int* which,
		      const int* lengths, int B, int T, int O)
{
  backprop_max_pool<<<B, 1>>>(dX, d_maxes, which, lengths, B, T, O);
}

void
gpu_hash_data(char* dest,
    const char* src, size_t out_size, size_t in_size, size_t n_items, uint32_t seed)
{
    hash_data<<<n_items,1>>>(dest, src, out_size, in_size, n_items, seed);
}

