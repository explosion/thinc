from libc.stdint cimport uint32_t, uint64_t

cimport numpy as np
import numpy as np


assert sizeof(int) == sizeof(np.int32_t)


cdef extern from "_cuda_shim.h":
    void gpu_maxout(float* best__bo, int* which__bo,
        const float* cands__bop, int B, int O, int P)
    void gpu_mean_pool(float* means,
        const float* X, const int* lengths, int B, int T, int O) nogil
    void gpu_sum_pool(float* sums,
        const float* X, const int* lengths, int B, int T, int O) nogil
    void gpu_max_pool(float* maxes, int* which,
        const float* X, const int* lengths, int B, int T, int O) nogil
    void gpu_backprop_mean_pool(float* dX,
        const float* d_means, const int* lengths, int B, int T, int O) nogil
    void gpu_backprop_sum_pool(float* dX,
        const float* d_sums, const int* lengths, int B, int T, int O) nogil
    void gpu_backprop_max_pool(float* dX,
        const float* d_maxes, const int* which, const int* lengths, int B, int T, int O) nogil
    void gpu_hash_data(char* dest,
        const char* src, size_t out_size, size_t in_size, size_t n_items, uint32_t seed) nogil


def maxout(*args, **kwargs):
    pass

def mean_pool(ops, x, lengths):
    means = ops.allocate((lengths.shape[0], x.shape[1]))
    cdef size_t means_ptr = means.data.ptr
    cdef size_t x_ptr = x.data.ptr
    cdef size_t lengths_ptr = lengths.data.ptr
    gpu_mean_pool(<float*>means_ptr,
        <const float*>x_ptr, <const int*>lengths_ptr, lengths.shape[0], x.shape[0], x.shape[1])
    return means


def max_pool(ops, x, lengths):
    maxes = ops.allocate((lengths.shape[0], x.shape[1]))
    which = ops.allocate((lengths.shape[0], x.shape[1]), dtype='i')
    cdef size_t maxes_ptr = maxes.data.ptr
    cdef size_t which_ptr = which.data.ptr
    cdef size_t x_ptr = x.data.ptr
    cdef size_t lengths_ptr = lengths.data.ptr
    gpu_max_pool(<float*>maxes_ptr, <int*>which_ptr,
        <const float*>x_ptr, <const int*>lengths_ptr, lengths.shape[0], x.shape[0], x.shape[1])
    return maxes, which


def sum_pool(ops, x, lengths):
    sums = ops.allocate((lengths.shape[0], x.shape[1]))
    cdef size_t sums_ptr = sums.data.ptr
    cdef size_t x_ptr = x.data.ptr
    cdef size_t lengths_ptr = lengths.data.ptr
    gpu_sum_pool(<float*>sums_ptr,
        <const float*>x_ptr, <const int*>lengths_ptr, lengths.shape[0], x.shape[0], x.shape[1])
    return sums


def backprop_mean_pool(ops, d_means, lengths):
    dX = ops.allocate((lengths.sum().get(), d_means.shape[1]))
    cdef size_t d_means_ptr = d_means.data.ptr
    cdef size_t dX_ptr = dX.data.ptr
    cdef size_t lengths_ptr = lengths.data.ptr
    cdef int B = lengths.shape[0]
    cdef int T = dX.shape[0]
    cdef int O = dX.shape[1]

    gpu_backprop_mean_pool(<float*>dX_ptr,
        <const float*>d_means_ptr, <const int*>lengths_ptr,
        B, T, O)
    return dX


def backprop_sum_pool(ops, d_sums, lengths):
    dX = ops.allocate((lengths.sum().get(), d_sums.shape[1]))
    cdef size_t d_sums_ptr = d_sums.data.ptr
    cdef size_t dX_ptr = dX.data.ptr
    cdef size_t lengths_ptr = lengths.data.ptr
    cdef int B = lengths.shape[0]
    cdef int T = dX.shape[0]
    cdef int O = dX.shape[1]

    gpu_backprop_sum_pool(<float*>dX_ptr,
        <const float*>d_sums_ptr, <const int*>lengths_ptr,
        B, T, O)
    return dX

def backprop_max_pool(ops, d_maxes, which, lengths):
    dX = ops.allocate((lengths.sum().get(), d_maxes.shape[1]))
    cdef size_t d_maxes_ptr = d_maxes.data.ptr
    cdef size_t which_ptr = which.data.ptr
    cdef size_t dX_ptr = dX.data.ptr
    cdef size_t lengths_ptr = lengths.data.ptr
    cdef int B = lengths.shape[0]
    cdef int T = dX.shape[0]
    cdef int O = dX.shape[1]

    gpu_backprop_max_pool(<float*>dX_ptr,
        <const float*>d_maxes_ptr, <const int*>which_ptr, <const int*>lengths_ptr,
        B, T, O)
    return dX


def hash(ops, ids, seed):
    keys = ops.allocate((ids.shape[0], 4), dtype='uint32')
    cdef size_t keys_ptr = keys.data.ptr
    cdef size_t ids_ptr = ids.data.ptr

    gpu_hash_data(<char*>keys_ptr,
        <const char*>ids_ptr, sizeof(uint32_t)*4, sizeof(uint64_t), ids.shape[0], seed)
    return keys
