# cython: infer_types=True
# cython: cdivision=True

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt
cimport cython.parallel

from thinc.neural._classes.maxout import Maxout

cimport numpy as np

from tokyo.tokyo cimport sgemm_, saxpy_
from tokyo.tokyo cimport *

from ..api import wrap

ctypedef float real_t
ctypedef int dim_t
ctypedef unsigned char uchar

cdef extern from "stdlib.h":
    void* aligned_alloc(size_t alignment, size_t size) nogil


def MaxoutWindowEncode(maxout, n_iter):
    ops = maxout.ops
    def mwe_fwd(Xs, drop=0.):
        cdef:
            dim_t i, nO, nI, nP, N
            np.ndarray weights, bias, inputs, outputs
        weights = maxout.W
        bias = maxout.b
        nO = maxout.nO
        nI = maxout.nI
        nP = maxout.nP

        inputs = ops.flatten(Xs)
        N = inputs.shape[0]
        outputs = ops.allocate((N, nO))

        # Buffers and return state
        prev = <real_t*>calloc(nO*(N+2), sizeof(real_t))
        tmp = <real_t*>calloc(N*nO*3+N*nO*nP, sizeof(real_t))
        state_for_bwd = NULL
        
        # Now do the actual work
        maxout_window_encode(<real_t*>outputs.data, prev, tmp,
            <real_t*>inputs.data, <real_t*>weights.data, <real_t*>bias.data,
            N, nO, nP, n_iter, state_for_bwd)
        
        free(prev)
        free(tmp)
        lengths = [len(X) for X in Xs]

        Ys = ops.unflatten(outputs, lengths)
        def mwe_bwd(dYs, sgd=None):
            dY = ops.flatten(dYs)
            d_maxout_window_encode(NULL, NULL, NULL, NULL,
                state_for_bwd, N, nO, nP, n_iter)
        return Ys, mwe_bwd
    return wrap(mwe_fwd, maxout)


cdef void maxout_window_encode(real_t* outputs, real_t* prev, real_t* tmp,
        const real_t* inputs, const real_t* weights, const real_t* bias,
        dim_t nr_row, dim_t nr_dim, dim_t nr_piece, int nr_iter,
        void* state_for_bwd) nogil:
    N = nr_dim * nr_row
    memcpy(&prev[nr_dim], inputs, sizeof(prev[0]) * N)
    if tmp is not NULL:
        X = tmp
        Xh = &tmp[N*3]
    else:
        X = <float*>calloc(N*3, sizeof(float))
        Xh = <float*>calloc(N*nr_piece, sizeof(float))

    for i in range(nr_iter):
        cnn_maxout(outputs, NULL, X, Xh,
            prev, weights, bias, nr_row, nr_dim, nr_dim, nr_piece)
        layer_norm(outputs, nr_row, nr_dim)
        residual(outputs, prev, N)
        memcpy(&prev[nr_dim], outputs, sizeof(prev[0]) * N)
    if tmp is NULL:
        free(X)
        free(Xh)


cdef void d_maxout_window_encode(real_t* dX, real_t* dW, real_t* db,
        real_t* tmp, void* state_from_fwd, dim_t N, dim_t nO, dim_t nP,
        dim_t n_iter) nogil:
    pass


cdef void cnn_maxout(float* best, unsigned char* which, float* X, float* Xh,
        const float* words, const float* weights, const float* bias,
        dim_t nW, dim_t nI, dim_t nO, dim_t nP) nogil:
    for w in range(nW):
        # Assume correct padding on words. This means words should
        # start with an eol row, pushing them off-alignment with
        # the outputs (best and which). Words also needs to end with
        # an eol row
        memcpy(&X[w*nI*3], &words[w*nI], 3*nI*sizeof(float))
    memset(Xh, 0, nW*nO*nP*sizeof(float))
    affine(Xh,
        X, weights, bias, nW, nO*nP, nI*3)
    cdef unsigned char j
    for w in range(nW):
        for i in range(nO):
            j = argmax(&Xh[w*nO*nP+i*nP], nP)
            best[w*nO+i] = Xh[w*nO*nP+i*nP+j]
            if which is not NULL:
                which[w*nO+i] = j


cdef void affine(float* outputs,
        const float* inputs, const float* weights, const float* bias,
        dim_t nB, dim_t nO, dim_t nI) nogil:
    sgemm_(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        nB,
        nO,
        nI,
        1.0,
        <float*>inputs,
        nI,
        <float*>weights,
        nI,
        1.0,
        outputs,
        nO)
    cdef dim_t _, j
    for i in range(nB):
        for j in range(nO):
            outputs[j] += bias[j]
        outputs += nO


cdef int argmax(const float* X, int n) nogil:
    cdef int m = 0
    cdef float best = X[0]
    for i in range(1, n):
        x = X[i]
        if x > best:
            m = i
            best = x
    return m


cdef void residual(real_t* output, const real_t* prev, int N) nogil:
    for i in range(N):
        output[i] += prev[i]


cdef void layer_norm(real_t* X, dim_t nr_dim, dim_t nr_row) nogil:
    for i in range(nr_row):
        mu = mean(X, nr_dim)
        v = variance(X, mu, nr_dim)
        for j in range(nr_dim):
            X[j] = sqrt((X[j] - mu) * v)
        X += nr_dim


cdef real_t mean(const real_t* X, int nr_dim) nogil:
    cdef real_t mean = 0.
    for x in X[:nr_dim]:
        mean += x
    return mean / nr_dim


cdef real_t variance(const real_t* X, real_t mean, int nr_dim) nogil:
    cdef double sum_ = 0.
    cdef double sum2 = 0.
    for x in X[:nr_dim]:
        diff = x-mean
        sum_ += diff
        sum2 += diff * diff
    v = ((sum2 - sum_*sum_) / nr_dim) / nr_dim
    return v + 1e-8


