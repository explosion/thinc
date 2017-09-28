# cython: infer_types=True
# cython: cdivision=True

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt
cimport cython.parallel

from ._classes.maxout import Maxout
from ..linalg cimport Vec, Mat, VecVec, MatVec, MatMat
from .ops cimport cpu_backprop_maxout as bwd_maxout
from .ops cimport backprop_seq2col as bwd_seq2col

cimport numpy as np

from tokyo.tokyo cimport sgemm_, saxpy_
from tokyo.tokyo cimport *
from blis import blis

from ..api import wrap

ctypedef float real_t
ctypedef int dim_t

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
        state_for_bwd = NULL
        
        # Now do the actual work
        maxout_window_encode(<real_t*>outputs.data,
            <real_t*>inputs.data, <real_t*>weights.data, <real_t*>bias.data,
            N, nO, nP, n_iter, <unsigned char*>state_for_bwd)
        
        lengths = [len(X) for X in Xs]

        Ys = ops.unflatten(outputs, lengths)
        def mwe_bwd(dYs, sgd=None):
            dY = ops.flatten(dYs)
            #bwd_maxout_window_encode(dX, dW, db,
            #    which, X, Xh, W, N, nO, nP, n_iter)
        return Ys, mwe_bwd
    return wrap(mwe_fwd, maxout)


cdef void maxout_window_encode(real_t* outputs, 
        const real_t* inputs, const real_t* weights, const real_t* bias,
        dim_t nr_row, dim_t nr_dim, dim_t nr_piece, int nr_iter,
        unsigned char* state_for_bwd) nogil:
    cdef int* which
    N = nr_dim * nr_row
    prev = <float*>calloc((nr_row+2)*nr_dim, sizeof(float))
    # If memory provided, use it (and keep the values in it)
    # Otherwise, allocate from heap.
    if state_for_bwd is NULL:
        which = NULL
        X = <float*>calloc(N*3, sizeof(float))
        Xh = <float*>calloc(N*nr_piece, sizeof(float))
    else:
        pass
        #which = state_for_bwd
        #state_for_bwd += N
        #X = <float*>state_for_bwd
        #state_for_bwd += N * 3 * sizeof(float)
        #Xh = <float*>state_for_bwd
        #state_for_bwd += N * nr_piece * sizeof(float)
    memcpy(&prev[nr_dim], inputs, sizeof(prev[0]) * N)
    for i in range(nr_iter):
        cnn_maxout(outputs, which, X, Xh,
            prev, weights, bias, nr_row, nr_dim, nr_dim, nr_piece)
        layer_norm(outputs, nr_row, nr_dim)
        residual(outputs, prev, N)
        memcpy(&prev[nr_dim], outputs, sizeof(prev[0]) * N)
        #if state_for_bwd is not NULL:
        #    X = <float*>state_for_bwd
        #    state_for_bwd += N * 3 * sizeof(float)
        #    Xh = <float*>state_for_bwd
        #    state_for_bwd += N * nr_piece * sizeof(float)
    if state_for_bwd is NULL:
        free(X)
        free(Xh)


cdef void bwd_maxout_window_encode(real_t* dX, real_t* dW, real_t* db,
        const int* which, const real_t* X, const real_t* Xh,
        dim_t N, dim_t nO, dim_t nP, dim_t nr_iter) nogil:
    '''
    The function in the inner loop is:

    Given x1:
      x2, bp_x2 = window(x1)
      x3, bp_x3 = affine(x2)
      x4, bp_x4 = maxpool(x3)
      x5, bp_x5 = layernorm(x4)
      x6, bp_x6 = rescale(x5)
      x7 = x1 + x5
    return x7, lambda dx7: dx7 + bp_x2(bp_x3(bp_x4(bp_x5(bp_x6(dx7)))))

    In the backward pass we must compute:

    Given dx7:
      dx7 = dx6
      dx5 = backprop_rescale(dx6)
      dx4 = backprop_layernorm(dx5)
      dx3 = backprop_maxpool(dx4)
      dx2 = backprop_affine(dx3)
      dx1 = backprop_window(dx2)
    Return dx7+dx1

    The functions (window, affine, maxpool) are grouped, for optimization.
    '''
    cdef float *x1, *x2, *x3, *x4, *x5, *x6
    cdef float *dx7, *dx6, *dx5, *dx4, *dx3, *dx2, *dx1
    cdef float *w_rescale_G, *w_rescale_b, *w_maxout_W, *w_maxout_b
    cdef float *dw_rescale_G, *dw_rescale_b, *dw_maxout_W, *dw_maxout_b
    cdef int* maxout_mask
    for i in range(nr_iter-1, -1, -1):
        bwd_rescale(dx5, dw_rescale_G, dw_rescale_b,
            dx6, x6, w_rescale_G, nO, N)
        bwd_layer_norm(dx4,
            dx5, x4, nO, N)
        bwd_cnn_maxout(dx1, dw_maxout_W, dw_maxout_b,
            dx4, x1, maxout_mask, w_maxout_W, nO, nP, N)
        VecVec.add_i(dx1,
            dx7, 1., N * nO)


cdef void cnn_maxout(float* best, int* which, float* X, float* Xh,
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
    cdef int j
    for w in range(nW):
        for i in range(nO):
            j = argmax(&Xh[w*nO*nP+i*nP], nP)
            best[w*nO+i] = Xh[w*nO*nP+i*nP+j]
            if which is not NULL:
                which[w*nO+i] = j


cdef void bwd_cnn_maxout(real_t* dX, real_t* dW, real_t* db,
        const real_t* dXh, const real_t* X, const int* which, const real_t* W,
        dim_t nO, dim_t nP, dim_t N) nogil:
    bwd_maxout(dX,
        dXh, which, N, nO, nP)
    bwd_affine(dX, dW, db,
        dXh, X, W, nO*nP, nO, N)
    bwd_seq2col(dX,
        dX, N, nO, 1)


cdef void affine(float* outputs,
        const float* inputs, const float* weights, const float* bias,
        dim_t nB, dim_t nO, dim_t nI) nogil:
    MatVec.batch_dot(outputs,
        weights, inputs, nO, nI, nB)
    for i in range(nB):
        for j in range(nO):
            outputs[j] += bias[j]
        outputs += nO


cdef void bwd_affine(float* d_inputs, float* dW, float* db,
        const float* d_outputs, const float* inputs, const float* weights,
        dim_t nO, dim_t nI, dim_t nB) nogil:
    for i in range(nB):
        VecVec.add_i(db,
            &d_outputs[i*nO], 1., nO)
    MatMat.batch_add_outer_i(dW,
        d_outputs, inputs, nO, nI, nB)
    MatVec.batch_T_dot(d_inputs,
        weights, d_outputs, nO, nI, nB)


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
        mu = Vec.mean(X, nr_dim)
        v = Vec.variance(X, mu, nr_dim)
        for j in range(nr_dim):
            X[j] = sqrt((X[j] - mu) * v)
        X += nr_dim


cdef void bwd_layer_norm(real_t* dX,
        const real_t* dXh, const real_t* X, dim_t nr_dim, dim_t nr_row) nogil:
    for i in range(nr_row):
        mu = Vec.mean(X, nr_dim)
        v = Vec.variance(X, mu, nr_dim)
        sqrt_var = sqrt(v)
        inv_var = 1. / v
        sum_dXh = Vec.sum(dX, nr_dim)
        sum_dXh_dist = 0.
        for j in range(nr_dim):
            sum_dXh_dist += dXh[j] * (X[j]-mu)
        for j in range(nr_dim):
            dX[j] = nr_dim * dXh[j]
            dX[j] -= sum_dXh
            dX[j] -= (X[j]-mu) * inv_var * sum_dXh_dist
            dX[j] *= -sqrt_var
            dX[j] /= nr_dim
        X += nr_dim
        dX += nr_dim
        dXh += nr_dim
 

cdef void rescale(float* X,
        const float* G, const float* bias, int nO, int nB) nogil:
    for i in range(nB):
        VecVec.mul_i(X,
            G, nO)
        VecVec.add_i(X,
            bias, 1., nO)
        X += nO


cdef void bwd_rescale(float* d_down, float* dG, float* d_bias,
        const float* d_up, const float* down, const float* G, int nO, int nB) nogil:
    memcpy(d_down,
        d_up, nO*nB*sizeof(d_down[0]))
    for i in range(nB):
        VecVec.mul_i(d_down,
            G, nO)
        d_down += nO
    for i in range(nB):
        VecVec.add_i(d_bias,
            d_up, 1., nO)
    MatMat.batch_add_outer_i(dG,
        d_up, down, nO, 1, nB)
