# cython: infer_types=True
# cython: cdivision=True

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt
cimport cython.parallel
from cymem.cymem cimport Pool
from libcpp.vector cimport vector

from ._classes.maxout import Maxout
from ._classes.layernorm import LayerNorm
from ..linalg cimport Vec, Mat, VecVec, MatVec, MatMat

cimport numpy as np

from blis import blis

from ..api import wrap

ctypedef float real_t
ctypedef int dim_t


cdef extern from "stdlib.h":
    void* aligned_alloc(size_t alignment, size_t size) nogil


cdef class _Activations:
    cdef float** a
    cdef float** b
    cdef float** c
    cdef float** d
    cdef float** e
    cdef float** f
    cdef int** which
    cdef int* lengths
    cdef vector[void*] pointers
    cdef int allocated

    def __init__(self, Xs, dim_t nP, dim_t nr_iter):
        # Allocate buffers
        # Total e.g. nO=128, nP=3, nN=1000
        #   128*3*1000
        # + 128*3*1000
        # + 128*1000
        # + 128*1000
        # + 128*1000
        # = 384000 * 3 * 4 iterations
        # = approx 2mb
        nX = len(Xs)
        self.a = <float**>calloc(nX, sizeof(float*))
        self.b = <float**>calloc(nX, sizeof(float*))
        self.c = <float**>calloc(nX, sizeof(float*))
        self.d = <float**>calloc(nX, sizeof(float*))
        self.e = <float**>calloc(nX, sizeof(float*))
        self.f = <float**>calloc(nX, sizeof(float*))
        self.which = <int**>calloc(nX, sizeof(int*))
        self.lengths = <int*>calloc(nX, sizeof(int))
        cdef np.ndarray X
        cdef dim_t nN
        cdef dim_t nO = Xs[0].shape[1]
        for i, X in enumerate(Xs):
            nN = X.shape[0]
            self.a[i] = <float*>calloc(nO*nN*nr_iter, sizeof(float))
            self.b[i] = <float*>calloc(nO*3*nN*nr_iter, sizeof(float))
            self.c[i] = <float*>calloc(nO*nP*nN*nr_iter, sizeof(float))
            self.d[i] = <float*>calloc(nO*nN*nr_iter, sizeof(float))
            self.e[i] = <float*>calloc(nO*nN*nr_iter, sizeof(float))
            self.f[i] = <float*>calloc(nO*nN*nr_iter, sizeof(float))
            self.which[i] = <int*>calloc(nO*nN*nr_iter, sizeof(int))
            self.lengths[i] = nN

            memcpy(self.a[i], <float*>X.data, nO*nN*sizeof(float))

            self.pointers.push_back(self.a[i])
            self.pointers.push_back(self.b[i])
            self.pointers.push_back(self.c[i])
            self.pointers.push_back(self.d[i])
            self.pointers.push_back(self.e[i])
            self.pointers.push_back(self.f[i])
            self.pointers.push_back(self.which[i])
        self.pointers.push_back(self.a)
        self.pointers.push_back(self.b)
        self.pointers.push_back(self.c)
        self.pointers.push_back(self.d)
        self.pointers.push_back(self.e)
        self.pointers.push_back(self.f)
        self.pointers.push_back(self.which)
        self.pointers.push_back(self.lengths)
        self.allocated = True

    def __dealloc__(self):
        if self.allocated:
            for i in range(self.pointers.size()):
                free(self.pointers[i])
            self.allocated = False
 

cdef class _Weights:
    cdef float* syn
    cdef float* bias
    cdef float* scale
    cdef float* shift
    def __init__(self, np.ndarray syn, np.ndarray bias,
                 np.ndarray scale, np.ndarray shift):
        self.syn = <float*>syn.data
        self.bias = <float*>bias.data
        self.scale = <float*>scale.data
        self.shift = <float*>shift.data


def MaxoutWindowEncoder(nr_unit, nr_iter):
    maxout = Maxout(nr_unit, nr_unit*3, pieces=3)
    normalize = LayerNorm(maxout)

    def mwe_fwd(Xs, drop=0.):
        return _mwe_fwd(nr_iter, maxout, normalize, Xs, drop=drop)

    model = wrap(mwe_fwd, normalize)
    model.maxout = maxout
    model.normalize = normalize
    return model


def _mwe_fwd(dim_t nr_iter, maxout, normalize, inputs, drop=0.):
    '''
    The function in the inner loop is:

    Given a:
    b, bp_b = window(a)
    c, bp_c = affine(b)
    d, bp_d = maxpool(c)
    e, bp_e = layernorm(d)
    f, bp_f = rescale(e)
    g = a + f
    return g, lambda dg: dg+bp_f(bp_e(bp_d(bp_c(bp_b(dg)))))

    In the backward pass we must compute:

    Given dg:
    df = dg 
    de = backprop_rescale(de)
    dd = backprop_layernorm(de)
    dc = backprop_maxpool(dd)
    db = backprop_affine(dc)
    da = backprop_window(db)
    Return dg+da
    '''
    cdef dim_t nO = maxout.nO
    cdef dim_t nP = maxout.nP
    cdef _Weights W = _Weights(maxout.W, maxout.b,
                               normalize.G, normalize.b)
    cdef _Activations X = _Activations(inputs, nP, nr_iter)

    cdef dim_t nX = len(inputs)
    cdef dim_t i
    for i in cython.parallel.prange(nX, nogil=True, num_threads=4):
        mwe_forward(X.a[i], X.b[i], X.c[i], X.d[i], X.e[i], X.f[i], X.which[i],
            W.syn, W.bias, W.scale, W.shift, nO, nP, X.lengths[i], nr_iter)

    outputs = []
    cdef np.ndarray Y
    for i in range(nX):
        Y = maxout.ops.allocate((X.lengths[i], nO))
        memcpy(<float*>Y.data,
            X.a[i], nO*X.lengths[i]*sizeof(float))
        outputs.append(Y)

    nonlocals = {}
    nonlocals['X'] = X

    #def mwe_bwd(d_output_seqs, sgd=None):
    #    ops = maxout.ops
    #    cdef np.ndarray d_outputs = ops.flatten(d_output_seqs)
    #    cdef _Weights W = _Weights(maxout.W, maxout.b,
    #                               normalize.G, normalize.b)
    #    cdef _Activations X = nonlocals['X']
    #    # Gradients
    #    cdef _Activations dX = _Activations(nO, nP, nN, 1)
    #    cdef _Weights dW = _Weights(maxout.d_W, maxout.d_b,
    #                                normalize.d_G, normalize.d_b)
    #    memcpy(dX.f, <float*>d_outputs.data, nO*nN*sizeof(float))
    #    cdef dim_t i
    #    for i in range(nr_iter-1, -1, -1):
    #        X.which -= nO*nN
    #        X.b -= nO*3*nN
    #        X.d -= nO*nN
    #        X.e -= nO*nN
    #        memset(dX.e, 0, nO*nN*sizeof(float))
    #        bwd_rescale(dX.e, dW.scale, dW.shift,
    #            dX.f, X.e, W.scale, nO, nN)
    #        memset(dX.d, 0, nO*nN*sizeof(float))
    #        bwd_layer_norm(dX.d,
    #            dX.e, X.d, nO, nN)

    #        memset(dX.c, 0, nO*nP*nN*sizeof(float))
    #        bwd_maxpool(dX.c,
    #            dX.d, X.which, nO, nP, nN)

    #        memset(dX.b, 0, nO*3*nN*sizeof(float))
    #        bwd_affine(dX.b, dW.syn, dW.bias,
    #            dX.c, X.b, W.syn, nO*nP, nO*3, nN) 

    #        memset(dX.a, 0, nO*nN*sizeof(float))
    #        bwd_seq2col(dX.a, 
    #            dX.b, 1, nO, nN) 
    #        VecVec.add_i(dX.a,
    #            dX.f, 1., nN * nO)
    #        memcpy(dX.f, dX.a, nO*nN*sizeof(float))
    #    cdef np.ndarray d_inputs = ops.allocate((nN, nO))
    #    memcpy(<float*>d_inputs.data,
    #        dX.a, nN*nO*sizeof(float))
    #    if sgd is not None:
    #        sgd(maxout._mem.weights, maxout._mem.gradient, key=maxout.id)
    #        sgd(normalize._mem.weights, normalize._mem.gradient, key=normalize.id)
    #    return ops.unflatten(d_inputs, lengths)
    return outputs, None #mwe_bwd

cdef void mwe_forward(float* Xa, float* Xb, float* Xc, float* Xd, float* Xe, float* Xf, int* which,
        const float* syn, const float* bias, const float* scale, const float* shift,
        dim_t nO, dim_t nP, dim_t nN, dim_t nr_iter) nogil:
    # Now do the actual work
    for i in range(nr_iter):
        seq2col(Xb,
            Xa, 1, nO, nN)
        affine(Xc,
            Xb, syn, bias, nO*nP, nO*3, nN) 
        maxpool(Xd, which,
            Xc, nO, nP, nN)
        memcpy(Xe,
            Xd, nO*nN*sizeof(Xe[0]))
        layer_norm(Xe,
            nO, nN)
        memcpy(Xf,
            Xe, nO*nN*sizeof(float))
        rescale(Xf,
            scale, shift, nO, nN)
        VecVec.add_i(Xa,
            Xf, 1., nO*nN)
        Xb += nO*nP*nN
        memset(Xc, 0, nO*3*nN*sizeof(float))
        Xd += nO*nN
        Xe += nO*nN
        which += nO*nN


cdef void seq2col(float* Xb,
        const float* Xa, dim_t nW, dim_t nI, dim_t nN) nogil:
    cdef dim_t nF = nW * 2 + 1
    Xb += nW * nI
    cdef dim_t i
    for i in range(nN-nW):
        memcpy(Xb,
            Xa, nI * (nW+1) * sizeof(Xb[0]))
        Xb += nI * (nW+1)
        memcpy(Xb,
            Xa, nI * nW * sizeof(Xb[0]))
        Xb += nI * nW
        Xa += nI
    memcpy(Xb,
        Xa, nI * nW * sizeof(Xb[0]))


cdef void bwd_seq2col(float* dXa,
        const float* dXb, dim_t nW, dim_t nI, dim_t nN) nogil:
    # Here's what we're doing, if we had 2d indexing.
    #for i in range(B):
    #    d_seq[i] += d_cols[i-2, 4]
    #    d_seq[i] += d_cols[i-1, 3]
    #    d_seq[i] += d_cols[i+2, 0]
    #    d_seq[i] += d_cols[i+1, 1]
    #    d_seq[i] += d_cols[i, 2]
    nF = nW * 2 + 1
    for i in range(nN):
        seq_row = &dXa[i * nI]
        col_row = &dXb[i * nI * nF]
        for f in range(-nW, nW+1):
            if nN > (i+f) >= 0:
                feat = col_row + (f * nI)
                VecVec.add_i(seq_row, &feat[(f+nW) * nI], 1., nI)


cdef void affine(float* outputs,
        const float* inputs, const float* weights, const float* bias,
        dim_t nO, dim_t nI, dim_t nB) nogil:
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


cdef void maxpool(float* Xb, int* which,
        const float* Xa, dim_t nO, dim_t nP, dim_t nN) nogil:
    cdef int j
    for w in range(nN):
        for i in range(nO):
            j = Vec.arg_max(&Xa[w*nO*nP+i*nP], nP)
            Xb[w*nO+i] = Xa[w*nO*nP+i*nP+j]
            if which is not NULL:
                which[w*nO+i] = j


cdef void bwd_maxpool(float* dXa,
        const float* dXb, const int* which, dim_t nO, dim_t nP, dim_t nN) nogil:
    for b in range(nN):
        for o in range(nO):
            dXa[which[0]] = dXb[0]
            dXa += nP
            dXb += 1
            which += 1


cdef void layer_norm(real_t* X, dim_t nr_dim, dim_t nr_row) nogil:
    for i in range(nr_row):
        mu = Vec.mean(X, nr_dim)
        v = Vec.variance(X, nr_dim)
        if mu == 0. and v == 0:
            X += nr_dim
            continue
        sqrt_var = v ** -0.5
        for j in range(nr_dim):
            X[j] = (X[j] - mu) * sqrt_var
        X += nr_dim


cdef void bwd_layer_norm(real_t* dXa,
        const real_t* dXb, const real_t* Xa, dim_t nr_dim, dim_t nr_row) nogil:
    for i in range(nr_row):
        mu = Vec.mean(Xa, nr_dim)
        v = Vec.variance(Xa, nr_dim)
        inv_sqrt_var = v ** (-1./2)
        inv_var = 1. / v
        sum_dXb = Vec.sum(dXb, nr_dim)
        sum_dXb_dist = 0.
        for j in range(nr_dim):
            sum_dXb_dist += dXb[j] * (Xa[j]-mu)
        for j in range(nr_dim):
            dXa[j] = (
                nr_dim * dXb[j]
                - sum_dXb
                - (Xa[j]-mu) * inv_var * sum_dXb_dist
            )
            dXa[j] *= inv_sqrt_var
            dXa[j] /= nr_dim
        Xa += nr_dim
        dXa += nr_dim
        dXb += nr_dim
 

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
