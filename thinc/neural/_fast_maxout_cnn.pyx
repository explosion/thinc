# cython: infer_types=True
# cython: cdivision=True
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt
from cymem.cymem cimport Pool
from libcpp.vector cimport vector

from ._classes.maxout import Maxout
from ._classes.layernorm import LayerNorm
from ..linalg cimport Vec, Mat, VecVec, MatVec, MatMat
from .. cimport openblas

import numpy
cimport numpy as np

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
    cdef Pool mem
    cdef dim_t nX
    cdef dim_t nO
    cdef dim_t nP
    cdef dim_t nr_iter

    def __init__(self, lengths, dim_t nO, dim_t nP, dim_t nr_iter, gradient=False):
        self.mem = Pool()
        # Allocate buffers
        # Total e.g. nO=128, nP=3, nN=1000
        #   128*3*1000
        # + 128*3*1000
        # + 128*1000
        # + 128*1000
        # + 128*1000
        # = 384000 * 3 * 4 iterations
        # = approx 2mb
        nX = len(lengths)
        self.nX = nX
        self.nO = nO
        self.nP = nP
        self.nr_iter = nr_iter
        self.a = <float**>self.mem.alloc(nX, sizeof(float*))
        self.b = <float**>self.mem.alloc(nX, sizeof(float*))
        self.c = <float**>self.mem.alloc(nX, sizeof(float*))
        self.d = <float**>self.mem.alloc(nX, sizeof(float*))
        self.e = <float**>self.mem.alloc(nX, sizeof(float*))
        self.f = <float**>self.mem.alloc(nX, sizeof(float*))
        self.which = <int**>self.mem.alloc(nX, sizeof(int*))
        self.lengths = <int*>self.mem.alloc(nX, sizeof(int))
        cdef np.ndarray X
        cdef dim_t nN
        for i, nN in enumerate(lengths):
            self.a[i] = <float*>self.mem.alloc(nO*nN*nr_iter, sizeof(float))
            self.b[i] = <float*>self.mem.alloc(nO*3*nN*nr_iter, sizeof(float))
            self.c[i] = <float*>self.mem.alloc(nO*nP*nN*nr_iter, sizeof(float))
            self.d[i] = <float*>self.mem.alloc(nO*nN*nr_iter, sizeof(float))
            self.e[i] = <float*>self.mem.alloc(nO*nN*nr_iter, sizeof(float))
            self.f[i] = <float*>self.mem.alloc(nO*nN*nr_iter, sizeof(float))
            self.which[i] = <int*>self.mem.alloc(nO*nN*nr_iter, sizeof(int))
            self.lengths[i] = nN

    def set_inputs(self, Xs):
        cdef np.ndarray X
        for i, X in enumerate(Xs):
            memcpy(self.a[i], <float*>X.data, X.shape[0]*X.shape[1]*sizeof(float))

    def get_d_inputs(self):
        d_inputs = []
        cdef np.ndarray Y
        for i in range(self.nX):
            Y = numpy.zeros((self.lengths[i], self.nO), dtype='f')
            memcpy(<float*>Y.data,
                self.a[i], self.nO*self.lengths[i]*sizeof(float))
            d_inputs.append(Y)
        return d_inputs
 
    def set_d_outputs(self, dXs):
        cdef np.ndarray dX
        for i, dX in enumerate(dXs):
            memcpy(self.f[i], <float*>dX.data, dX.shape[0]*dX.shape[1]*sizeof(float))

    def get_outputs(self):
        outputs = []
        cdef np.ndarray Y
        for i in range(self.nX):
            Y = numpy.zeros((self.lengths[i], self.nO), dtype='f')
            memcpy(<float*>Y.data,
                self.a[i], self.nO*self.lengths[i]*sizeof(float))
            outputs.append(Y)
        return outputs
 

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
        if drop is not None and drop > 0:
            raise ValueError("MaxoutWindow encoder doesn't support dropout yet")
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
    ops = maxout.ops
    cdef dim_t nO = maxout.nO
    cdef dim_t nP = maxout.nP
    cdef dim_t nX = len(inputs)
    cdef np.ndarray lengths = ops.asarray([len(x) for x in inputs], dtype='i')
    cdef _Weights W = _Weights(maxout.W, maxout.b, normalize.G, normalize.b)
    cdef _Activations X = _Activations(lengths, nO, nP, nr_iter)
    X.set_inputs(inputs)

    cdef dim_t i
    for i in range(nX):
        mwe_forward(X.a[i], X.b[i], X.c[i], X.d[i], X.e[i], X.f[i], X.which[i],
            W.syn, W.bias, W.scale, W.shift, nO, nP, X.lengths[i], nr_iter)

    nonlocals = {}
    nonlocals['X'] = X
    nonlocals['lengths'] = lengths

    def mwe_bwd(d_outputs, sgd=None):
        cdef _Weights W = _Weights(maxout.W, maxout.b,
                                   normalize.G, normalize.b)
        cdef _Activations X = nonlocals['X']
        cdef np.ndarray lengths = nonlocals['lengths']
        cdef _Activations dX = _Activations(lengths, nO, nP, 1)
        dX.set_d_outputs(d_outputs) 

        cdef _Weights dW = _Weights(maxout.d_W, maxout.d_b,
                                    normalize.d_G, normalize.d_b)
        cdef dim_t i
        for i in range(nX):
            mwe_backward(
                dX.a[i], dX.b[i], dX.c[i], dX.d[i], dX.e[i], dX.f[i],
                dW.syn, dW.bias, dW.scale, dW.shift,
                    X.which[i], X.b[i], X.d[i], X.e[i],
                    W.syn, W.scale,
                    nO, nP, dX.lengths[i], nr_iter)
        if sgd is not None:
            sgd(maxout._mem.weights, maxout._mem.gradient, key=maxout.id)
            sgd(normalize._mem.weights, normalize._mem.gradient, key=normalize.id)
        return dX.get_d_inputs()
    return X.get_outputs(), mwe_bwd


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


cdef void mwe_backward(
        float* dXa, float* dXb, float* dXc, float* dXd, float* dXe, float* dXf,
        float* dW_syn, float* dW_bias, float* dW_scale, float* dW_shift,
            const int* which, const float* Xb, const float* Xd, const float* Xe,
            const float* W_syn, const float* W_scale,
            dim_t nO, dim_t nP, dim_t nN, dim_t nr_iter) nogil:
    for i in range(nr_iter):
        which += nO*nN
        Xb += nO*3*nN
        Xd += nO*nN
        Xe += nO*nN
    for i in range(nr_iter-1, -1, -1):
        which -= nO*nN
        Xb -= nO*3*nN
        Xd -= nO*nN
        Xe -= nO*nN

        bwd_rescale(dXe, dW_scale, dW_shift,
            dXf, Xe, W_scale, nO, nN)
        bwd_layer_norm(dXd,
            dXe, Xd, nO, nN)
        bwd_maxpool(dXc,
            dXd, which, nO, nP, nN)
        bwd_affine(dXb, dW_syn, dW_bias,
            dXc, Xb, W_syn, nO*nP, nO*3, nN) 
        bwd_seq2col(dXa, 
            dXb, 1, nO, nN) 
        VecVec.add_i(dXa,
            dXf, 1., nN * nO)

        memcpy(dXf, dXa, nO*nN*sizeof(float))


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
    memset(dXa, 0, nI*nN*sizeof(float))
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
    openblas.simple_gemm(outputs,
        nB, nO, inputs, nB, nI, weights, nO, nI, 0, 1)
    for i in range(nB):
        openblas.simple_axpy(&outputs[i*nO], nO,
            bias, 1.)


cdef void bwd_affine(float* d_inputs, float* dW, float* db,
        const float* d_outputs, const float* inputs, const float* weights,
        dim_t nO, dim_t nI, dim_t nB) nogil:
    memset(d_inputs, 0, nB*nI*sizeof(float))
    for i in range(nB):
        openblas.simple_axpy(db, nO,
            &d_outputs[i*nO], 1.)
    openblas.simple_gemm(dW, nO, nI,
        d_outputs, nB, nO, inputs, nB, nI, 1, 0)
    openblas.simple_gemm(d_inputs, nB, nI,
        d_outputs, nB, nO, weights, nO, nI, 0, 0)


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
    cdef double sqrt_var
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
    cdef double inv_var, inv_sqrt_var, sum_dXb_dist
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
