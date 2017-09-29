# cython: infer_types=True
# cython: cdivision=True

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt
cimport cython.parallel
from cymem.cymem cimport Pool

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


def MaxoutWindowEncode(nr_unit, nr_iter):
    maxout = Maxout(nr_unit, nr_unit*3, pieces=3)
    normalize = LayerNorm(maxout)
    ops = maxout.ops
    nonlocals = {}
    def mwe_fwd(Xs, drop=0.):
        cdef np.ndarray weights = maxout.W
        cdef np.ndarray bias = maxout.b
        cdef np.ndarray scale_weights = normalize.G
        cdef np.ndarray shift_weights = normalize.b
 
        cdef np.ndarray inputs = ops.flatten(Xs)
        lengths = ops.asarray([len(x) for x in Xs], dtype='i')
        cdef dim_t nO = maxout.nO
        cdef dim_t nI = maxout.nI
        cdef dim_t nP = maxout.nP
        cdef dim_t nN = inputs.shape[0]

        Ww = <float*>weights.data
        Wb = <float*>bias.data
        Wg = <float*>scale_weights.data
        Wbeta = <float*>shift_weights.data

        # Allocate buffers
        # Total e.g. nO=128, nP=3, nN=1000
        #   128*3*1000
        # + 128*3*1000
        # + 128*1000
        # + 128*1000
        # + 128*1000
        # = 384000 * 3 * 4 iterations
        # = approx 2mb
        #
        # TODO: We're leaking this memory!!
        cdef float* Xa = <float*>calloc(nO*nN, sizeof(float))
        memcpy(Xa, <float*>inputs.data, nO*nN*sizeof(float))
        cdef float* Xb = <float*>calloc(nO*3*nN*nr_iter, sizeof(float))  # ExtractWindow
        cdef float* Xc = <float*>calloc(nO*nP*nN*nr_iter, sizeof(float)) # Affine
        cdef float* Xd = <float*>calloc(nO*nN*nr_iter, sizeof(float))    # MaxPool
        cdef int* which = <int*>calloc(nO*nN*nr_iter, sizeof(int))  # Maxpool
        cdef float* Xe = <float*>calloc(nO*nN*nr_iter, sizeof(float))    # LayerNorm
        cdef float* Xf = <float*>calloc(nO*nN*nr_iter, sizeof(float))    # Rescale, residual
        nonlocals.update({
            'inputs': inputs,
            'Xa': <size_t>inputs.data,
            'Xb': <size_t>Xb,
            'Xc': <size_t>Xc,
            'Xd': <size_t>Xd,
            'Xe': <size_t>Xe,
            'Xf': <size_t>Xf,
            'Ww': <size_t>Ww,
            'Wb': <size_t>Wb,
            'Wg': <size_t>Wg,
            'Wbeta': <size_t>Wbeta,
            'which': <size_t>which
        })
        # Now do the actual work
        for i in range(nr_iter):
            seq2col(Xb,
                Xa, 1, nO, nN)
            affine(Xc,
                Xb, Ww, Wb, nO*nP, nO*3, nN) 
            maxpool(Xd, which,
                Xc, nO, nP, nN)
            memcpy(Xe,
                Xd, nO*nN*sizeof(Xe[0]))
            layer_norm(Xe,
                nO, nN)
            memcpy(Xf, Xe, nO*nN*sizeof(float))
            rescale(Xf,
                Wg, Wbeta, nO, nN)
            VecVec.add_i(Xa,
                Xf, 1., nO*nN)
            Xb += nO*3*nN
            Xc += nO*nP*nN
            Xd += nO*nN
            Xe += nO*nN

        cdef np.ndarray outputs = ops.allocate((nN, nO))
        memcpy(<float*>outputs.data,
            Xa, nO*nN*sizeof(float))

        def mwe_bwd(d_output_seqs, sgd=None):
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
            cdef np.ndarray d_outputs = ops.flatten(d_output_seqs)
            dWw = <float*>(<np.ndarray>maxout.d_W).data
            dWb = <float*>(<np.ndarray>maxout.d_b).data
            dWg = <float*>(<np.ndarray>normalize.d_G).data
            dWbeta = <float*>(<np.ndarray>normalize.d_b).data
            dXf = <float*>d_outputs.data
            Xa = <float*><size_t>nonlocals['Xa']
            Xb = <float*><size_t>nonlocals['Xb']
            Xc = <float*><size_t>nonlocals['Xc']
            Xd = <float*><size_t>nonlocals['Xd']
            Xe = <float*><size_t>nonlocals['Xe']
            Xf = <float*><size_t>nonlocals['Xf']
            Ww = <float*><size_t>nonlocals['Ww']
            Wb = <float*><size_t>nonlocals['Wb']
            Wg = <float*><size_t>nonlocals['Wg']
            Wbeta = <float*><size_t>nonlocals['Wbeta']
            which = <int*><size_t>nonlocals['which']
            dXa = <float*>calloc(nO*nN*nr_iter, sizeof(float))  # ExtractWindow
            dXb = <float*>calloc(nO*3*nN*nr_iter, sizeof(float))  # ExtractWindow
            dXc = <float*>calloc(nO*nP*nN*nr_iter, sizeof(float)) # Affine
            dXd = <float*>calloc(nO*nN*nr_iter, sizeof(float))    # MaxPool
            dXe = <float*>calloc(nO*nN*nr_iter, sizeof(float))    # LayerNorm
            cdef dim_t i
            for i in range(nr_iter-1, -1, -1):
                bwd_rescale(dXe, dWg, dWbeta,
                    dXf, Xe, Wg, nO, nN)
                bwd_layer_norm(dXd,
                    dXe, Xd, nO, nN)
                bwd_maxpool(dXc,
                    dXd, which, nO, nP, nN)
                bwd_affine(dXb, dWw, dWb,
                    dXc, Xb, Ww, nO*nP, nO, nN) 
                bwd_seq2col(dXa, 
                    dXb, 1, nO, nN) 
                VecVec.add_i(dXa,
                    dXf, 1., nN * nO)

            cdef np.ndarray d_inputs = ops.allocate((nN, nO))
            memcpy(<float*>d_inputs.data,
                dXa, nN*nO*sizeof(float))
            return ops.unflatten(d_inputs, lengths)
        return ops.unflatten(outputs, lengths), mwe_bwd
    model = wrap(mwe_fwd, normalize)
    model.maxout = maxout
    model.normalize = normalize
    return model


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
            dXb += nP
            dXa += 1
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


cdef void bwd_layer_norm(real_t* dX,
        const real_t* dXh, const real_t* X, dim_t nr_dim, dim_t nr_row) nogil:
    for i in range(nr_row):
        mu = Vec.mean(X, nr_dim)
        v = Vec.variance(X, nr_dim)
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
