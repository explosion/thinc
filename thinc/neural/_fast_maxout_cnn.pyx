# cython: infer_types=True
# cython: cdivision=True

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt
cimport cython.parallel
from cymem.cymem cimport Pool

from ._classes.maxout import Maxout
from ..linalg cimport Vec, Mat, VecVec, MatVec, MatMat
from .ops cimport cpu_backprop_maxout as bwd_maxout
from .ops cimport backprop_seq2col as bwd_seq2col

cimport numpy as np

from blis import blis

from ..api import wrap

ctypedef float real_t
ctypedef int dim_t

cdef extern from "stdlib.h":
    void* aligned_alloc(size_t alignment, size_t size) nogil


def MaxoutWindowEncode(maxout, normalize, nr_iter):
    ops = maxout.ops
    def mwe_fwd(Xs, drop=0.):
        cdef np.ndarray weights = maxout.W
        cdef np.ndarray bias = maxout.b
        cdef np.ndarray scale_weights = normalize.G
        cdef np.ndarray shift_weights = normalize.b
 
        cdef np.ndarray inputs = ops.flatten(Xs)
        cdef dim_t nO = maxout.nO
        cdef dim_t nI = maxout.nI
        cdef dim_t nP = maxout.nP
        cdef dim_t nN = inputs.shape[0]

        Ww = <float*>weights.data
        Wb = <float*>bias.data
        Wg = <float*>scale_weights.data
        Wbeta = <float*>shift_weights.data

        cdef Pool mem = Pool()
        # Allocate buffers
        # Total e.g. nO=128, nP=3, nN=1000
        #   128*3*1000
        # + 128*3*1000
        # + 128*1000
        # + 128*1000
        # + 128*1000
        # = 384000 * 3 * 4 iterations
        # = approx 2mb
        cdef float* Xa = <float*>inputs.data
        cdef float* Xb = <float*>mem.alloc(nO*3*nN*nr_iter, sizeof(float))  # ExtractWindow
        cdef float* Xc = <float*>mem.alloc(nO*nP*nN*nr_iter, sizeof(float)) # Affine
        cdef float* Xd = <float*>mem.alloc(nO*nN*nr_iter, sizeof(float))    # MaxPool
        cdef int* which = <int*>mem.alloc(nO*nP*nN*nr_iter, sizeof(int))  # Maxpool
        cdef float* Xe = <float*>mem.alloc(nO*nN*nr_iter, sizeof(float))    # LayerNorm
        cdef float* Xf = <float*>mem.alloc(nO*nN*nr_iter, sizeof(float))    # Rescale, residual
        nonlocals = {
            'mem': mem,
            'Xa': <size_t>Xa,
            'Xb': <size_t>Xb,
            'Xc': <size_t>Xc,
            'Xe': <size_t>Xe,
            'Xf': <size_t>Xf,
            'Ww': <size_t>Ww,
            'Wb': <size_t>Wb,
            'Wg': <size_t>Wg,
            'Wbeta': <size_t>Wbeta,
            'which': <size_t>which
        }
        # Now do the actual work
        for i in range(nr_iter):
            extract_window(Xb,
                Xa, 1, nO, nN)
            affine(Xc,
                Xb, Ww, Wb, nO*nP, nO, nN) 
            maxpool(Xd, which,
                Xc, nO, nP, nN)
            memcpy(Xe,
                Xd, nO*nN*sizeof(Xe[0]))
            layer_norm(Xe,
                nO, nN)
            memcpy(Xf, Xe, nO*nN*sizeof(float))
            rescale(Xf,
                Wg, Wbeta, nO, nN)
            VecVec.add_i(Xf,
                Xa, 1., nO*nN)
            Xa += nO*nN
            Xb += nO*3*nN
            Xc += nO*nP*nN
            Xd += nO*nN
            Xe += nO*nN
            Xf += nO*nN
        Xf -= nO*nN
        cdef np.ndarray outputs = ops.allocate((nN, nO))
        memcpy(<float*>outputs.data,
            Xf, nO*nN*sizeof(float))

        def mwe_bwd(np.ndarray d_outputs, sgd=None):
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
            dWw = <float*>(<np.ndarray>maxout.dW).data
            dWb = <float*>(<np.ndarray>maxout.d_b).data
            dWg = <float*>(<np.ndarray>normalize.dG).data
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
            cdef Pool mem = Pool()
            dXa = <float*>mem.alloc(nO*nN*nr_iter, sizeof(float))  # ExtractWindow
            dXb = <float*>mem.alloc(nO*3*nN*nr_iter, sizeof(float))  # ExtractWindow
            dXc = <float*>mem.alloc(nO*nP*nN*nr_iter, sizeof(float)) # Affine
            dXd = <float*>mem.alloc(nO*nN*nr_iter, sizeof(float))    # MaxPool
            dXe = <float*>mem.alloc(nO*nN*nr_iter, sizeof(float))    # LayerNorm
            for i in range(nr_iter-1, -1, -1):
                bwd_rescale(dXe, dWg, dWbeta,
                    dXf, Xe, Wg, nO, nN)
                bwd_layer_norm(dXd,
                    dXe, Xd, nO, nN)
                bwd_maxpool(dXc,
                    dXd, which, nO, nP, nN)
                bwd_affine(dXb, dWw, dWb,
                    dXc, Xb, Ww, nO*nP, nO, nN) 
                bwd_window(dXa, 
                    dXb, nO, 1, nN) 
                VecVec.add_i(dXa,
                    dXf, 1., nN * nO)
            cdef np.ndarray d_inputs = ops.allocate((nN, nO))
            memcpy(<float*>d_inputs.data,
                dXa, nN*nO*sizeof(float))
            return d_inputs
        return outputs, mwe_bwd
    return wrap(mwe_fwd, maxout)


cdef void maxout_window_encode(real_t* Xe, real_t* Xd, real_t* Xc, real_t* Xb, int* which,
        const real_t* Xa, const real_t* Ww, const real_t* Wb,
        dim_t nO, dim_t nP, dim_t nN) nogil:
    cnn_maxout(Xd, which, Xc, Xb,
        Xa, Ww, Wb, nO, nO, nP, nN)
    memcpy(Xe,
        Xd, nO*nN*sizeof(Xe[0]))
    layer_norm(Xe,
        nO, nN)
    VecVec.add_i(Xe,
        Xa, 1., nO*nN)


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


cdef void extract_window(float* Xb,
        const float* Xa, dim_t nW, dim_t nO, dim_t nN) nogil:
    for w in range(nN):
        # Assume correct padding on words. This means words should
        # start with an eol row, pushing them off-alignment with
        # the outputs (best and which). Words also needs to end with
        # an eol row
        memcpy(&Xb[w*nO*3], &Xa[w*nO], 3*nO*sizeof(float))

cdef void bwd_window(float* dXa,
        const float* dXb, dim_t nO, dim_t nW, dim_t nN) nogil:
    pass

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


cdef void maxpool(float* Xb, int* which,
        const float* Xa, dim_t nO, dim_t nP, dim_t nN) nogil:
    cdef int j
    for w in range(nN):
        for i in range(nO):
            j = argmax(&Xa[w*nO*nP+i*nP], nP)
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
