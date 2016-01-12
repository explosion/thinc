# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_set as Map_set

from .structs cimport IteratorC
from .structs cimport FeatureC
from .structs cimport ConstantsC

from .typedefs cimport len_t
from .typedefs cimport idx_t

from .blas cimport MatMat, MatVec, VecVec, Vec

from .structs cimport do_iter_t
from .structs cimport do_feed_fwd_t
from .structs cimport do_end_fwd_t
from .structs cimport do_begin_fwd_t
from .structs cimport do_begin_bwd_t
from .structs cimport do_end_bwd_t
from .structs cimport do_feed_bwd_t
from .structs cimport do_update_t


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


DEF EPS = 0.000001 
DEF ALPHA = 1.0


cdef void dot_plus__ELU(
    float* output,
        const float* bias,
        len_t nr_out,
        const float* input_,
            len_t nr_in,
        const float* W
) nogil:
    dot_plus(output,
        bias, nr_out, input_, nr_in, W)
    ELU(output, nr_out)


cdef void dELU__dDot(
    float* dX,
    float* dY,
        len_t nr_wide,
        const float* Y,
        len_t nr_above,
        const float* W
) nogil:
    d_ELU(dY,
        Y, nr_above)
    d_dot(dX,
        nr_above, dY, nr_wide, W)


cdef void dot_plus(
    float* out,
        const float* bias,
            len_t nr_out,
        const float* in_,
            len_t nr_in,
        const float* W
) nogil:
    MatVec.dot(out,
        W, in_, nr_out, nr_in)
    VecVec.add_i(out,
        bias, 1.0, nr_out)


cdef void sparse_dot_plus(
    float* out,
        const float* bias,
            len_t nr_out,
        const FeatureC* feats,
            len_t nr_feat,
        const MapC* const* Ws
) nogil:
    for i in range(nr_feat):
        W = Ws[feats[i].i]
        if W is not NULL: # Shouldn't be NULL
            row = <const float*>Map_get(W, feats[i].key)
            if row is not NULL: # Can be NULL
                VecVec.add_i(out,
                    row, feats[i].value, nr_out)
    VecVec.add_i(out,
        bias, 1.0, nr_out)


cdef void d_dot(
    float* btm_diff,
        len_t nr_btm,
        const float* top_diff,
        len_t nr_top,
        const float* W,
) nogil:
    MatVec.T_dot(btm_diff,
        W, top_diff, nr_top, nr_btm)


cdef void ELU(float* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = ALPHA * (expf(out[i]) - 1)


cdef void d_ELU(float* delta, const float* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] < 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void softmax(float* out, len_t nr_out) nogil:
    #w = exp(w - max(w))
    Vec.add_i(out,
        -Vec.max(out, nr_out), nr_out)
    Vec.exp_i(out,
        nr_out)
    #w = w / sum(w)
    cdef float norm = Vec.sum(out, nr_out)
    if norm != 0:
        Vec.div_i(out,
            norm, nr_out)


cdef void d_log_loss(
    float* loss,
        const float* costs,
        const float* scores,
            len_t nr_out
) nogil:
    # This assumes only one true class
    cdef idx_t i
    for i in range(nr_out):
        loss[i] = scores[i] - (costs[i] == 0)


@cython.cdivision(True)
cdef void adam(
    float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    cdef float beta1 = 0.90
    cdef float beta2 = 0.999
    # Add the derivative of the L2-loss to the gradient
    cdef idx_t i
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    mom1 = moments
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)
    mom2 = &moments[nr_weight]
    for i in range(nr_weight):
        mom2[i] = (beta2 * mom2[i]) + ((1-beta2) * gradient[i] * gradient[i])
    # More efficient version, from the paper
    cdef float a_t = hp.e * sqrtf(1-beta2**hp.t) / (1-beta1**hp.t)
    for i in range(nr_weight):
        weights[i] -= a_t * (mom1[i] / (sqrtf(mom2[i]) + EPS))
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void adagrad(
    float* weights, float* moments, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    # Add the derivative of the L2-loss to the gradient
    cdef int i
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    VecVec.add_pow_i(moments,
        gradient, 2.0, nr_weight)
    for i in range(nr_weight):
        gradient[i] *= hp.e / (sqrtf(moments[i]) + EPS)
    # Make the (already scaled) update
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void adadelta(float* weights, float* momentum, float* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    cdef float alpha = 0.90
    # Add the derivative of the L2-loss to the gradient
    cdef int i
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    avg = momentum
    Vec.mul_i(avg,
        alpha, nr_weight)
    for i in range(nr_weight):
        avg[i] += (1-alpha) * gradient[i] * gradient[i]
    step = &momentum[nr_weight]
    for i in range(nr_weight):
        gradient[i] *= sqrtf(step[i] + EPS) / sqrtf(avg[i] + EPS)
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)
    Vec.mul_i(step,
        alpha, nr_weight)
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)
 

@cython.cdivision(True)
cdef void vanilla_sgd_update_step(float* weights, float* moments, float* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil:
    '''
    Update weights with vanilla SGD
    '''
    # Add the derivative of the L2-loss to the gradient
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    VecVec.add_i(weights,
        gradient, -hp.e, nr_weight)
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


########
# Batch Normalization, non-functional draft


cdef void normalize(float* x, float* Ex, float* Vx, len_t nr_x, float alpha) nogil:
    # Upd EMA estimate of mean and variance
    # See eq at the end of this:
    # http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    cdef idx_t i
    cdef float diff
    cdef float incr
    for i in range(nr_x):
        diff = x[i] - Ex[i]
        incr = alpha * diff
        Vx[i] = (1.0 - alpha) * (Vx[i] + diff * incr)
        Ex[i] += incr
    # Normalize
    for i in range(nr_x):
        x[i] = (x[i] - Ex[i]) / sqrtf(Vx[i] + EPS)


cdef void d_normalize(float* bwd, float* E_dEdXh, float* E_dEdXh_dot_Xh,
        const float* Xh, const float* Vx, len_t n, float alpha) nogil:
    # Update EMA estimate of mean(dL/dX_hat)
    Vec.mul_i(E_dEdXh,
        alpha, n)
    VecVec.add_i(E_dEdXh,
        bwd, 1-alpha, n)
    # Update EMA estimate of mean(dE/dX_hat \cdot X_hat)
    Vec.mul_i(E_dEdXh_dot_Xh,
        alpha, n)
    for i in range(n):
        E_dEdXh_dot_Xh[i] += (1-alpha) * bwd[i] * Xh[i]
    # Simplification taken from Caffe, I think by cdoersch
    # if X' = (X-mean(X))/sqrt(var(X)+eps), then
    # dE/dX =
    #   (dE/dXh - mean(dE/dXh) - mean(dE/dXh * Xh) * Xh)
    #     ./ sqrt(var(X) + eps)
    # bwd is dE/dXh to start with. We change it to dE/dX in-place.
    for i in range(n):
        bwd[i] -= E_dEdXh[i] - E_dEdXh_dot_Xh[i] * Xh[i]
        bwd[i] /= sqrtf(Vx[i] + EPS)


cdef void dot__normalize__dot_plus__ELU(
    float* output,
    float* mid_result,
    float* Ex,
    float* Vx,
        const float* bias,
        const float* gamma,
        len_t nr_out,
        const float* input_,
            len_t nr_in,
        const float* W,
        float alpha
) nogil:
    MatVec.dot(mid_result,
        input_, W, nr_out, nr_in)
    normalize(mid_result, Ex, Vx,
        nr_out, alpha)
    VecVec.mul(output,
        mid_result, gamma, nr_out)
    VecVec.add_i(output,
        bias, 1.0, nr_out)
    ELU(output, nr_out)


cdef void d_ELU__dot__normalize__dot(
    float* dY,
    float* dXh,
    float* dX,
    float* E_dXh,
    float* E_dXh_Xh,
        const float* Y,
        const float* Xh,
        const float* Vx,
        const float* gamma,
        len_t nr_out,
        len_t nr_in,
        const float* W,
        float ema_speed
) nogil:
    # This must be wrong. X is from bottom, right?
    # Y = ELU(dot(G, BN(W*x+b))), i.e. our layer's final output
    d_ELU(dY,
        Y, nr_out) 
    VecVec.mul(dXh,
        dY, gamma, nr_out)
    d_normalize(dXh, E_dXh, E_dXh_Xh,
        Xh, Vx, nr_out, ema_speed)
    d_dot(dX,
        nr_in, dXh, nr_out, W)
