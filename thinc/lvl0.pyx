# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_set as Map_set

from .structs cimport FeatureC
from .structs cimport ConstantsC

from .typedefs cimport len_t
from .typedefs cimport idx_t

from .blas cimport MatMat, MatVec, VecVec, Vec

from .structs cimport do_feed_fwd_t
from .structs cimport do_feed_bwd_t
from .structs cimport do_update_t


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


DEF EPS = 0.000001 
DEF ALPHA = 1.0


cdef void dot_plus__ELU(float** fwd,
        const float* W, const len_t* shape, int nr_above) nogil:
    bias = W + shape[1] * shape[0]
    # Linear
    MatVec.dot(fwd[1],
        W, fwd[0], shape[1], shape[0])
    VecVec.add_i(fwd[1],
        bias, 1.0, shape[1])
    # Apply non-linearity
    if nr_above >= 2:
        ELU(fwd[1],
            shape[1])
    else:
        softmax(fwd[1],
            shape[1])
 

cdef void d_ELU__dot(float** bwd,
        const float* W, const float* const* fwd, const len_t* shape, int iteration) nogil:
    # Set the gradient for bwd[1] 
    #MatMat.add_outer_i(gradient,
    #    bwd[1], fwd[0], shape[1], shape[0])
    ##VecVec.add_i(gradient + shape[1] * shape[0],
    #    bwd[1], 1.0, shape[1])
    # Set the partial derivative for bwd[0], so next step can set its gradient
    MatVec.T_dot(bwd[0],
        W, bwd[1], shape[1], shape[0])
    d_ELU(bwd[0],
        fwd[0], shape[0])
    

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
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)
