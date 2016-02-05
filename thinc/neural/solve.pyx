# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrtf


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


@cython.cdivision(True)
cdef void vanilla_sgd(float* weights, float* moments, float* gradient,
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


@cython.cdivision(True)
cdef void sgd_cm(float* weights, float* momentum, float* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil:
    '''
    Update weights with SGD and classical momentum
    '''
    # Add the derivative of the L2-loss to the gradient
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    Vec.mul_i(momentum, hp.m, nr_weight)
    VecVec.add_i(momentum,
        gradient, hp.e, nr_weight)
    VecVec.add_i(weights,
        momentum, -1.0, nr_weight)
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)


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
