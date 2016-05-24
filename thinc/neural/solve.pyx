# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrt
from .. cimport prng

import numpy


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


prng.normal_setup()


@cython.cdivision(True)
cdef void noisy_update(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    # Add the derivative of the L2-loss to the gradient
    cdef int i
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    # Clip gradient
    grad_norm = Vec.norm(gradient, nr_weight)
    if grad_norm >= 100:
        Vec.mul_i(gradient, 100.0 / grad_norm, nr_weight)
    #cdef weight_t variance 
    # Add gradient noise
    #variance = hp.e / ((1 + hp.t) ** 0.55)
    variance = hp.e
    for i in range(nr_weight):
        if gradient[i] != 0:
            gradient[i] += prng.normal() * variance
    VecVec.add_i(weights,
        gradient, -hp.e, nr_weight)
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void vanilla_sgd(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    VecVec.add_i(weights,
        gradient, -hp.e, nr_weight)
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void sgd_cm(weight_t* weights, weight_t* momentum, weight_t* gradient,
        len_t nr_weight,const ConstantsC* hp) nogil:
    '''
    Update weights with SGD and classical momentum
    '''
    Vec.mul_i(momentum, hp.m, nr_weight)
    VecVec.add_i(momentum,
        gradient, hp.e, nr_weight)
    VecVec.add_i(weights,
        momentum, -1.0, nr_weight)
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void adam(
    weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    cdef weight_t beta1 = 0.90
    cdef weight_t beta2 = 0.999
    mom1 = moments
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)
    mom2 = &moments[nr_weight]
    for i in range(nr_weight):
        mom2[i] = (beta2 * mom2[i]) + ((1-beta2) * gradient[i] * gradient[i])
    # More efficient version, from the paper
    cdef weight_t a_t = hp.e * sqrt(1-beta2**hp.t) / (1-beta1**hp.t)
    for i in range(nr_weight):
        weights[i] -= a_t * (mom1[i] / (sqrt(mom2[i]) + EPS))
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void adagrad(
    weight_t* weights, weight_t* moments, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    VecVec.add_pow_i(moments,
        gradient, 2.0, nr_weight)
    for i in range(nr_weight):
        gradient[i] *= hp.e / (sqrt(moments[i]) + EPS)
    # Make the (already scaled) update
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


@cython.cdivision(True)
cdef void adadelta(weight_t* weights, weight_t* momentum, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    cdef weight_t alpha = 0.90
    cdef int i
    avg = momentum
    Vec.mul_i(avg,
        alpha, nr_weight)
    for i in range(nr_weight):
        avg[i] += (1-alpha) * gradient[i] * gradient[i]
    step = &momentum[nr_weight]
    for i in range(nr_weight):
        gradient[i] *= sqrt(step[i] + EPS) / sqrt(avg[i] + EPS)
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)
    Vec.mul_i(step,
        alpha, nr_weight)
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)
