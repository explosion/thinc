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
cdef void clip_gradient(weight_t* gradient, weight_t threshold, int nr_weight) nogil:
    # Clip gradient
    grad_norm = Vec.norm(gradient, nr_weight)
    if grad_norm >= threshold:
        Vec.mul_i(gradient, threshold / grad_norm, nr_weight)


@cython.cdivision(True)
cdef void add_gradient_noise(weight_t* gradient, weight_t noise_level,
        weight_t timestep, int nr_weight) nogil:
    variance = noise_level / ((1 + timestep) ** 0.55)
    if variance >= 0.000001:
        for i in range(nr_weight):
            if gradient[i] != 0:
                gradient[i] += prng.normal() * variance


@cython.cdivision(True)
cdef void update_averages(weight_t* ema,
        const weight_t* weights, int nr_weight, weight_t t) nogil:
    cdef weight_t decay = (1.0 + t) / (10.0 + t)
    if decay > 0.9999:
        decay = 0.9999
    for i in range(nr_weight):
        ema[i] -= (1-decay) * (ema[i] - weights[i])


@cython.cdivision(True)
cdef void vanilla_sgd(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
 
    VecVec.add_i(weights,
        gradient, -hp.e, nr_weight)
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)

    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void sgd_cm(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    '''
    Update weights with SGD and classical momentum
    '''
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
    
    momentum = weights + nr_weight * 2
    Vec.mul_i(momentum, hp.m, nr_weight)
    VecVec.add_i(momentum,
        gradient, hp.e, nr_weight)
    VecVec.add_i(weights,
        momentum, -1.0, nr_weight)
    
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void adam(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
 
    cdef weight_t beta1 = 0.90
    cdef weight_t beta2 = 0.999
    cdef weight_t eps = 1e-08
    cdef weight_t learn_rate = hp.e
    mom1 = weights + nr_weight * 2
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)
    mom2 = weights + nr_weight * 3
    for i in range(nr_weight):
        mom2[i] = (beta2 * mom2[i]) + ((1-beta2) * gradient[i] * gradient[i])
    # More efficient version, from the paper
    cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t)
    for i in range(nr_weight):
        weights[i] -= a_t * (mom1[i] / (sqrt(mom2[i]) + eps))

    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void adagrad(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
    
    momentum = weights + nr_weight * 2
    VecVec.add_pow_i(momentum,
        gradient, 2.0, nr_weight)
    for i in range(nr_weight):
        gradient[i] *= hp.e / (sqrt(momentum[i]) + EPS)
    # Make the (already scaled) update
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)
    
    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void adadelta(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
    
    avg = weights + nr_weight * 2
    step = weights + nr_weight * 3
    cdef weight_t alpha = 0.90
    Vec.mul_i(avg,
        alpha, nr_weight)
    for i in range(nr_weight):
        avg[i] += (1-alpha) * gradient[i] * gradient[i]
    for i in range(nr_weight):
        gradient[i] *= sqrt(step[i] + EPS) / sqrt(avg[i] + EPS)
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)
    Vec.mul_i(step,
        alpha, nr_weight)

    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)
