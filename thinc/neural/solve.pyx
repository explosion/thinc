# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport sqrt

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport v_norm, v_fill, v_mul, v_pow
from ..linalg cimport vv_add, vv_batch_add, vv_add_pow, vv_mul, vv_dot
from .. cimport prng

import numpy


DEF EPS = 0.00000001
DEF ALPHA = 1.0


cdef void clip_gradient(weights_ft gradient, weight_t threshold, int nr_weight) nogil:
    # Clip gradient
    grad_norm = v_norm(gradient, nr_weight)
    if grad_norm >= threshold:
        v_mul(gradient, threshold / grad_norm, nr_weight)


cdef void add_gradient_noise(weights_ft gradient, weight_t noise_level,
        weight_t timestep, int nr_weight) nogil:
    if noise_level == 0:
        return
    if weights_ft is dense_weights_t:
        for i in range(nr_weight):
            if gradient[i] != 0:
                gradient[i] += prng.get_normal() * noise_level
    else:
        for i in range(nr_weight):
            G = gradient[i]
            while G.key >= 0:
                if G.val != 0:
                    G.val += prng.get_normal() * noise_level
                G += 1

cdef void update_averages(weights_ft ema,
        const_weights_ft weights, int nr_weight, weight_t t) nogil:
    cdef weight_t decay = (1.0 + t) / (10.0 + t)
    if decay > 0.9999:
        decay = 0.9999

    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        for i in range(nr_weight):
            ema[i] -= (1-decay) * (ema[i] - weights[i])
    elif weights_ft is sparse_weights_t and const_weights_ft is const_sparse_weights_t:
        for i in range(nr_weight):
            e = ema[i]
            w = weights[i]
            while e.key >= 0:
                e.val -= (1-decay) * (e.val - w.val)
                e += 1
                w += 1
    else:
        # TODO panic
        pass


cdef void ensure_sparsity(weights_ft gradient,
        const_weights_ft weights, int nr_weight) nogil:
    if weights_ft is dense_weights_t and const_weights_ft is const_dense_weights_t:
        for i in range(nr_weight):
            if weights[i] == 0:
                gradient[i] = 0
    else:
        pass
 

@cython.cdivision(True)
cdef void sgd_cm(weights_ft weights, weights_ft gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    '''
    Update weights with SGD and classical momentum
    '''
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
    
    if hp.m <= 0:
        vv_add(weights,
            gradient, -hp.e, nr_weight)
    else:
        momentum = weights + nr_weight * 2
        v_mul(momentum, hp.m, nr_weight)
        vv_add(momentum,
            gradient, hp.e, nr_weight)
        vv_add(weights,
            momentum, -1.0, nr_weight)
    v_fill(gradient,
        0, nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void nag(weights_ft weights, weights_ft gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    '''
    Update weights with SGD and classical momentum
    '''
    clip_gradient(gradient,
        100.0, nr_weight)
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
    
    momentum = weights + nr_weight * 2
    # http://cs231n.github.io/neural-networks-3/
    # v_prev = v # back this up
    # v = mu * v - lr * gradient # velocity update stays the same
    # x += -mu * v_prev + (1 + mu) * v # position update changes form
    # Implement this as
    # x += -mu * v
    # v *= mu
    # v -= lr * gradient
    # x += (1+mu) * v
    vv_add(weights,
        momentum, -hp.m, nr_weight)
    v_mul(momentum,
        hp.m, nr_weight)
    vv_add(momentum,
        gradient, -hp.e, nr_weight)
    vv_add(weights,
        momentum, 1+hp.m, nr_weight)

    v_fill(gradient,
        0, nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void adam(weights_ft weights, weights_ft gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    clip_gradient(gradient,
        100.0, nr_weight)
    ensure_sparsity(gradient,
        weights, nr_weight)
 
    add_gradient_noise(gradient,
        hp.w, hp.t, nr_weight)
 
    cdef weight_t beta1 = 0.90
    cdef weight_t beta2 = 0.999
    cdef weight_t eps = 1e-08
    cdef weight_t learn_rate = hp.e
    mom1 = weights + nr_weight * 2
    v_mul(mom1,
        beta1, nr_weight)
    vv_add(mom1,
        gradient, 1-beta1, nr_weight)
    mom2 = weights + nr_weight * 3
    v_mul(mom2,
        beta2, nr_weight)
    v_pow(gradient,
        2.0, nr_weight)
    vv_add(mom2,
        gradient, 1-beta2, nr_weight)
    # More efficient version, from the paper
    cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t)
    for i in range(nr_weight):
        if weights_ft is dense_weights_t:
            weights[i] -= a_t * (mom1[i] / (sqrt(mom2[i]) + eps))
        else:
            w = weights[i]
            m1 = mom1[i]
            m2 = mom2[i]
            while w.key >= 0:
                w.val -= a_t * (m1.val / (sqrt(m2.val) + eps))
                w += 1
                m1 += 1
                m2 += 1
    v_fill(gradient,
        0, nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)
