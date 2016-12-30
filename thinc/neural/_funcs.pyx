# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..typedefs cimport weight_t
from ..structs cimport ConstantsC

from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


cdef void ELU(weight_t* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = ALPHA * (exp(out[i]) - 1)


cdef void ReLu(weight_t* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = 0


cdef void softmax(weight_t* out, len_t nr_out) nogil:
    #w = exp(w - max(w))
    Vec.add_i(out,
        -Vec.max(out, nr_out), nr_out)
    Vec.exp_i(out,
        nr_out)
    #w = w / sum(w)
    cdef weight_t norm = Vec.sum(out, nr_out)
    if norm != 0:
        Vec.div_i(out,
            norm, nr_out)

# Backward functions

cdef void d_ReLu(weight_t* delta, const weight_t* signal_out, int n) nogil:
    # Backprop the ReLu transformation
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] = 0


cdef inline void d_ELU(weight_t* delta, const weight_t* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void d_log_loss(weight_t* loss,
    const weight_t* costs, const weight_t* scores, len_t nr_out
) nogil:
    # If there's more than one gold class, appoint the top scoring one the best
    cdef idx_t i
    cdef idx_t best = 0
    cdef weight_t score = 0.0
    for i in range(nr_out):
        loss[i] = scores[i]
        if scores[i] >= score and costs[i] == 0:
            score = scores[i]
            best = i
    loss[best] -= 1


cdef void d_hinge_loss(weight_t* loss,
        const weight_t* costs, const weight_t* scores, len_t nr_out) nogil:
    cdef int best = -1
    cdef int guess = -1
    for i in range(nr_out):
        loss[i] = 0.0
        if costs[i] == 0:
            if best == -1 or scores[i] >= scores[best]:
                best = i
        elif costs[i] > 0:
            if guess == -1 or scores[i] >= scores[guess]:
                guess = i
    margin = (scores[guess] - scores[best]) + 1
    if margin > 0:
        loss[best] = -margin
        loss[guess] = margin


@cython.boundscheck(False)
cdef void d_batchnorm(weight_t* diff, weight_t* est_mean, weight_t* est_var,
        const weight_t* _x, int nr_out, int nr_batch) nogil:
    # Simplification by Clement Thorey, here:
    # http://cthorey.github.io./backpropagation/
    # N = nr_batch
    # D = nr_out
    # cdef np.ndarray[double, ndim=1] var, inv_sqrt_var, inv_var
    # var = x.var(0) + EPS
    # inv_var = var ** -1.
    # inv_sqrt_var = var ** (-1. / 2.)
    # x_mu = x - x.mean(0)
    #dx = (1. / N) \
    #   * inv_sqrt_var \
    #   * (N \
    #     * dy \
    #     - np.sum(dy, axis=0) \
    #     - x_mu \
    #       * inv_var \
    #       * np.sum(dy * x_mu, axis=0))
    if nr_batch == 1:
        return

    sum_dy = <weight_t*>calloc(nr_out, sizeof(weight_t))
    sum_dy_x_mu = <weight_t*>calloc(nr_out, sizeof(weight_t))
    Ex = <weight_t*>calloc(nr_out, sizeof(weight_t))
    Vx = <weight_t*>calloc(nr_out, sizeof(weight_t))
    Mat.mean_row(Ex, _x, nr_batch, nr_out)
    Mat.var_row(Vx, _x, Ex, nr_batch, nr_out, 0.0)
    for i from 0 <= i < (nr_batch * nr_out) by nr_out:
        for j in range(nr_out):
            sum_dy[j] += diff[i+j]
            sum_dy_x_mu[j] += diff[i+j] * (_x[i+j] - Ex[j])
    for i from 0 <= i < (nr_batch * nr_out) by nr_out:
        Vec.mul_i(&diff[i], nr_batch, nr_out)
        VecVec.add_i(&diff[i], sum_dy, -1., nr_out)
        for j in range(nr_out):
            diff[i+j] -= (_x[i+j] - Ex[j]) * Vx[j] ** -1. * sum_dy_x_mu[j]
    Vec.mul_i(diff, 1. / nr_batch, nr_batch * nr_out)
    for i in range(nr_out):
        est_mean[i] = (0.9 * est_mean[i]) + (0.1 * Ex[i])
        est_var[i] = (0.9 * est_var[i]) + (0.1 * Vx[i])
    free(Ex)
    free(Vx)
    free(sum_dy)
    free(sum_dy_x_mu)


# Optimization functions

@cython.cdivision(True)
cdef void clip_gradient(weight_t* gradient, weight_t threshold, int nr_weight) nogil:
    # Clip gradient
    grad_norm = Vec.norm(gradient, nr_weight)
    if grad_norm >= threshold:
        Vec.mul_i(gradient, threshold / grad_norm, nr_weight)


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
cdef void nag(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    '''
    Update weights with SGD and Nesterov momentum
    '''
    clip_gradient(gradient,
        100.0, nr_weight)
    
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
    VecVec.add_i(weights,
        momentum, -hp.m, nr_weight)
    Vec.mul_i(momentum,
        hp.m, nr_weight)
    VecVec.add_i(momentum,
        gradient, -hp.e, nr_weight)
    VecVec.add_i(weights,
        momentum, 1+hp.m, nr_weight)

    memset(gradient,
        0, sizeof(gradient[0]) * nr_weight)
    update_averages(weights+nr_weight,
        weights, nr_weight, hp.t)


@cython.cdivision(True)
cdef void adam(weight_t* weights, weight_t* gradient,
        len_t nr_weight, const ConstantsC* hp) nogil:
    clip_gradient(gradient,
        100.0, nr_weight)
 
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
