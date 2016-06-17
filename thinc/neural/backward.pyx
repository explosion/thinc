# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memset, memcpy
from libc.stdlib cimport calloc, free
cimport numpy as np
import numpy as np

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrt, exp


np.import_array()


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0


cdef void ELU_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    for i in range(nr_layer-1, 0, -1):
        nr_out = widths[i]
        nr_in = widths[i-1]
        top_x = fwd[i]
        btm_x = fwd[i-1]
        top_d = bwd[i]
        btm_d = bwd[i-1]

        W -= nr_out * nr_in + nr_out * 3
        G -= nr_out * nr_in + nr_out * 3

        if (i+1) < nr_layer:
            d_ELU(top_d,
                top_x, nr_out * nr_batch)
        # Set the gradient for W
        MatMat.batch_add_outer_i(G,
            top_d, btm_x, nr_out, nr_in, nr_batch)
        # Set the gradient for bias
        VecVec.batch_add_i(G + nr_out * nr_in,
            top_d, 1.0, nr_out, nr_batch)
        # Set the gradient of fwd[i]
        MatVec.batch_T_dot(btm_d,
            W, top_d, nr_out, nr_in, nr_batch)
    

cdef void ReLu_backward(weight_t* gradient, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* shape,
        int nr_above, int nr_below, int nr_batch, const ConstantsC* hp) nogil:
    d_ReLu(bwd[1],
        fwd[1], shape[1])
    # Set the gradient for F(W * fwd[0]) 
    MatMat.add_outer_i(gradient,
        bwd[1], fwd[0], shape[1], shape[0])
    VecVec.add_i(gradient + shape[1] * shape[0],
        bwd[1], 1.0, shape[1])
    # Set the partial derivative for bwd[0], so next step can set its gradient
    MatVec.T_dot(bwd[0],
        W, bwd[1], shape[1], shape[0])


cdef inline void d_ELU(weight_t* delta, const weight_t* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void d_ReLu(weight_t* delta, const weight_t* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] = 0


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
    const weight_t* costs, const weight_t* scores, len_t nr_out
) nogil:
    for i in range(nr_out):
        loss[i] = 0.0
    guess = Vec.arg_max(scores, nr_out)
    best = VecVec.arg_max_if_zero(scores, costs, nr_out)
    margin = scores[guess] - scores[best]
    loss[best] = -(margin * costs[guess])
    loss[guess] = (margin * costs[guess])


from .forward cimport affine, normalize
        

cdef void ELU_batch_norm_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:

    x = <weight_t**>calloc(nr_layer, sizeof(void*))
    x_norm = <weight_t**>calloc(nr_layer, sizeof(void*))
    for i in range(nr_layer):
        x[i] = <weight_t*>calloc(widths[i] * nr_batch, sizeof(weight_t))
        x_norm[i] = <weight_t*>calloc(widths[i] * nr_batch, sizeof(weight_t))
    memcpy(x[0], fwd[0], sizeof(weight_t) * widths[0] * nr_batch)

    i = nr_layer-1
    nr_out = widths[i]
    nr_in = widths[i-1]

    W -= nr_out * nr_in + nr_out * 5
    G -= nr_out * nr_in + nr_out * 5
    b = nr_out * nr_in
    gamma = b + nr_out
    beta = gamma + nr_out
    mean = beta + nr_out
    variance = mean + nr_out

    d_affine(bwd[i-1], G, G+b,
        bwd[i], fwd[i-1],
        W, nr_out, nr_in, nr_batch)

    for i in range(nr_layer-2, 0, -1):
        nr_out = widths[i]
        nr_in = widths[i-1]

        W -= nr_out * nr_in + nr_out * 5
        G -= nr_out * nr_in + nr_out * 5
        b = nr_out * nr_in
        gamma = b + nr_out
        beta = gamma + nr_out
        mean = beta + nr_out
        variance = mean + nr_out

        # Recalculate x and x_norm, for batchnorm
        affine(x[i],
            fwd[i-1], W, W+b, nr_out, nr_in, nr_batch)
        memcpy(x_norm[i], x[i], sizeof(weight_t) * nr_batch * nr_out)
        normalize(x_norm[i],
            W+mean, W+variance, nr_out, nr_batch)

        d_ELU(bwd[i],
            fwd[i], nr_out * nr_batch)
        d_transform(bwd[i], G + gamma, G + beta,
            x_norm[i], W + gamma, nr_out, nr_batch)
        with gil:
            d_batchnorm(bwd[i], <weight_t*>(W+mean), <weight_t*>(W+variance),
                x[i], nr_out, nr_batch)
        d_affine(bwd[i-1], G, G + b,
            bwd[i], fwd[i-1], W, nr_out, nr_in, nr_batch)
    for i in range(nr_layer):
        free(x[i])
        free(x_norm[i])
    free(x)
    free(x_norm)


cdef void d_batchnorm(weight_t* _dx, weight_t* est_mean, weight_t* est_var,
        const weight_t* _x, int nr_out, int nr_batch) except *:
    if nr_batch == 1:
        return
    dy = np.zeros(shape=(nr_batch, nr_out), dtype='float64')
    x = np.zeros(shape=(nr_batch, nr_out), dtype='float64')
    for i in range(nr_batch):
        for j in range(nr_out):
            dy[i, j] = _dx[i * nr_out + j]
            x[i, j] = _x[i * nr_out + j]
    mu = np.zeros(shape=(nr_out,), dtype='float64')
    var = np.zeros(shape=(nr_out,), dtype='float64')
    for i in range(nr_out):
        mu[i] = est_mean[i]
        var[i] = est_var[i]

    # Simplification by Clement Thorey, here:
    # http://cthorey.github.io./backpropagation/
    N = nr_batch
    D = nr_out
    inv_sqrt_var = var ** (-1. / 2.)
    inv_var = var ** -1.

    dx = (1. / N) \
       * inv_sqrt_var \
       * (N \
         * dy \
         - np.sum(dy, axis=0) \
         - (x - mu) \
           * inv_var \
           * np.sum(dy * (x - mu), axis=0))

    for i in range(nr_batch):
        for j in range(nr_out):
            _dx[i * nr_out + j] = dx[i, j]
    true_mu = x.mean(0)
    true_var = x.var(0)
    for i in range(nr_out):
        est_mean[i] = (0.9 * est_mean[i]) + (0.1 * true_mu[i])
        est_var[i] = (0.9 * est_var[i]) + (0.1 * true_var[i])


cdef void d_affine(weight_t* d_x, weight_t* d_w, weight_t* d_b,
        const weight_t* d_out, const weight_t* x, const weight_t* w,
        int nr_out, int nr_in, int nr_batch) nogil:
    # Set the gradient for W
    MatMat.batch_add_outer_i(d_w,
        d_out, x, nr_out, nr_in, nr_batch)
    # Set the gradient for bias
    VecVec.batch_add_i(d_b,
        d_out, 1.0, nr_out, nr_batch)
    # Set the gradient of fwd[i]
    MatVec.batch_T_dot(d_x,
        w, d_out, nr_out, nr_in, nr_batch)


cdef void d_transform(weight_t* d_x, weight_t* d_gamma, weight_t* d_beta,
        const weight_t* x_norm, const weight_t* gamma,
        int nr_out, int nr_batch) nogil:
    # Set the gradient for gamma
    for i in range(nr_batch):
        for j in range(nr_out):
            d_gamma[j] += x_norm[i * nr_out + j] * d_x[i * nr_out + j]
    # Set the gradient for beta
    VecVec.batch_add_i(d_beta,
        d_x, 1.0, nr_out, nr_batch)
    # Calculate d_x given d_y
    for i in range(nr_batch):
        VecVec.mul_i(d_x + i * nr_out,
            gamma, nr_out)
