# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memset, memcpy
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint64_t
cimport numpy as np
import numpy as np
from cpython.exc cimport PyErr_CheckSignals
from murmurhash.mrmr cimport hash64

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt, exp
from .weights cimport parse_weights, parse_batch_norm_weights
from .forward cimport skip_layer
from .forward cimport affine, normalize


np.import_array()


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0


cdef void ELU_backward(weight_t* gradient, weight_t** bwd,
        const weight_t* weights, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    '''Backward pass with ELU activation.

    Sequence of operations to compute the ith layer of activations
    
    x[i] = x[i-1] * W + b # Affine
    x[i] = ELU(x[i]) # ELU

    Arguments:
        fwd: array to write the forward activations
        widths: array of layer widths
        nr_layer: length of the widths array
        nr_batch: batch size
        hp: Hyper-parameters
    '''
    cdef int W
    cdef int bias
    for i in range(1, nr_layer-1):
        parse_weights(&W, &bias, 
            widths, i, nr_layer)

        if (i+1) < nr_layer:
            d_ELU(bwd[i],
                fwd[i], widths[i] * nr_batch)
        # Set the gradient for W
        MatMat.batch_add_outer_i(&gradient[W],
            bwd[i], fwd[i-1], widths[i], widths[i-1], nr_batch)
        # Set the gradient for bias
        VecVec.batch_add_i(&gradient[bias],
            bwd[i], 1.0, widths[i], nr_batch)
        # Set the gradient of fwd[i]
        MatVec.batch_T_dot(bwd[i-1],
            &weights[W], bwd[i], widths[i], widths[i-1], nr_batch)
 

cdef void ReLu_backward(weight_t* gradient, weight_t** bwd,
        const weight_t* weights, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    '''Backward pass with ELU activation.

    Sequence of operations to compute the ith layer of activations
    
    x[i] = x[i-1] * W + b # Affine
    x[i] = ELU(x[i]) # ELU

    Arguments:
        fwd: array to write the forward activations
        widths: array of layer widths
        nr_layer: length of the widths array
        nr_batch: batch size
        hp: Hyper-parameters
    '''
    cdef int W
    cdef int bias
    for i in range(1, nr_layer-1):
        parse_weights(&W, &bias, 
            widths, i, nr_layer)

        if (i+1) < nr_layer:
            d_ReLu(bwd[i],
                fwd[i], widths[i] * nr_batch)
        # Set the gradient for W
        MatMat.batch_add_outer_i(&gradient[W],
            bwd[i], fwd[i-1], widths[i], widths[i-1], nr_batch)
        # Set the gradient for bias
        VecVec.batch_add_i(&gradient[bias],
            bwd[i], 1.0, widths[i], nr_batch)
        # Set the gradient of fwd[i]
        MatVec.batch_T_dot(bwd[i-1],
            &gradient[W], bwd[i], widths[i], widths[i-1], nr_batch)
 
        
cdef void ELU_batch_norm_residual_backward(weight_t* gradient, weight_t** bwd,
        const weight_t* weights, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    cdef int W
    cdef int bias
    cdef int gamma
    cdef int beta
    cdef int mean
    cdef int variance

    x = <weight_t**>calloc(nr_layer, sizeof(void*))
    x_norm = <weight_t**>calloc(nr_layer, sizeof(void*))
    for i in range(nr_layer):
        x[i] = <weight_t*>calloc(widths[i] * nr_batch, sizeof(weight_t))
        x_norm[i] = <weight_t*>calloc(widths[i] * nr_batch, sizeof(weight_t))
    memcpy(x[0], fwd[0], sizeof(weight_t) * widths[0] * nr_batch)
    # Recalculate x and x_norm, for batchnorm
    for i in range(1, nr_layer-1):
        parse_batch_norm_weights(&W, &bias, &gamma, &beta, &mean, &variance,
            widths, i, nr_layer)

        affine(x[i],
            fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
        memcpy(x_norm[i],
            x[i], sizeof(weight_t) * nr_batch * widths[i])
        normalize(x_norm[i],
            &weights[mean], &weights[variance], widths[i], nr_batch)

    i = nr_layer-1
    parse_batch_norm_weights(&W, &bias, &gamma, &beta, &mean, &variance,
        widths, i, nr_layer)

    d_affine(bwd[i-1], gradient + W, gradient + bias,
        bwd[i], fwd[i-1],
        weights + W, widths[i], widths[i-1], nr_batch)

    for i in range(nr_layer-2, 0, -1):
        parse_batch_norm_weights(&W, &bias, &gamma, &beta, &mean, &variance,
            widths, i, nr_layer)

        d_ELU(bwd[i],
            fwd[i], widths[i] * nr_batch)
        d_residual(bwd,
            2, i, widths, nr_layer, nr_batch)
        d_transform(bwd[i], gradient + gamma, gradient + beta,
            x_norm[i], weights + gamma, widths[i], nr_batch)
        d_batchnorm(bwd[i], <weight_t*>&weights[mean], <weight_t*>&weights[variance],
            x[i], widths[i], nr_batch)
        d_affine(bwd[i-1], gradient + W, gradient + bias,
            bwd[i], fwd[i-1], weights + W, widths[i], widths[i-1], nr_batch)
        l2_regularize(gradient + W,
            weights + W, hp.r, widths[i] * widths[i-1])
    for i in range(nr_layer):
        free(x[i])
        free(x_norm[i])
    free(x)
    free(x_norm)


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


cdef void d_residual(weight_t** bwd, int skip, int i, const len_t* widths,
        int nr_layer, int nr_batch) nogil:
    if i < skip:
        return
    elif i >= nr_layer:
        # Error!
        return
    elif widths[i] != widths[i-skip]:
        return
    else:
        VecVec.add_i(bwd[i-skip],
            bwd[i], 1.0, widths[i-skip] * nr_batch)


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

    best = VecVec.arg_max_if_zero(scores, costs, nr_out)
    if best == -1:
        for i in range(nr_out):
            loss[i] = 1.0
    else:
        for i in range(nr_out):
            if costs[i] != 0 and scores[i] > (scores[best] + 1.0):
                loss[best] -= 1.0
                loss[i] = 1.0


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



@cython.cdivision(True)
cdef inline void l2_regularize(weight_t* gradient,
        const weight_t* weights, weight_t strength, int nr_weight) nogil:
    # Add the derivative of the L2-loss to the gradient
    if strength != 0:
        VecVec.add_i(gradient,
            weights, strength, nr_weight)


@cython.cdivision(True)
cdef inline void l1_regularize(weight_t* gradient,
        const weight_t* weights, weight_t cross,
        weight_t strength, int nr_weight) nogil:
    # Add the derivative of the L1-loss to the gradient
    if strength != 0:
        for i in range(nr_weight):
            if weights[i] > cross:
                gradient[i] += strength
            elif weights[i] < cross:
                gradient[i] -= strength
