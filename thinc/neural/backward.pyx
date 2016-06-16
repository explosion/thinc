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

        W -= nr_out * nr_in + nr_out + nr_out
        G -= nr_out * nr_in + nr_out + nr_out

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


cdef void d_dot(weight_t* btm_diff,
        int nr_btm,
        const weight_t* top_diff, int nr_top,
        const weight_t* W) nogil:
    # And calculate the error w.r.t the previous layer
    MatVec.T_dot(btm_diff,
        W, top_diff, nr_top, nr_btm)
 

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
        

from .forward cimport normalize

cdef void ELU_batch_norm_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:

    affine_backward()
    for i in range(nr_layer-2, 0, -1):
        nr_out = widths[i]
        nr_in = widths[i-1]
        top_x = fwd[i]
        btm_x = fwd[i-1]
        top_d = bwd[i]
        btm_d = bwd[i-1]

        W -= nr_out * nr_in + nr_out + nr_out
        G -= nr_out * nr_in + nr_out + nr_out

        affine_norm_ELU_backward(G, btm_x,
            top_y, top_x, top_xnorm, W, nr_out, nr_in, nr_batch)


cdef void affine_norm_ELU_backward(weight_t* G, weight_t* d_x,
        weight_t* d_out,
        const weight_t* x, const weight_t* x_norm, const weight_t* W,
        int nr_out, int nr_in, int nr_batch):
    w = W
    b = w + nr_out * nr_in
    beta = b + nr_out
    gamma = beta + nr_out
    
    d_w = G
    d_b = d_w + (nr_out * nr_in)
    d_beta = d_b + nr_out
    d_gamma = d_beta + nr_out
    
    elu_backward(d_out,
        nr_batch, nr_out)

    d_xnorm = <weight_t*>calloc(nr_out * nr_batch, sizeof(weight_t))
    batchnorm_backward(d_xnorm, d_gamma, d_beta,
        d_out, gamma, x, x_norm, nr_out, nr_batch)

    affine_backward(d_x, d_w, d_b,
        d_xnorm, x, w, b, nr_out, nr_in, nr_batch)
    free(d_xnorm)
        

cdef void ELU_batch_norm_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    # Recompute x and x_norm. 
    x = <weight_t**>calloc(nr_layer, sizeof(void*))
    x_norm = <weight_t**>calloc(nr_layer, sizeof(void*))
    for i in range(nr_layer):
        x[i] = <weight_t*>calloc(widths[i] * nr_batch, sizeof(x[i][0]))
        x_norm[i] = <weight_t*>calloc(widths[i] * nr_batch, sizeof(x_norm[i][0]))
    
    norm_W = W
    for i in range(nr_layer-2, -1, -1):
        norm_W -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        MatVec.batch_dot(x[i+1],
            norm_W, fwd[i], widths[i+1], widths[i], nr_batch)
        memcpy(x_norm[i+1], x[i+1], sizeof(x_norm[i+1][0]) * widths[i+1] * nr_batch)
        with gil:
            normalize(x_norm[i+1],
                nr_batch, widths[i+1])
 
    for i in range(nr_layer-2, -1, -1):
        W -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        G -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        beta = W + widths[i+1] * widths[i]
        gamma = beta + widths[i+1]
        
        if i < (nr_layer-2):
            d_ReLu(bwd[i+1],
                fwd[i+1], widths[i+1] * nr_batch)
        # Set the gradient for beta
        VecVec.batch_add_i(G + widths[i+1] * widths[i],
            bwd[i+1], 1.0, widths[i+1], nr_batch)
        if i < (nr_layer-2):
            with gil:
                backward_batchnorm(bwd[i+1],
                    x[i+1], beta, gamma, widths[i+1], nr_batch)
        MatMat.batch_add_outer_i(G,
            bwd[i+1], fwd[i], widths[i+1], widths[i], nr_batch)
        # Set the partial derivative for bwd[0], so next step can set its gradient
        MatVec.batch_T_dot(bwd[i],
            W, bwd[i+1], widths[i+1], widths[i], nr_batch)
    for i in range(nr_layer):
        free(x[i])
        free(x_norm[i])
    free(x)
    free(x_norm)


cdef void backward_batchnorm(weight_t* _dout, const weight_t* _x, const weight_t* _beta,
        const weight_t* _gamma, int D, int N) except *:
    cdef np.npy_intp shape[2]
    shape[0] = N
    shape[1] = D
    dout = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, _dout)
    x = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <weight_t*>_x)
    beta = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, <weight_t*>_beta)
    gamma = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, <weight_t*>_gamma)

    var = x.var(0) + EPS
    x_min_mu = x - x.mean(0)
    
    dx = (1. / N) \
         * gamma \
         * var ** (-1. / 2.) \
         * (N * dout \
            - np.sum(dout, axis=0) \
            - x_min_mu \
              * var ** (-1.0) \
              * np.sum(dout * x_min_mu, axis=0))
    for i in range(N):
        for j in range(D):
            _dout[i * D + j] = dx[i, j]


#cdef void _d_batch_norm_layer(weight_t* gradient, weight_t* top_diff,
#        const weight_t* W, const weight_t* top_x, const weight_t* btm_x,
#        int nr_batch, int nr_above, int nr_below) nogil:
#    x = <weight_t*>calloc(nr_above * nr_batch, sizeof(weight_t))
#    MatVec.batch_dot(x,
#        W, btm_x, nr_above, nr_below, nr_batch)
#    with gil:
#        d_normalize(top_diff,
#            x, nr_above, nr_batch)
#    free(x)
 
    ## D{ELU(BN(Lin(x)))} = ELU'(BN(Lin(x))) * BN'(Lin(x)) * Lin'(x)
    ## Recover x_norm
    #x_norm = <weight_t*>calloc(nr_above * nr_batch, sizeof(weight_t))
    #MatVec.batch_dot(x_norm,
    #    W, btm_x, nr_above, nr_below, nr_batch)
    #with gil:
    #    normalize(x_norm,
    #        nr_batch, nr_above)

    #gamma = W + (nr_above * nr_below + nr_above)
    #for i in range(nr_batch):
    #    for j in range(nr_above):
    #        if gamma[j] != 0:
    #            x_norm[i * nr_above + j] /= gamma[j]
    ## At this point we have what the paper refers to as dE/dY. 
    ## Set the gradient for the gamma param now.
    #gamma_grad = gradient + (nr_above * nr_below + nr_above)
    #for i in range(nr_batch):
    #    for j in range(nr_above):
    #        idx = i * nr_above + j
    #        gamma_grad[j] += top_diff[idx] * x_norm[idx]
    ## Now continue computing BN'.
    ## Transform dE/dY into dE/dX_norm.
    #MatVec.mul_i(top_diff,
    #    gamma, nr_batch, nr_above)
    ## We have to transform it into dE/dX, so that we can calculate the gradient
    ## for our weights.
    ## Here's where it gets annoying --- we have to recompute x.
    ## We'll reuse the Xh memory.
    #cdef weight_t[300] Ex
    #cdef weight_t[300] Vx
    #memset(x_norm, 0, sizeof(x_norm[0]) * nr_batch * nr_above)
    #MatVec.batch_dot(x_norm,
    #    W, btm_x, nr_above, nr_below, nr_batch)
 
    #memset(Ex, 0, sizeof(weight_t) * 300)
    #memset(Vx, 0, sizeof(weight_t) * 300)
    #with gil:
    #    d_normalize(top_diff,
    #        x_norm, nr_above, nr_batch)
    #free(x_norm)
 

#cdef void d_normalize(weight_t* _delta, weight_t* _x, int n, int nr_batch) except *:
#    cdef weight_t[300] _dVx
#    cdef weight_t[300] _dEx
#    memset(_dVx, 0, sizeof(weight_t) * 300)
#    memset(_dEx, 0, sizeof(weight_t) * 300)
#    _bn_variance_delta(_dVx, _delta, _x, n, nr_batch)
#    _bn_mean_delta(_dEx, _dVx, _delta, _x, n, nr_batch)
#
#    cdef np.npy_intp shape[2]
#    shape[0] = nr_batch
#    shape[1] = n
#    cdef np.ndarray delta, x, dVx
#    delta = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, _delta)
#    x = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, _x)
#    dVx = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, _dVx)
#    dEx = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, _dEx)
#
#    mean = x.mean(0)
#    inv_sqrt_var = 1.0 / (np.sqrt(x.var(0)) + EPS)
#    for i in range(nr_batch):
#        delta[i] *= inv_sqrt_var
#        delta[i] += dVx * ((2 * (x[i] - mean)) / nr_batch)
#        delta[i] += dEx / nr_batch
#    for i in range(nr_batch):
#        for j in range(n):
#            _delta[i * n + j] = delta[i, j]
#

#cdef void _bn_variance_delta(weight_t* _dVx,
#        const weight_t* _delta, const weight_t* _x,
#        int n, int nr_batch) except *:
#    cdef np.npy_intp shape[2]
#    shape[0] = nr_batch
#    shape[1] = n
#    cdef np.ndarray delta, x, dVx
#    delta = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <weight_t*>_delta)
#    x = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <weight_t*>_x)
#    dVx = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, <weight_t*>_dVx)
# 
#    cdef np.ndarray mean = x.mean(0)
#    var = -0.5 * (x.var(0) + EPS) ** -1.5
#    for i in range(nr_batch):
#        dVx += delta[i] * (x[i] - mean) * var
#    for i in range(n):
#        _dVx[i] = dVx[i]
#
#
#cdef void _bn_mean_delta(weight_t* _dEx, weight_t* _dVx,
#        const weight_t* _delta, const weight_t* _x,
#        int n, int nr_batch) except *:
#    cdef np.npy_intp shape[2]
#    shape[0] = nr_batch
#    shape[1] = n
#    cdef np.ndarray delta, x, dVx, dEx
#    delta = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <weight_t*>_delta)
#    x = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, <weight_t*>_x)
#    dVx = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, <weight_t*>_dVx)
#    dEx = np.PyArray_SimpleNewFromData(1, &shape[1], np.NPY_DOUBLE, <weight_t*>_dEx)
#    
#    inv_sq_var = -1.0 / numpy.sqrt(x.var(0) + EPS)
#    for i in range(nr_batch):
#        dEx += delta[i] * inv_sq_var
#    mean = x.mean(0)
#    dEx += dVx * sum(-2.0 * (x[i] - mean) for i in range(nr_batch)) / nr_batch
#    for i in range(n):
#        _dEx[i] = dEx[i]
#
# 
#cdef void _get_dist_x(weight_t* x, weight_t* Ex, weight_t* Vx,
#        const weight_t* btm_x, const weight_t* W,
#        int nr_above, int nr_below, int nr_batch) nogil:
#    _get_mean(Ex, x, nr_above, nr_batch)
#    for i in range(nr_batch):
#        VecVec.add_i(x + (i * nr_above), Ex, -1.0, nr_above)
#    _get_variance(Vx, Ex, x, nr_above, nr_batch) 
#
#
#cdef void _get_mean(weight_t* Ex, const weight_t* x, int n, int nr_batch) nogil:
#    for i in range(nr_batch):
#        VecVec.add_i(Ex, x + (i * n), 1.0, n)
#    Vec.mul_i(Ex, 1.0 / nr_batch, n)
# 
#
#cdef void _get_variance(weight_t* Vx, const weight_t* Ex, const weight_t* dist_x,
#        int n, int nr_batch) nogil:
#    for i in range(nr_batch):
#        VecVec.add_pow_i(Vx, dist_x + (i * n), 2.0, n)
#    Vec.mul_i(Vx, 1.0 / nr_batch, n)
#
# 
#cdef void d_normalize(weight_t* d, weight_t* dist_x, const weight_t* Ex,
#        weight_t* Vx, int n, int nr_batch) nogil:
#    cdef weight_t[300] dVx
#    memset(dVx, 0, sizeof(dVx))
#    _bn_variance_delta(dVx,
#        d, dist_x, Vx, n, nr_batch)
#
#    cdef weight_t[300] dEx
#    memset(dEx, 0, sizeof(dEx))
#    _bn_mean_delta(dEx,
#        d, dist_x, Vx, dVx, n, nr_batch) 
#
#    for i in range(n):
#        Vx[i] = 1.0 / sqrt(Vx[i] + EPS)
#    MatVec.mul_i(d,
#        Vx, nr_batch, n)
#    MatVec.add_i(d,
#        dEx, 1.0 / nr_batch, nr_batch, n)
#    
#    Vec.mul_i(dist_x, 2.0, nr_batch * n)
#    Vec.mul_i(dist_x, 1.0 / nr_batch, nr_batch * n)
#    MatVec.mul_i(dist_x,
#        dVx, nr_batch, n)
#    VecVec.add_i(d, dist_x, 1.0, nr_batch * n)
#
#
#cdef void _bn_variance_delta(weight_t* dVx,
#        const weight_t* d, const weight_t* dist, const weight_t* Vx,
#        int n, int nr_batch) nogil:
#    for i in range(nr_batch):
#        for j in range(n):
#            idx = i * n + j
#            dVx[j] += d[idx] * dist[idx]
#    for i in range(n):
#        dVx[i] *= -0.5 * (Vx[i] + EPS) ** -1.5
#
#
#cdef void _bn_mean_delta(weight_t* dEx,
#        const weight_t* d, const weight_t* dist,
#        const weight_t* Vx, const weight_t* dVx,
#        int n, int nr_batch) nogil:
#    for i in range(nr_batch):
#        VecVec.add_i(dEx, d + (i * n), 1.0, n)
#    for i in range(n):
#        dEx[i] *= -1.0 / sqrt(Vx[i] + EPS)
#    for i in range(nr_batch):
#        for j in range(n):
#            dEx[j] += dVx[j] * (-2 * dist[i * n + j] * (1.0 / nr_batch))
