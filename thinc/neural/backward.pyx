# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memset, memcpy
from libc.stdlib cimport calloc, free

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrt, exp


DEF EPS = 0.00001 
DEF ALPHA = 1.0


cdef void ELU_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    for i in range(nr_layer-2, -1, -1):
        W -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        G -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        if i < (nr_layer-2):
            d_ELU(bwd[i+1],
                fwd[i+1], widths[i+1] * nr_batch)
        # Set the gradient for F(W * fwd[0]) 
        MatMat.batch_add_outer_i(G,
            bwd[i+1], fwd[i], widths[i+1], widths[i], nr_batch)
        VecVec.batch_add_i(G + widths[i+1] * widths[i],
            bwd[i+1], 1.0, widths[i+1], nr_batch)
        # Set the partial derivative for bwd[0], so next step can set its gradient
        MatVec.batch_T_dot(bwd[i],
            W, bwd[i+1], widths[i+1], widths[i], nr_batch)
    

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
        

cdef void ELU_batch_norm_backward(weight_t* G, weight_t** bwd,
        const weight_t* W, const weight_t* const* fwd, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    for i in range(nr_layer-2, -1, -1):
        W -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        G -= widths[i+1] * widths[i] + widths[i+1] + widths[i+1]
        #if i < (nr_layer-2):
        #    d_ELU(bwd[i+1],
        #        fwd[i+1], widths[i+1] * nr_batch)
        # Set the gradient for the bias terms
        VecVec.batch_add_i(G + widths[i+1] * widths[i],
            bwd[i+1], 1.0, widths[i+1], nr_batch)
        # Now backprop through the batch normalisation layer, setting the gradient
        # for the gamma term
        _d_batch_norm_layer(G, bwd[i+1],
            W, fwd[i+1], fwd[i], nr_batch, widths[i+1], widths[i])
        MatMat.batch_add_outer_i(G,
            bwd[i+1], fwd[i], widths[i+1], widths[i], nr_batch)
        # Set the partial derivative for bwd[0], so next step can set its gradient
        MatVec.batch_T_dot(bwd[i],
            W, bwd[i+1], widths[i+1], widths[i], nr_batch)
 
 
cdef void _d_batch_norm_layer(weight_t* gradient, weight_t* top_diff,
        const weight_t* W, const weight_t* top_x, const weight_t* btm_x,
        int nr_batch, int nr_above, int nr_below) nogil:
    # D{ELU(BN(Lin(x)))} = ELU'(BN(Lin(x))) * BN'(Lin(x)) * Lin'(x)
    # Recover Xh, by unwinding the affine transform that makes Xh into "y"
    Xh = <weight_t*>calloc(nr_above * nr_batch, sizeof(weight_t))

    memcpy(Xh, top_x, sizeof(Xh[0]) * nr_batch * nr_above)
    #MatVec.add_i(Xh,
    #    W + nr_above * nr_below, -1.0, nr_batch, nr_above)
    #gamma = W + (nr_above * nr_below + nr_above)
    #for i in range(nr_batch):
    #    for j in range(nr_above):
    #        if gamma[j] != 0:
    #            Xh[i * nr_above + j] /= gamma[j]
    # At this point we have what the paper refers to as dE/dY. Set the gradient
    # for the gamma param now.
    #gamma_grad = gradient + (nr_above * nr_below + nr_above)
    #for i in range(nr_batch):
    #    for j in range(nr_above):
    #        idx = i * nr_above + j
    #        gamma_grad[j] += top_diff[idx] * Xh[idx]
    # Now continue computing BN'.
    # Transform dE/dY into dE/dX_norm.
    #MatVec.mul_i(top_diff,
    #    gamma, nr_batch, nr_above)
    # We have to transform it into dE/dX, so that we can calculate the gradient
    # for our weights.
    # Here's where it gets annoying --- we have to recompute x.
    # We'll reuse the Xh memory.
    cdef weight_t[300] Ex
    cdef weight_t[300] Vx
    memset(Xh, 0, sizeof(Xh[0]) * nr_batch * nr_above)
    memset(Ex, 0, sizeof(weight_t) * 300)
    memset(Vx, 0, sizeof(weight_t) * 300)
    _get_dist_x(Xh, Ex, Vx,
        btm_x, W, nr_above, nr_below, nr_batch)
    d_normalize(top_diff,
        Xh, Ex, Vx, nr_above, nr_batch)
    free(Xh)


cdef void _get_dist_x(weight_t* x, weight_t* Ex, weight_t* Vx,
        const weight_t* btm_x, const weight_t* W,
        int nr_above, int nr_below, int nr_batch) nogil:
    MatVec.batch_dot(x,
        W, btm_x, nr_above, nr_below, nr_batch)
    _get_mean(Ex, x, nr_above, nr_batch)
    for i in range(nr_batch):
        VecVec.add_i(x + (i * nr_above), Ex, -1.0, nr_above)
    _get_variance(Vx, Ex, x, nr_above, nr_batch) 


cdef void _get_mean(weight_t* Ex, const weight_t* x, int n, int nr_batch) nogil:
    for i in range(nr_batch):
        VecVec.add_i(Ex, x + (i * n), 1.0, n)
    Vec.mul_i(Ex, 1.0 / nr_batch, n)
 

cdef void _get_variance(weight_t* Vx, const weight_t* Ex, const weight_t* dist_x,
        int n, int nr_batch) nogil:
    for i in range(nr_batch):
        VecVec.add_pow_i(Vx, dist_x + (i * n), 2.0, n)
    Vec.mul_i(Vx, 1.0 / nr_batch, n)

 
cdef void d_normalize(weight_t* d, weight_t* dist_x, const weight_t* Ex,
        weight_t* Vx, int n, int nr_batch) nogil:
    cdef weight_t[300] dVx
    memset(dVx, 0, sizeof(dVx))
    _bn_variance_delta(dVx,
        d, dist_x, Vx, n, nr_batch)

    cdef weight_t[300] dEx
    memset(dEx, 0, sizeof(dEx))
    _bn_mean_delta(dEx,
        d, dist_x, Vx, dVx, n, nr_batch) 

    for i in range(n):
        Vx[i] = 1.0 / sqrt(Vx[i] + EPS)
    MatVec.mul_i(d,
        Vx, nr_batch, n)
    MatVec.add_i(d,
        dEx, 1.0 / nr_batch, nr_batch, n)
    
    Vec.mul_i(dist_x, 2.0, nr_batch * n)
    Vec.mul_i(dist_x, 1.0 / nr_batch, nr_batch * n)
    MatVec.mul_i(dist_x,
        dVx, nr_batch, n)
    VecVec.add_i(d, dist_x, 1.0, nr_batch * n)


cdef void _bn_variance_delta(weight_t* dVx,
        const weight_t* d, const weight_t* dist, const weight_t* Vx,
        int n, int nr_batch) nogil:
    for i in range(nr_batch):
        for j in range(n):
            idx = i * n + j
            dVx[j] += d[idx] * dist[idx]
    for i in range(n):
        dVx[i] *= -0.5 * (Vx[i] + EPS) ** -1.5


cdef void _bn_mean_delta(weight_t* dEx,
        const weight_t* d, const weight_t* dist,
        const weight_t* Vx, const weight_t* dVx,
        int n, int nr_batch) nogil:
    for i in range(nr_batch):
        VecVec.add_i(dEx, d + (i * n), 1.0, n)
    for i in range(n):
        dEx[i] *= -1.0 / sqrt(Vx[i] + EPS)
    for i in range(nr_batch):
        for j in range(n):
            dEx[j] += dVx[j] * (-2 * dist[i * n + j] * (1.0 / nr_batch))
