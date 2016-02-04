# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrtf, expf


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


cdef void d_ELU__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil:
    d_ELU(bwd[1],
        fwd[1], shape[1])
    # Set the gradient for F(W * fwd[0]) 
    MatMat.add_outer_i(gradient,
        bwd[1], fwd[0], shape[1], shape[0])
    VecVec.add_i(gradient + shape[1] * shape[0],
        bwd[1], 1.0, shape[1])
    # Set the partial derivative for bwd[0], so next step can set its gradient
    MatVec.T_dot(bwd[0],
        W, bwd[1], shape[1], shape[0])


cdef void d_ReLu__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil:
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


cdef void d_ELU__dot__normalize__dot(float* gradient, float** bwd, float* averages,
        const float* W, const float* const* fwd, const len_t* shape,
        int nr_above, int nr_below, const ConstantsC* hp) nogil:
    # D{ELU(BN(Lin(x)))} = ELU'(BN(Lin(x))) * BN'(Lin(x)) * Lin'(x)
    d_ELU(bwd[1],
        fwd[1], shape[1])
    # At this point we have what the paper refers to as dE/dY. Set the gradient
    # for the bias and gamma params now.
    bias = W + (shape[1] * shape[0])
    gamma = bias + shape[1]
    x_norm = fwd[1] + shape[1]
    gamma_grad = gradient + (shape[1] * shape[0]) + shape[1]
    for i in range(shape[1]):
        gamma_grad[i] += bwd[1][i] * x_norm[i]
    VecVec.add_i(gradient + (shape[1] * shape[0]),
        bwd[1], 1.0, shape[1])
    # Now continue computing BN'. We transform dE/dY into dE/dX'.
    # We have to transform it into dE/dX, so that we can calculate the gradient
    # for our weights.
    VecVec.mul_i(bwd[1],
        gamma, shape[1])
    # Read the E(x), Var(x), E_dXh, E_dXh_dot_Xh estimates from 'averages'
    cdef const float* Ex = averages
    cdef const float* Vx = averages + shape[1]
    cdef float* E_dXh = averages + shape[1] * 2
    cdef float* E_dXh_Xh = averages + shape[1] * 3
    d_normalize(bwd[1], E_dXh, E_dXh_Xh,
        x_norm, Vx, shape[1], hp.a, hp.t)
    # Finally we have dE/dX. Now we can calculate the gradient of W
    MatMat.add_outer_i(gradient,
        bwd[1], fwd[0], shape[1], shape[0])
    # And calculate the error w.r.t the previous layer
    MatVec.T_dot(bwd[0],
        W, bwd[1], shape[1], shape[0])


cdef void d_dot(float* btm_diff,
        int nr_btm,
        const float* top_diff, int nr_top,
        const float* W) nogil:
    # And calculate the error w.r.t the previous layer
    MatVec.T_dot(btm_diff,
        W, top_diff, nr_top, nr_btm)
 

cdef void d_ELU(float* delta, const float* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] *= signal_out[i] + ALPHA



cdef void d_ReLu(float* delta, const float* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] = 0


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


cdef void d_normalize(float* bwd, float* E_dEdXh, float* E_dEdXh_dot_Xh,
        const float* Xh, const float* Vx, len_t n, float alpha, float time) nogil:
    # Update EMA estimate of mean(dL/dX_hat)
    Vec.mul_i(E_dEdXh,
        alpha, n)
    VecVec.add_i(E_dEdXh,
        bwd, 1-alpha, n)
    # Update EMA estimate of mean(dE/dX_hat \cdot X_hat)
    Vec.mul_i(E_dEdXh_dot_Xh,
        alpha, n)
    for i in range(n):
        E_dEdXh_dot_Xh[i] += (1-alpha) * bwd[i] * Xh[i]
    # Simplification taken from Caffe, I think by cdoersch
    # if X' = (X-mean(X))/sqrt(var(X)+eps), then
    # dE/dX =
    #   (dE/dXh - mean(dE/dXh) - mean(dE/dXh * Xh) * Xh)
    #     ./ sqrt(var(X) + eps)
    # bwd is dE/dXh to start with. We change it to dE/dX in-place.
    if time >= 100:
        for i in range(n):
            bwd[i] -= E_dEdXh[i] - E_dEdXh_dot_Xh[i] * Xh[i]
            bwd[i] /= sqrtf(Vx[i] + EPS)
