# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_set as Map_set

from .structs cimport FeatureC
from .structs cimport ConstantsC

from .typedefs cimport len_t
from .typedefs cimport idx_t

from .blas cimport MatMat, MatVec, VecVec, Vec

from .structs cimport do_feed_fwd_t
from .structs cimport do_feed_bwd_t
from .structs cimport do_update_t


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


cdef void dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil:
    bias = W + shape[1] * shape[0]
    # Linear
    MatVec.dot(fwd[1],
        W, fwd[0], shape[1], shape[0])
    VecVec.add_i(fwd[1],
        bias, 1.0, shape[1])
    # Apply non-linearity
    if nr_above >= 2:
        ELU(fwd[1],
            shape[1])
    else:
        softmax(fwd[1],
            shape[1])
 

cdef void dot_plus__residual__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil:
    bias = W + shape[1] * shape[0]
    # Linear
    MatVec.dot(fwd[1],
        W, fwd[0], shape[1], shape[0])
    VecVec.add_i(fwd[1],
        bias, 1.0, shape[1])
    if nr_below >= 1 and shape[-1] == shape[1]:
        VecVec.add_i(fwd[1],
            fwd[-1], 1.0, shape[1])
    # Apply non-linearity
    if nr_above >= 2:
        ELU(fwd[1],
            shape[1])
    else:
        softmax(fwd[1],
            shape[1])


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


cdef void dot__normalize__dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_before, int nr_above,
        const ConstantsC* hp) nogil:
    # Read the bias and gamma terms from the weights data

    bias = W + shape[1] * shape[0]
    # Gamma is the normalization rescaling weights
    gamma = bias + shape[1]
    # Read the E(x) and Var(x) estimates from 'averages'
    Ex = averages
    Vx = &averages[shape[1]]
    # We write our output in fwd[1][0...n]
    # An imporant intermediary result is the batch normed activation, which
    # we compute in fwd[1][n...2n], and preserve for the backward pass.
    x_norm = fwd[1] + shape[1]

    MatVec.dot(fwd[1],
        W, fwd[0], shape[1], shape[0])
    normalize(x_norm, Ex, Vx,
        fwd[1], shape[1], hp.a, hp.t)
    VecVec.mul(fwd[1],
        x_norm, gamma, shape[1])
    VecVec.add_i(fwd[1],
        bias, 1.0, shape[1])
    # Apply non-linearity
    if nr_above >= 2:
        ELU(fwd[1],
            shape[1])
    else:
        softmax(fwd[1],
            shape[1])


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
   

cdef void ELU(float* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = ALPHA * (expf(out[i]) - 1)


cdef void d_ELU(float* delta, const float* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] < 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void softmax(float* out, len_t nr_out) nogil:
    #w = exp(w - max(w))
    Vec.add_i(out,
        -Vec.max(out, nr_out), nr_out)
    Vec.exp_i(out,
        nr_out)
    #w = w / sum(w)
    cdef float norm = Vec.sum(out, nr_out)
    if norm != 0:
        Vec.div_i(out,
            norm, nr_out)


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
 

@cython.cdivision(True)
cdef void vanilla_sgd_update_step(float* weights, float* moments, float* gradient,
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


cdef void normalize(float* x_norm, float* Ex, float* Vx,
        const float* x, len_t nr_x, float alpha, float time) nogil:
    # Upd EMA estimate of mean and variance
    # See eq at the end of this:
    # http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    cdef idx_t i
    cdef float diff
    cdef float incr
    cdef float one = 1.0
    for i in range(nr_x):
        diff = x[i] - Ex[i]
        incr = alpha * diff
        Vx[i] = (one - alpha) * (Vx[i] + diff * incr)
        Ex[i] += incr
    # Normalize
    if time < 100:
        for i in range(nr_x):
            x_norm[i] = x[i]
    else:
        for i in range(nr_x):
            if (x[i] - Ex[i]) == 0:
                x_norm[i] = 0
            else:
                x_norm[i] = (x[i] - Ex[i]) / sqrtf(Vx[i] + EPS)


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
