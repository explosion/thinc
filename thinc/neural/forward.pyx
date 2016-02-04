# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrtf, expf


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


cdef void dot_plus__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil:
    bias = W + shape[1] * shape[0]
    dot_plus(fwd[1],
        bias, shape[1], fwd[0], shape[0], W)
    # Apply non-linearity
    if nr_above >= 2:
        ELU(fwd[1],
            shape[1])
    else:
        softmax(fwd[1],
            shape[1])
 

cdef void dot_plus__ReLu(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil:
    bias = W + shape[1] * shape[0]
    dot_plus(fwd[1],
        bias, shape[1], fwd[0], shape[0], W)
    # Apply non-linearity
    if nr_above >= 2:
        ReLu(fwd[1],
            shape[1])
    else:
        softmax(fwd[1],
            shape[1])
 

cdef void dot_plus__residual__ELU(float** fwd, float* averages,
        const float* W, const len_t* shape, int nr_below, int nr_above,
        const ConstantsC* hp) nogil:
    bias = W + shape[1] * shape[0]
    dot_plus(fwd[1],
        bias, shape[1], fwd[0], shape[0], W)
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


cdef void dot_plus(float* out,
        const float* bias, len_t nr_out,
        const float* x, len_t nr_in,
        const float* W) nogil:
    MatVec.dot(out,
        W, x, nr_out, nr_in)
    cdef float one = 1.0
    if bias is not NULL:
        VecVec.add_i(out,
            bias, one, nr_out)


cdef void ELU(float* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = ALPHA * (expf(out[i]) - 1)


cdef void ReLu(float* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = 0


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
