# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.stdlib cimport rand
from libc.string cimport memset
cimport numpy as np
import numpy as np

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrt, exp


np.import_array()


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0


cdef void ELU_forward(weight_t** fwd,
        const weight_t* W, const len_t* widths, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil:
    for i in range(1, nr_layer):
        nr_in = widths[i-1]
        nr_out = widths[i]
        b = nr_out * nr_in

        affine(fwd[i],
            fwd[i-1], W, W+b, nr_out, nr_in, nr_batch)
 
        if (i+1) < nr_layer:
            ELU(fwd[i],
                nr_out * nr_batch)
        W += nr_out * nr_in + nr_out * 3
 
    scores = fwd[nr_layer - 1]
    nr_out = widths[nr_layer - 1]
    for _ in range(nr_batch):
        softmax(scores,
            nr_out)
        scores += nr_out


cdef void ReLu_forward(weight_t** fwd,
        const weight_t* W, const len_t* shape, int nr_below, int nr_above,
        int nr_batch, const ConstantsC* hp) nogil:
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
 

cdef void dot_plus(weight_t* out,
        const weight_t* bias, len_t nr_out,
        const weight_t* x, len_t nr_in,
        const weight_t* W) nogil:
    MatVec.dot(out,
        W, x, nr_out, nr_in)
    cdef weight_t one = 1.0
    if bias is not NULL:
        VecVec.add_i(out,
            bias, one, nr_out)


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


cdef void ReLu_batch_norm_forward(weight_t** fwd,
        const weight_t* W, const len_t* widths, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil:
    for i in range(1, nr_layer-1):
        nr_out = widths[i]
        nr_in = widths[i-1]
        b = nr_out * nr_in
        gamma = b + nr_out
        beta = gamma + nr_out
        mean = beta + nr_out
        variance = mean + nr_out

        affine(fwd[i],
            fwd[i-1], W, W+b, nr_out, nr_in, nr_batch)
        normalize(fwd[i],
            nr_out, nr_batch)
        transform(fwd[i],
            W+gamma, W+beta, nr_out, nr_batch)
        ReLu(fwd[i],
            nr_out * nr_batch)
        W += nr_out * nr_in + nr_out * 5

    i = nr_layer-1
    nr_out = widths[i]
    nr_in = widths[i-1]
    b = nr_out * nr_in
    
    affine(fwd[i],
        fwd[i-1], W, W+b, nr_out, nr_in, nr_batch)
    for j in range(nr_batch):
        softmax(fwd[i] + j * nr_out,
            nr_out)


cdef void affine(weight_t* out,
        const weight_t* x, const weight_t* w, const weight_t* bias,
        int nr_out, int nr_in, int nr_batch) nogil:
    MatVec.batch_dot(out,
        w, x, nr_out, nr_in, nr_batch)
    MatVec.add_i(out,
        bias, 1.0, nr_batch, nr_out)


cdef void transform(weight_t* x,
        const weight_t* gamma, const weight_t* beta, int nr_out, int nr_batch) nogil:
    MatVec.mul_i(x,
        gamma, nr_batch, nr_out)
    MatVec.add_i(x,
        beta, 1.0, nr_batch, nr_out)
 

cdef void normalize(weight_t* x, int nr_out, int nr_batch) nogil:
    if nr_batch == 1:
        return
    cdef weight_t[300] Ex
    memset(Ex, 0, sizeof(Ex))
    for i in range(nr_batch):
        VecVec.add_i(Ex, x + (i * nr_out), 1.0, nr_out)
    Vec.mul_i(Ex, 1.0 / nr_batch, nr_out)
    cdef weight_t[300] Vx
    memset(Vx, 0, sizeof(Vx))
    for i in range(nr_batch):
        VecVec.add_i(x + (i * nr_out), Ex, -1.0, nr_out)
        VecVec.add_pow_i(Vx, x + (i * nr_out), 2.0, nr_out)
    Vec.mul_i(Vx, 1.0 / nr_batch, nr_out)
    for i in range(nr_out):
        Vx[i] = 1. / sqrt(Vx[i] + EPS)
    MatVec.mul_i(x,
        Vx, nr_batch, nr_out)
