# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.stdlib cimport rand
from libc.string cimport memset, memcpy
cimport numpy as np
import numpy as np
from libc.stdint cimport uint64_t
from murmurhash.mrmr cimport hash64

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec, sqrt, exp
from .weights cimport parse_weights, parse_batch_norm_weights


np.import_array()


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0


cdef void ELU_batch_norm_residual_forward(weight_t** fwd,
        const weight_t* weights, const len_t* widths, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil:
    '''Forward pass with ELU activation, using batch normalization and residual
    connections.

    Sequence of operations to compute the ith layer of activations
    
    x[i] = x[i-1] * W + b # Affine
    x[i] = normalize(x[i]) * gamma + beta # Batch norm and transform
    x[i] = ELU(x[i]) # ELU
    x[i] += x[i-2] # Residual, skip-level 2
    x[i] = ELU(x[i]) # ELU

    Arguments:
        fwd: array to write the forward activations
        weights_buffer: The weights data for all layers, as a contiguous array
        widths: array of layer widths
        nr_layer: length of the widths array
        nr_batch: batch size
        hp: Hyper-parameters
    '''
    cdef int W
    cdef int bias
    cdef int gamma
    cdef int beta
    cdef int mean
    cdef int variance
    for i in range(1, nr_layer-1):
        parse_batch_norm_weights(&W, &bias, &gamma, &beta, &mean, &variance,
            widths, i, nr_layer)

        affine(fwd[i],
            fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
        normalize(fwd[i],
            &weights[mean], &weights[variance], widths[i], nr_batch)
        transform(fwd[i],
            &weights[gamma], &weights[beta], widths[i], nr_batch)
        ELU(fwd[i],
            widths[i] * nr_batch)
        residual(fwd,
            2, i, widths, nr_layer, nr_batch)
        ELU(fwd[i],
            widths[i] * nr_batch)

    i = nr_layer-1
    parse_batch_norm_weights(&W, &bias, &gamma, &beta, &mean, &variance,
        widths, i, nr_layer)

    affine(fwd[i],
        fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
    for j in range(nr_batch):
        softmax(fwd[i] + j * widths[i],
            widths[i])


cdef void ELU_forward(weight_t** fwd,
        const weight_t* weights, const len_t* widths, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil:
    '''Forward pass with ELU activation.

    Sequence of operations to compute the ith layer of activations
    
    x[i] = x[i-1] * W + b # Affine
    x[i] = ELU(x[i]) # ELU

    Arguments:
        fwd: array to write the forward activations
        weights_buffer: The weights data for all layers, as a contiguous array
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

        affine(fwd[i],
            fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
        ELU(fwd[i],
            widths[i] * nr_batch)

    i = nr_layer-1
    parse_weights(&W, &bias, 
        widths, i, nr_layer)
 
    affine(fwd[i],
        fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
    for j in range(nr_batch):
        softmax(fwd[i] + j * widths[i],
            widths[i])


cdef void ReLu_forward(weight_t** fwd,
        const weight_t* weights, const len_t* widths, int nr_layer, int nr_batch,
        const ConstantsC* hp) nogil:
    '''Forward pass with ReLu activation.

    Sequence of operations to compute the ith layer of activations
    
    x[i] = x[i-1] * W + b # Affine
    x[i] = ReLu(x[i]) # ReLu

    Arguments:
        fwd: array to write the forward activations
        weights_buffer: The weights data for all layers, as a contiguous array
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

        affine(fwd[i],
            fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
        ReLu(fwd[i],
            widths[i] * nr_batch)

    i = nr_layer-1
    parse_weights(&W, &bias, 
        widths, i, nr_layer)
 
    affine(fwd[i],
        fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)
    for j in range(nr_batch):
        softmax(fwd[i] + j * widths[i],
            widths[i])


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
 

cdef void normalize(weight_t* x, const weight_t* est_Ex, const weight_t* est_Vx,
        int nr_out, int nr_batch) nogil:
    if nr_batch == 1:
        for i in range(nr_batch):
            for j in range(nr_out):
                x[i * nr_out + j] -= est_Ex[j]
                x[i * nr_out + j] /= sqrt(est_Vx[j] + EPS)
        return
    if nr_out > 300:
        # Error!
        return
    cdef weight_t[300] Ex
    cdef weight_t[300] Vx
    memset(Vx, 0, sizeof(Vx))
    memset(Ex, 0, sizeof(Ex))
    for i in range(nr_batch):
        VecVec.add_i(Ex, x + (i * nr_out), 1.0, nr_out)
    Vec.mul_i(Ex, 1.0 / nr_batch, nr_out)
    for i in range(nr_batch):
        VecVec.add_i(x + (i * nr_out), Ex, -1.0, nr_out)
        VecVec.add_pow_i(Vx, x + (i * nr_out), 2.0, nr_out)
    Vec.mul_i(Vx, 1.0 / nr_batch, nr_out)
    for i in range(nr_out):
        Vx[i] = 1. / sqrt(Vx[i] + EPS)
    MatVec.mul_i(x,
        Vx, nr_batch, nr_out)


cdef void residual(weight_t** fwd, int skip, int i, const len_t* widths,
        int nr_layer, int nr_batch) nogil:
    if i < skip:
        return
    elif i >= nr_layer:
        # Error!
        return
    elif widths[i] != widths[i-skip]:
        return
    else:
        VecVec.add_i(fwd[i],
            fwd[i-skip], 1.0, widths[i] * nr_batch)


cdef int skip_layer(weight_t timestep, uint64_t layer, int nr_in, int nr_out) nogil:
    if nr_in != nr_out:
        return False
    elif hash64(&timestep, sizeof(timestep), layer) % 2:
        return False
    else:
        return True


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
