# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.stdlib cimport rand, calloc, free
from libc.string cimport memset, memcpy
cimport numpy as np
import numpy as np
from libc.stdint cimport uint64_t
from murmurhash.mrmr cimport hash64

from ..typedefs cimport len_t
from ..typedefs cimport idx_t
from ..structs cimport LayerC

from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt, exp
from .weights cimport parse_weights, parse_batch_norm_weights


np.import_array()


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0


cdef void ELU_forward(weight_t** fwd,
        const weight_t* weights, const weight_t* randoms, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
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
            widths[i], nr_batch)
    i = nr_layer-1
    parse_weights(&W, &bias, 
        widths, i, nr_layer)

    affine(fwd[i],
        fwd[i-1], &weights[W], &weights[bias], widths[i], widths[i-1], nr_batch)


cdef void ReLu_forward(weight_t** fwd,
        const LayerC* weights, const weight_t* randoms, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
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
    for i in range(nr_layer-1):
        layer = weights[i]

        if randoms is not NULL:
            randoms = dropout(fwd[i], randoms, hp.d, widths[i] * nr_batch)
        affine(fwd[i+1],
            fwd[i], layer.dense, layer.bias, widths[i+1], widths[i], nr_batch)
        layer.activate(fwd[i+1],
            widths[i+1], nr_batch)


cdef void affine(weight_t* out,
        const weight_t* x, const weight_t* w, const weight_t* bias,
        int nr_out, int nr_in, int nr_batch) nogil:
    MatVec.batch_dot(out,
        w, x, nr_out, nr_in, nr_batch)
    MatVec.add_i(out,
        bias, 1.0, nr_batch, nr_out)


cdef void ELU(weight_t* out, len_t nr_out, len_t nr_batch) nogil:
    cdef idx_t i
    for i in range(nr_out * nr_batch):
        if out[i] <= 0:
            out[i] = ALPHA * (exp(out[i]) - 1)


cdef void ReLu(weight_t* out, len_t nr_out, len_t nr_batch) nogil:
    cdef idx_t i
    for i in range(nr_out * nr_batch):
        if out[i] < 0:
            out[i] = 0


cdef void softmax(weight_t* out, len_t nr_out, len_t nr_batch) nogil:
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


cdef const weight_t* dropout(weight_t* x,
        const weight_t* random_state, weight_t drop_prob, int nr) nogil:
    for i in range(nr):
        if random_state[i] < drop_prob:
            x[i] = 0
        else:
            x[i] /= 1-drop_prob
    return &random_state[nr]
