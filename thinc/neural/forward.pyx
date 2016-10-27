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
from ..structs cimport const_weights_ft, const_dense_weights_t, const_sparse_weights_t
from ..structs cimport weights_ft, dense_weights_t, sparse_weights_t

from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt, exp
from .weights cimport parse_weights, parse_batch_norm_weights


np.import_array()


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0

cdef void ReLu_forward(weight_t** fwd,
        const LayerC* weights, const weight_t* randoms, const len_t* widths,
        int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    '''Forward pass with ReLu activation'''
    for i in range(1, nr_layer):
        layer = weights[i]
        if layer.sparse:
            sparse_affine(fwd[i],
                fwd[i-1], layer.sparse, layer.bias,
                widths[i], widths[i-1], nr_batch)
        else:
            dense_affine(fwd[i],
                fwd[i-1], layer.dense, layer.bias,
                widths[i], widths[i-1], nr_batch)
        layer.activate(fwd[i],
            widths[i], nr_batch)


cdef void affine(weight_t* out,
        const weight_t* in_, const_weights_ft W, const weight_t* bias,
        int nr_out, int nr_in, int nr_batch) nogil:
    if const_weights_ft is const_dense_weights_t:
        dense_affine(out,
            in_, W, bias, nr_out, nr_in, nr_batch)
    else:
        sparse_affine(out,
            in_, W, bias, nr_out, nr_in, nr_batch)


cdef void dense_affine(weight_t* out,
        const weight_t* x, const weight_t* w, const weight_t* bias,
        int nr_out, int nr_in, int nr_batch) nogil:
    MatVec.batch_dot(out,
        w, x, nr_out, nr_in, nr_batch)
    MatVec.add_i(out,
        bias, 1.0, nr_batch, nr_out)


cdef void sparse_affine(weight_t* out,
        const weight_t* in_, const SparseArrayC* const* W, const weight_t* bias,
        int nr_out, int nr_in, int nr_batch) nogil:
    for i in range(nr_out):
        syn = W[i]
        while syn.key >= 0:
            out[i] += in_[syn.key] * syn.val
            syn += 1
       

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
    cdef float norm = Vec.sum(out, nr_out)
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
