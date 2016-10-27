# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memset, memcpy
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint64_t
from cpython.exc cimport PyErr_CheckSignals
from murmurhash.mrmr cimport hash64

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..structs cimport LayerC
from ..structs cimport SparseArrayC
from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt, exp
from .weights cimport parse_weights, parse_batch_norm_weights
#from .forward cimport skip_layer
from .forward cimport affine, ReLu
#normalize, layer_normalize


cdef weight_t EPS = 1e-5
DEF ALPHA = 1.0

cdef void ReLu_backward(LayerC* gradients, weight_t** bwd, 
        const LayerC* weights, const weight_t* const* fwd, const weight_t* randoms,
        const len_t* widths, int nr_layer, int nr_batch, const ConstantsC* hp) nogil:
    '''Backward pass'''

    for i in range(nr_layer-1, 0, -1):
        gradient = gradients[i]
        layer = weights[i]
        if i != nr_layer-1:
            d_ReLu(bwd[i],
                fwd[i], widths[i] * nr_batch)
        if gradient.sparse:
            d_affine_sparse(bwd[i-1], gradient.sparse, gradient.bias,
                bwd[i], fwd[i-1], layer.sparse, widths[i],
                widths[i-1], nr_batch)
        else:
            d_affine_dense(bwd[i-1], gradient.dense, gradient.bias,
                bwd[i], fwd[i-1], layer.dense, widths[i],
                widths[i-1], nr_batch)
        # d_dropout
        for j in range(widths[i-1] * nr_batch):
            if fwd[i-1][j] == 0:
                bwd[i-1][j] = 0
        #l2_regularize(gradient.dense,
        #    weights.dense, hp.r, widths[i] * widths[i-1])
 

cdef inline void d_ELU(weight_t* delta, const weight_t* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void d_ReLu(weight_t* delta, const weight_t* signal_out, int n) nogil:
    # Backprop the ReLu transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] <= 0:
            delta[i] = 0


cdef void d_softmax(weight_t* loss,
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
        const weight_t* costs, const weight_t* scores, len_t nr_out) nogil:
    cdef int best = -1
    cdef int guess = -1
    for i in range(nr_out):
        loss[i] = 0.0
        if costs[i] == 0:
            if best == -1 or scores[i] >= scores[best]:
                best = i
        elif costs[i] > 0:
            if guess == -1 or scores[i] >= scores[guess]:
                guess = i
    margin = (scores[guess] - scores[best]) + 1
    if margin > 0:
        loss[best] = -margin
        loss[guess] = margin


cdef void d_affine(weight_t* d_x, weights_ft d_w, weight_t* d_b,
        const weight_t* d_out, const weight_t* x, weights_ft w,
        int nr_out, int nr_in, int nr_batch) nogil:
    # Compute gradient for (sparse) weights matrix
    # Compute loss for next layer
    if weights_ft is dense_weights_t:
        d_affine_dense(d_x, d_w, d_b, d_out, x, w, nr_out, nr_in, nr_batch)
    else:
        d_affine_sparse(d_x, d_w, d_b, d_out, x, w, nr_out, nr_in, nr_batch)



cdef void d_affine_dense(weight_t* d_x, weight_t* d_w, weight_t* d_b,
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


cdef void d_affine_sparse(weight_t* d_x, SparseArrayC** d_w, weight_t* d_b,
        const weight_t* d_out, const weight_t* X, const SparseArrayC* const* w,
        int nr_out, int nr_in, int nr_batch) nogil:
    for i in range(nr_out):
        for j in range(nr_batch):
            weight_grads = d_w[i]
            weights = w[i]
            x = &X[j*nr_in]
            neuron_loss = d_out[j*nr_out+i]
            while weight_grads.key >= 0:
                weight_grads.val += x[weight_grads.key] * neuron_loss
                d_x[weights.key] += weights.val * neuron_loss
                weights += 1
                weight_grads += 1
    # Compute gradient for (dense) bias
    VecVec.batch_add_i(d_b,
        d_out, 1.0, nr_out, nr_batch)


@cython.cdivision(True)
cdef void l2_regularize(weight_t* gradient,
        const weight_t* weights, weight_t strength, int nr_weight) nogil:
    # Add the derivative of the L2-loss to the gradient
    if strength != 0:
        VecVec.add_i(gradient,
            weights, strength, nr_weight)


@cython.cdivision(True)
cdef void l1_regularize(weight_t* gradient,
        const weight_t* weights, weight_t cross,
        weight_t strength, int nr_weight) nogil:
    # Add the derivative of the L1-loss to the gradient
    if strength != 0:
        for i in range(nr_weight):
            if weights[i] > cross:
                gradient[i] += strength
            elif weights[i] < cross:
                gradient[i] -= strength


cdef const weight_t* d_dropout(weight_t* diff, 
        const weight_t* random_state, weight_t drop_prob, int nr) nogil:
    for i in range(nr):
        random_state -= 1
        if random_state[0] < drop_prob:
            diff[i] = 0
    return random_state
