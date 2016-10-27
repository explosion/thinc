from libc.stdint cimport int16_t, int32_t, uint64_t
from preshed.maps cimport MapStruct
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memcpy, memset
from murmurhash.mrmr cimport real_hash64 as hash64

from .typedefs cimport len_t, idx_t, atom_t, weight_t
from . cimport prng


include "compile_time_constants.pxi"

ctypedef SparseArrayC** sparse_weights_t
ctypedef weight_t* dense_weights_t 
ctypedef const SparseArrayC* const* const_sparse_weights_t
ctypedef const weight_t* const_dense_weights_t 


ctypedef vector[weight_t] vector_weight_t


cdef fused weights_ft:
    sparse_weights_t
    dense_weights_t


cdef fused const_weights_ft:
    const_sparse_weights_t
    const_dense_weights_t


ctypedef void (*do_update_t)(
<<<<<<< HEAD
    float* weights,
    float* momentum,
    float* gradient,
=======
    weights_ft weights,
    weights_ft gradient,
>>>>>>> parser_nn_2016
        len_t nr,
        const ConstantsC* hp
) nogil


ctypedef void (*dense_update_t)(
    dense_weights_t weights,
    dense_weights_t gradient,
        len_t nr,
        const ConstantsC* hp
) nogil


ctypedef void (*do_activate_t)(weight_t* x, len_t nr_out, len_t nr_batch) nogil


ctypedef void (*do_feed_fwd_t)(
<<<<<<< HEAD
    float** fwd,
    float* averages,
        const float* W,
=======
    weight_t** fwd,
        const LayerC* W,
        const weight_t* randoms,
>>>>>>> parser_nn_2016
        const len_t* shape,
        int nr_layer,
        int nr_batch,
        const ConstantsC* hp
) nogil
 

ctypedef void (*do_feed_bwd_t)(
<<<<<<< HEAD
    float* G,
    float** bwd,
    float* averages,
        const float* W,
        const float* const* fwd,
=======
    LayerC* G,
    weight_t** bwd,
        const LayerC* W,
        const weight_t* const* fwd,
        const weight_t* randoms,
>>>>>>> parser_nn_2016
        const len_t* shape,
        int nr_layer,
        int nr_batch,
        const ConstantsC* hp
) nogil


# Alias this, so that it matches our naming scheme
ctypedef MapStruct MapC


cdef struct ConstantsC:
    float a
    float b
    float c
    float d
    float e
    float g
    float h
    float i
    float j
    float k
    float l
    float m
    float n
    float o
    float p
    float q
    float r
    float s
    float t
    float u
    float w
    float x
    float y
    float z


cdef struct EmbedC:
    MapC** weights
    MapC** gradients
    weight_t** defaults
    weight_t** d_defaults
    idx_t* offsets
    len_t* lengths
    len_t nr
    int nr_support


cdef struct LayerC:
    SparseArrayC** sparse
    weight_t* dense
    weight_t* bias
    do_activate_t activate


cdef struct NeuralNetC:
    do_feed_fwd_t feed_fwd
    do_feed_bwd_t feed_bwd
    
    int update

    len_t* widths
<<<<<<< HEAD
    float* weights
    float* gradient
    float* momentum

    float** averages
    
    EmbedC embed
=======
    weight_t* weights
    weight_t* gradient
    LayerC* layers
    LayerC* d_layers

    EmbedC* embed
>>>>>>> parser_nn_2016

    len_t nr_layer
    len_t nr_weight
    len_t nr_node

    ConstantsC hp


cdef extern from "stdlib.h":
    int posix_memalign(void **memptr, size_t alignment, size_t size) nogil
    void* valloc (size_t size) nogil


cdef struct ExampleC:
    int* is_valid
<<<<<<< HEAD
    float* costs
    uint64_t* atoms
    FeatureC* features
    float* scores

    float** fwd_state
    float** bwd_state
    int* widths

=======
    weight_t* costs
    atom_t* atoms
    void* features
    weight_t* scores

>>>>>>> parser_nn_2016
    int nr_class
    int nr_atom
    int nr_feat
    int is_sparse


cdef cppclass MinibatchC:
    weight_t** _fwd
    weight_t** _bwd
    
    void** _feats
    len_t* _nr_feat
    int* _is_sparse
    
    weight_t* _costs
    int* _is_valid
    uint64_t* signatures

    len_t* widths
    int i
    int nr_layer
    int batch_size

    __init__(len_t* widths, int nr_layer, int batch_size) nogil:
        this.i = 0
        this.nr_layer = nr_layer
        this.batch_size = batch_size
        this.widths = <len_t*>calloc(nr_layer, sizeof(len_t))
        this._fwd = <weight_t**>calloc(nr_layer, sizeof(weight_t*))
        this._bwd = <weight_t**>calloc(nr_layer, sizeof(weight_t*))
        for i in range(nr_layer):
            this.widths[i] = widths[i]
            this._fwd[i] = <weight_t*>calloc(this.widths[i] * batch_size, sizeof(weight_t))
            this._bwd[i] = <weight_t*>calloc(this.widths[i] * batch_size, sizeof(weight_t))
        this._feats = <void**>calloc(batch_size, sizeof(void*))
        this._nr_feat = <len_t*>calloc(batch_size, sizeof(len_t))
        this._is_sparse = <int*>calloc(batch_size, sizeof(int))
        this._is_valid = <int*>calloc(batch_size * widths[nr_layer-1], sizeof(int))
        this._costs = <weight_t*>calloc(batch_size * widths[nr_layer-1], sizeof(weight_t))
        this.signatures = <uint64_t*>calloc(batch_size, sizeof(uint64_t))

    __dealloc__() nogil:
        free(this.widths)
        for i in range(this.nr_layer):
            free(this._fwd[i])
            free(this._bwd[i])
        for i in range(this.i):
            free(this._feats[i])
        free(this._fwd)
        free(this._bwd)
        free(this._feats)
        free(this._nr_feat)
        free(this._is_sparse)
        free(this._is_valid)
        free(this._costs)
        free(this.signatures)

    void reset() nogil:
        for i in range(this.nr_layer):
            memset(this._fwd[i],
                0, sizeof(this._fwd[i][0]) * this.batch_size * this.widths[i])
            memset(this._bwd[i],
                0, sizeof(this._bwd[i][0]) * this.batch_size * this.widths[i])
        memset(this._nr_feat, 0, sizeof(this._nr_feat[0]) * this.batch_size)
        memset(this._is_sparse, 0, sizeof(this._is_sparse[0]) * this.batch_size)
        memset(this.signatures, 0, sizeof(this.signatures[0]) * this.batch_size)
        memset(this._costs,
            0, sizeof(this._costs[0]) * this.nr_out() * this.batch_size)
        memset(this._is_valid,
            0, sizeof(this._is_valid[0]) * this.nr_out() * this.batch_size)
        for i in range(this.i):
            free(this._feats[i])
            this._feats[i] = NULL
        this.i = 0

    int nr_in() nogil:
        return this.widths[0]

    int nr_out() nogil:
        return this.widths[this.nr_layer - 1]

    int push_back(const void* feats, int nr_feat, int is_sparse,
            const weight_t* costs, const int* is_valid, uint64_t key) nogil:
        if key != 0:
            for i in range(this.i):
                if this.signatures[i] == key:
                    my_costs = this.costs(i)
                    for j in range(this.nr_out()):
                        my_costs[j] += costs[j]
                    return 0
        if this.i >= this.batch_size:
            this.reset()
            this.i = 0 # This is done in reset() --- but make it obvious
 
        this.signatures[this.i] = key
        this._nr_feat[this.i] = nr_feat
        this._is_sparse[this.i] = is_sparse
        if is_sparse:
            this._feats[this.i] = calloc(nr_feat, sizeof(FeatureC))
            memcpy(this._feats[this.i],
                feats, nr_feat * sizeof(FeatureC))
        else:
            this._feats[this.i] = calloc(nr_feat, sizeof(weight_t))
            memcpy(this._feats[this.i],
                feats, nr_feat * sizeof(weight_t))

        memcpy(this.costs(this.i),
            costs, this.nr_out() * sizeof(costs[0]))
        if is_valid is not NULL:
            memcpy(this.is_valid(this.i),
                is_valid, this.nr_out() * sizeof(is_valid[0]))
        else:
            for i in range(this.nr_out()):
                this.is_valid(this.i)[i] = 1
        this.i += 1
        return this.i >= this.batch_size

    void* features(int i) nogil:
        return this._feats[i]

    int nr_feat(int i) nogil:
        return this._nr_feat[i]
    
    int is_sparse(int i) nogil:
        return this._is_sparse[i]

    weight_t* fwd(int i, int j) nogil:
        return this._fwd[i] + (j * this.widths[i])
 
    weight_t* bwd(int i, int j) nogil:
        return this._bwd[i] + (j * this.widths[i])
  
    weight_t* scores(int i) nogil:
        return this.fwd(this.nr_layer-1, i)

    weight_t* losses(int i) nogil:
        return this.bwd(this.nr_layer-1, i)
 
    weight_t* costs(int i) nogil:
        return this._costs + (i * this.nr_out())

    int* is_valid(int i) nogil:
        return this._is_valid + (i * this.nr_out())

    int guess(int i) nogil:
        # Don't use the linalg methods here to avoid circular import
        guess = -1
        for clas in range(this.nr_out()):
            if this.is_valid(i)[clas]:
                if guess == -1 or this.scores(i)[clas] >= this.scores(i)[guess]:
                    guess = clas
        return guess
    
    int best(int i) nogil:
        # Don't use the linalg methods here to avoid circular import
        best = -1
        for clas in range(this.nr_out()):
            if this.costs(i)[clas] == 0:
                if best == -1 or this.scores(i)[clas] >= this.scores(i)[best]:
                    best = clas
        return best
 


cdef packed struct SparseArrayC:
    int32_t key
    float val


cdef struct FeatureC:
    int i
    uint64_t key
    float value


cdef fused input_ft:
    FeatureC
    weight_t


cdef struct SparseAverageC:
    SparseArrayC* curr
    SparseArrayC* avgs
    SparseArrayC* times


cdef struct TemplateC:
    int[MAX_TEMPLATE_LEN] indices
    int length
    atom_t[MAX_TEMPLATE_LEN] atoms
