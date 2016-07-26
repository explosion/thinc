from libc.stdint cimport int16_t, int, int32_t, uint64_t
from preshed.maps cimport MapStruct
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memcpy, memset
from murmurhash.mrmr cimport real_hash64 as hash64

from .typedefs cimport len_t, idx_t, atom_t, weight_t
from .linalg cimport VecVec


include "compile_time_constants.pxi"

ctypedef vector[weight_t] vector_weight_t


ctypedef void (*do_update_t)(
    weight_t* weights,
    weight_t* gradient,
        len_t nr,
        const ConstantsC* hp,
) nogil


ctypedef void (*do_feed_fwd_t)(
    weight_t** fwd,
        const weight_t* W,
        const len_t* shape,
        int nr_layer,
        int nr_batch,
        const ConstantsC* hp
) nogil
 

ctypedef void (*do_feed_bwd_t)(
    weight_t* G,
    weight_t** bwd,
        const weight_t* W,
        const weight_t* const* fwd,
        const len_t* shape,
        int nr_layer,
        int nr_batch,
        const ConstantsC* hp
) nogil


# Alias this, so that it matches our naming scheme
ctypedef MapStruct MapC


cdef struct ConstantsC:
    weight_t a
    weight_t b
    weight_t c
    weight_t d
    weight_t e
    weight_t g
    weight_t h
    weight_t i
    weight_t j
    weight_t k
    weight_t l
    weight_t m
    weight_t n
    weight_t o
    weight_t p
    weight_t q
    weight_t r
    weight_t s
    weight_t t
    weight_t u
    weight_t w
    weight_t x
    weight_t y
    weight_t z


cdef struct EmbedC:
    MapC** weights
    MapC** gradients
    weight_t** defaults
    weight_t** d_defaults
    idx_t* offsets
    len_t* lengths
    len_t nr
    int nr_support


cdef struct NeuralNetC:
    do_feed_fwd_t feed_fwd
    do_feed_bwd_t feed_bwd
    do_update_t update

    len_t* widths
    weight_t* weights
    weight_t* gradient

    EmbedC* embed

    len_t nr_layer
    len_t nr_weight
    len_t nr_node

    ConstantsC hp


cdef extern from "stdlib.h":
    int posix_memalign(void **memptr, size_t alignment, size_t size) nogil
    void* valloc (size_t size) nogil


cdef cppclass ExampleC:
    int* is_valid
    weight_t* costs
    uint64_t* atoms
    FeatureC* features
    weight_t* scores

    weight_t** fwd_state
    weight_t** bwd_state
    int* widths

    int nr_class
    int nr_atom
    int nr_feat
    int nr_layer

    __init__(int nr_class=0, int nr_atom=0, int nr_feat=0, widths=None):
        if widths is None:
            widths = [nr_class]
        if nr_class == 0:
            nr_class = widths[-1]

        this.nr_class = nr_class
        this.nr_atom = nr_atom
        this.nr_feat = nr_feat
        this.nr_layer = len(widths)

        this.scores = <weight_t*>calloc(nr_class, sizeof(this.scores[0]))
        this.costs = <weight_t*>calloc(nr_class, sizeof(this.costs[0]))
        this.atoms = <atom_t*>calloc(nr_atom, sizeof(this.atoms[0]))
        this.features = <FeatureC*>calloc(nr_feat, sizeof(this.features[0]))
        
        this.is_valid = <int*>calloc(nr_class, sizeof(this.is_valid[0]))
        this.fill_is_valid(1)

        this.widths = <int*>calloc(len(widths), sizeof(this.widths[0]))
        this.fwd_state = <weight_t**>calloc(len(widths), sizeof(this.fwd_state[0]))
        this.bwd_state = <weight_t**>calloc(len(widths), sizeof(this.bwd_state[0]))
        for i, width in enumerate(widths):
            this.widths[i] = width
            this.fwd_state[i] = <weight_t*>calloc(width, sizeof(this.fwd_state[i][0]))
            this.bwd_state[i] = <weight_t*>calloc(width, sizeof(this.bwd_state[i][0]))
    
    __dealloc__() nogil:
        free(this.scores)
        free(this.costs)
        free(this.atoms)
        free(this.features)
        free(this.is_valid)
        for i in range(this.nr_layer):
            free(this.fwd_state[i])
            free(this.bwd_state[i])
        free(this.fwd_state)
        free(this.bwd_state)
        free(this.widths)

    int resize_atoms(int nr_atom) nogil:
        if nr_atom != this.nr_atom:
            this.atoms = <atom_t*>realloc(this.atoms,
                sizeof(this.atoms[0]) * nr_atom)
            this.nr_atom = nr_atom

    int resize_features(int nr_feat) nogil:
        if nr_feat != this.nr_feat:
            this.features = <FeatureC*>realloc(this.features,
                sizeof(this.features[0]) * nr_feat)
            this.nr_feat = nr_feat

    int fill_features(int value) nogil:
        for i in range(nr_feat):
            this.features[i].i = value
            this.features[i].key = value
            this.features[i].value = value

    int fill_atoms(atom_t value) nogil:
        for i in range(this.nr_atom):
            this.atoms[i] = value

    int fill_scores(weight_t value) nogil:
        for i in range(this.nr_class):
            this.scores[i] = value

    int fill_is_valid(int value) nogil:
        for i in range(this.nr_class):
            this.is_valid[i] = value
   
    int fill_costs(weight_t value) nogil:
        for i in range(this.nr_class):
            this.costs[i] = value

    int fill_state(weight_t value) nogil:
        for i in range(this.nr_layer):
            for j in range(this.widths[i]):
                this.fwd_state[i][j] = value
                this.bwd_state[i][j] = value
    
    void reset() nogil:
        this.fill_features(0)
        this.fill_atoms(0)
        this.fill_scores(0)
        this.fill_costs(0)
        this.fill_is_valid(1)
        this.fill_state(0)


cdef cppclass MinibatchC:
    weight_t** _fwd
    weight_t** _bwd
    
    FeatureC** _feats
    int* _nr_feat
    
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
        this._feats = <FeatureC**>calloc(batch_size, sizeof(void*))
        this._nr_feat = <int*>calloc(batch_size, sizeof(int))
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
        free(this._is_valid)
        free(this._costs)
        free(this.signatures)

    int nr_in() nogil:
        return this.widths[0]

    int nr_out() nogil:
        return this.widths[this.nr_layer - 1]

    int push_back(const void* feats, int nr_feat, int is_sparse,
            const weight_t* costs, const int* is_valid, uint64_t key=0) nogil:
        # Hash the features, to see if the batch has a matching input.
        # If it does, just update the gradient for it.
        if key != 0:
            for i in range(this.i):
                if this.signatures[i] == key:
                    VecVec.add_i(this.costs(i),
                        costs, 1.0, this.nr_out())
                    return 0
 
        if this.i < this.batch_size:
            this.signatures[this.i] = key
            if is_sparse:
                this._nr_feat[this.i] = nr_feat
                this._feats[this.i] = <FeatureC*>calloc(nr_feat, sizeof(FeatureC))
                memcpy(this._feats[this.i],
                    feats, nr_feat * sizeof(this._feats[this.i][0]))
            else:
                this._nr_feat[this.i] = 0
                this._feats[this.i] = NULL
                memcpy(this.fwd(this.i, 0),
                    feats, this.widths[0] * sizeof(weight_t))

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

    FeatureC* features(int i) nogil:
        return this._feats[i]

    int nr_feat(int i) nogil:
        return this._nr_feat[i]

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
        return VecVec.arg_max_if_true(this.scores(i), this.is_valid(i), this.nr_out())
    
    int best(int i) nogil:
        return VecVec.arg_max_if_zero(this.scores(i), this.costs(i), this.nr_out())



cdef packed struct SparseArrayC:
    int32_t key
    weight_t val


cdef struct FeatureC:
    int i
    uint64_t key
    weight_t value


cdef struct SparseAverageC:
    SparseArrayC* curr
    SparseArrayC* avgs
    SparseArrayC* times


cdef struct TemplateC:
    int[MAX_TEMPLATE_LEN] indices
    int length
    atom_t[MAX_TEMPLATE_LEN] atoms
