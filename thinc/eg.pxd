from cymem.cymem cimport Pool
from libc.string cimport memset, memcpy

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get

from .structs cimport ExampleC, BatchC, FeatureC, MapC
from .typedefs cimport weight_t, atom_t
from .blas cimport Vec, VecVec


cdef class Example:
    cdef Pool mem
    cdef ExampleC c

    @staticmethod
    cdef inline Example from_ptr(Pool mem, ExampleC* ptr):
        cdef Example eg = Example.__new__(Example)
        eg.mem = mem
        eg.c = ptr[0]
        return eg

    @staticmethod
    cdef inline void init_class(ExampleC* eg, Pool mem, int nr_class) except *:
        eg.nr_class = nr_class
        eg.scores = <weight_t*>mem.alloc(nr_class, sizeof(eg.scores[0]))
        eg.is_valid = <int*>mem.alloc(nr_class, sizeof(eg.is_valid[0]))
        memset(eg.is_valid, 1, sizeof(eg.is_valid[0]) * eg.nr_class)
        
        eg.costs = <weight_t*>mem.alloc(nr_class, sizeof(eg.costs[0]))

        eg.guess = 0
        eg.best = 0
        eg.cost = 0

    @staticmethod
    cdef inline void init_dense(ExampleC* eg, Pool mem, dense_input) except *:
       cdef weight_t input_value
       for i, input_value in enumerate(dense_input):
           eg.fwd_state[0][i] = input_value

    @staticmethod
    cdef inline void init_nn(ExampleC* eg, Pool mem, widths) except *:
        eg.fwd_state = <weight_t**>mem.alloc(len(widths), sizeof(void*))
        eg.bwd_state = <weight_t**>mem.alloc(len(widths), sizeof(void*))
        for i, width in enumerate(widths):
            eg.fwd_state[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            eg.bwd_state[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
        # Each layer is x wide and connected to y nodes in the next layer.
        # So each layer has a weight matrix W with x*y weights, and an array
        # of bias weights, of length y. So each layer has x*y+y weights.
        nr_weight = sum([x * y + y for x, y in zip(widths, widths[1:])])
        eg.gradient = <weight_t*>mem.alloc(nr_weight, sizeof(weight_t))
        eg.fine_tune = <weight_t*>mem.alloc(widths[0], sizeof(weight_t))

    @staticmethod
    cdef inline void set_scores(ExampleC* eg, const weight_t* scores) nogil:
        memcpy(eg.scores, scores, eg.nr_class * sizeof(weight_t))
        eg.guess = arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        eg.best = arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)


cdef class Batch:
    cdef Pool mem
    cdef BatchC c

    @staticmethod
    cdef inline void init_sparse_gradients(MapC* map_, Pool mem,
            const ExampleC* egs, int nr_eg) except *:
        cdef const ExampleC* eg
        cdef int i, j
        for i in range(nr_eg):
            eg = &egs[i]
            for j in range(eg.nr_feat):
                feat = eg.features[j]
                grad = Map_get(map_, feat.key)
                if grad is NULL:
                    grad = mem.alloc(feat.length, sizeof(weight_t))
                    Map_set(mem, map_,
                        feat.key, grad)
    @staticmethod
    cdef inline void average_gradients(weight_t* gradient,
            const ExampleC* egs, int nr_weight, int nr_eg) nogil:
        for i in range(nr_eg):
            VecVec.add_i(gradient, egs[i].gradient, 1.0, nr_weight)
        Vec.div_i(gradient, nr_eg, nr_weight)

    @staticmethod
    cdef inline void average_sparse_gradients(MapC* gradients,
            const ExampleC* egs, int nr_eg) nogil:
        cdef int i, j
        # Average the examples' 'fine tuning' gradients
        # First we collect the total values of each feature.
        cdef weight_t total = 0.0
        for i in range(nr_eg):
            for j in range(egs[i].nr_feat):
                total += egs[i].features[j].val
        for i in range(nr_eg):
            for j in range(egs[i].nr_feat):
                feat = egs[i].features[j]
                feat_grad = <weight_t*>Map_get(gradients, feat.key)
                if feat_grad is not NULL:
                    # egs[i].bwd_state[0] holds the delta (error) for the input
                    # for this example. We weight the example by the feature value's
                    # proportion of the total.
                    VecVec.add_i(feat_grad,
                        egs[i].bwd_state[0], feat.val / total, feat.length)
 

cdef inline int arg_max(const weight_t* scores, const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = scores[0]
    for i in range(1, n_classes):
        if scores[i] > mode:
            mode = scores[i]
            best = i
    return best


cdef inline int arg_max_if_true(const weight_t* scores, const int* is_valid,
                         const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = -900000
    for i in range(n_classes):
        if is_valid[i] and scores[i] > mode:
            mode = scores[i]
            best = i
    return best


cdef inline int arg_max_if_zero(const weight_t* scores, const weight_t* costs,
                         const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef weight_t mode = -900000
    for i in range(n_classes):
        if costs[i] == 0 and scores[i] > mode:
            mode = scores[i]
            best = i
    return best
