from cymem.cymem cimport Pool
from libc.math cimport sqrt as c_sqrt
from libc.string cimport memset, memcpy

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get

from .structs cimport ExampleC, BatchC, FeatureC, MapC
from .typedefs cimport feat_t, weight_t, atom_t
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
    cdef inline void init(ExampleC* self, Pool mem, model_shape) except *:
        self.fwd_state = <weight_t**>mem.alloc(len(model_shape) * 2, sizeof(void*))
        self.bwd_state = <weight_t**>mem.alloc(len(model_shape) * 2, sizeof(void*))
        i = 0
        for width in model_shape:
            self.fwd_state[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            self.fwd_state[i+1] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            self.bwd_state[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            self.bwd_state[i+1] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            i += 2
        # Each layer is x wide and connected to y nodes in the next layer.
        # So each layer has a weight matrix W with x*y weights, and an array
        # of bias weights, of length y. So each layer has x*y+y weights.
        self.fine_tune = <weight_t*>mem.alloc(model_shape[0], sizeof(weight_t))

        self.nr_class = model_shape[-1]
        self.scores = <weight_t*>mem.alloc(self.nr_class, sizeof(self.scores[0]))
        self.is_valid = <int*>mem.alloc(self.nr_class, sizeof(self.is_valid[0]))
        memset(self.is_valid, 1, sizeof(self.is_valid[0]) * self.nr_class)
        self.costs = <weight_t*>mem.alloc(self.nr_class, sizeof(self.costs[0]))

    @staticmethod
    cdef inline void init_dense(ExampleC* eg, Pool mem, dense_input) except *:
       cdef weight_t input_value
       for i, input_value in enumerate(dense_input):
           eg.fwd_state[0][i] = input_value

    @staticmethod
    cdef inline void set_scores(ExampleC* eg, const weight_t* scores) nogil:
        memcpy(eg.scores, scores, eg.nr_class * sizeof(weight_t))
        eg.guess = arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        eg.best = arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)


cdef class Batch:
    cdef Pool mem
    cdef BatchC c


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
