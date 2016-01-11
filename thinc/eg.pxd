from cymem.cymem cimport Pool
from libc.math cimport sqrt as c_sqrt
from libc.string cimport memset, memcpy, memmove

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get

from .structs cimport ExampleC, FeatureC, MapC
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
    cdef inline void init(ExampleC* self, Pool mem, model_shape,
                          blocks_per_layer) except *:
        self.fwd_state = <weight_t**>mem.alloc(len(model_shape), sizeof(void*))
        self.bwd_state = <weight_t**>mem.alloc(len(model_shape), sizeof(void*))
        self.widths = <int*>mem.alloc(len(model_shape), sizeof(int))
        for i, (width, nr_block) in enumerate(model_shape):
            self.widths[i] = width
            self.fwd_state[i] = <weight_t*>mem.alloc(width * blocks_per_layer,
                                                     sizeof(weight_t))
            self.bwd_state[i] = <weight_t*>mem.alloc(width * blocks_per_layer,
                                                     sizeof(weight_t))
        self.nr_layer = len(model_shape)
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
