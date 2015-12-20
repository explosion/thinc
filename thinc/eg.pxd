from cymem.cymem cimport Pool
from libc.string cimport memset

from .structs cimport ExampleC, BatchC, FeatureC
from .typedefs cimport weight_t, atom_t


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
 

cdef class Batch:
    cdef Pool mem
    cdef BatchC c
