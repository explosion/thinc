from cymem.cymem cimport Pool
from libc.string cimport memset

from .typedefs cimport weight_t, atom_t
from .structs cimport FeatureC, ExampleC, LayerC
from .features cimport Extracter 
from .model cimport Model
from .update cimport Updater


cdef int arg_max(const weight_t* scores, const int n_classes) nogil

cdef int arg_max_if_true(const weight_t* scores, const int* is_valid,
                         const int n_classes) nogil

cdef int arg_max_if_zero(const weight_t* scores, const weight_t* costs,
                         const int n_classes) nogil


cdef class Example:
    cdef Pool mem
    cdef ExampleC c

    @staticmethod
    cdef inline void init_classes(ExampleC* eg, Pool mem, int nr_class) except *:
        eg.nr_class = nr_class
        
        eg.scores = <weight_t*>mem.alloc(nr_class, sizeof(eg.scores[0]))
        
        eg.is_valid = <int*>mem.alloc(nr_class, sizeof(eg.is_valid[0]))
        memset(eg.is_valid, 1, sizeof(eg.is_valid[0]) * eg.nr_class)
        
        eg.costs = <weight_t*>mem.alloc(nr_class, sizeof(eg.costs[0]))

        eg.guess = 0
        eg.best = 0
        eg.cost = 0

    @staticmethod
    cdef inline void init_features(ExampleC* eg, Pool mem, int nr_atom, int nr_feat) except *:
        eg.nr_atom = nr_atom
        eg.nr_feat = nr_feat
        eg.atoms = <atom_t*>mem.alloc(nr_atom, sizeof(eg.atoms[0]))
        eg.features = <FeatureC*>mem.alloc(nr_feat, sizeof(eg.features[0]))

    @staticmethod
    cdef inline void init_nn_state(ExampleC* eg, Pool mem, const LayerC* layers,
                                   int nr_layer, int nr_dense, nr_class) except *:
        eg.fwd_state = <weight_t**>mem.alloc(nr_layer+1, sizeof(eg.fwd_state[0]))
        eg.bwd_state = <weight_t**>mem.alloc(nr_layer+1, sizeof(eg.bwd_state[0]))
        cdef int i
        for i in range(nr_layer):
            # Fwd state[i] is the incoming signal, so equal to layer size
            eg.fwd_state[i] = <weight_t*>mem.alloc(layers[i].nr_wide,
                                                   sizeof(eg.fwd_state[0][0]))
            # Bwd state[i] is the incoming error, so equal to layer width
            eg.bwd_state[i] = <weight_t*>mem.alloc(layers[i].nr_wide,
                                                   sizeof(eg.bwd_state[0][0]))
        eg.fwd_state[nr_layer] = <weight_t*>mem.alloc(nr_class,
                                                      sizeof(eg.fwd_state[0][0]))
        eg.bwd_state[nr_layer] = <weight_t*>mem.alloc(nr_class,
                                                      sizeof(eg.bwd_state[0][0]))
        eg.gradient = <weight_t*>mem.alloc(nr_dense, sizeof(eg.gradient[0]))


cdef class Learner:
    cdef readonly Extracter extracter
    cdef readonly Model model
    cdef readonly Updater updater
    cdef readonly int nr_class
    cdef readonly int nr_atom
    cdef readonly int nr_templ
    cdef readonly int nr_embed

    cdef void set_prediction(self, ExampleC* eg) except *

    cdef void set_costs(self, ExampleC* eg, int gold) except *

    cdef void update(self, ExampleC* eg) except *


cdef class AveragedPerceptron(Learner):
    pass
