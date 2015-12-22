from cymem.cymem cimport Pool
from libc.string cimport memset

from .typedefs cimport weight_t, atom_t
from .structs cimport FeatureC, ExampleC
from .features cimport Extracter 
from .model cimport Model
from .update cimport Updater


cdef int arg_max(const weight_t* scores, const int n_classes) nogil

cdef int arg_max_if_true(const weight_t* scores, const int* is_valid,
                         const int n_classes) nogil

cdef int arg_max_if_zero(const weight_t* scores, const weight_t* costs,
                         const int n_classes) nogil


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
