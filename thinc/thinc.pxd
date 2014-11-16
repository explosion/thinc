from cymem.cymem cimport Pool

from .typedefs cimport *
from .learner cimport LinearModel
from .features cimport Extractor


cdef class Brain:
    cdef Pool mem
    cdef Extractor _extr
    cdef LinearModel _model
    cdef int n_atoms
    cdef int n_classes
    cdef class_t clas

    cdef feat_t* feats
    cdef weight_t* values
    cdef weight_t* scores

    cdef bint* _is_valid
    cdef bint* _is_gold
    
    cdef void score(self, weight_t* scores, atom_t* atoms)
    cdef class_t predict(self, atom_t* atoms) except 0
    cdef class_t predict_among(self, atom_t* atoms, list valid_classes) except 0
    cdef tuple learn(self, atom_t* atoms, list valid_classes, list gold_classes)
