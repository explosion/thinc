from cymem.cymem cimport Pool

from .typedefs cimport weight_t, atom_t
from .features cimport Feature


cdef struct ExampleC:
    int* is_valid
    int* costs
    atom_t* atoms
    Feature* features
    weight_t* embeddings
    weight_t* scores

    int nr_class
    int nr_atom
    int nr_feat
    int nr_embed
    
    int guess
    int best
    int cost
    weight_t loss


cdef class Example:
    cdef Pool mem
    cdef ExampleC c
    cdef int[:] is_valid
    cdef int[:] costs
    cdef atom_t[:] atoms
    cdef weight_t[:] embeddings
    cdef weight_t[:] scores

    @staticmethod
    cdef inline ExampleC init(Pool mem, int nr_class, int nr_atom,
            int nr_feat, int nr_embed) except *:
        return ExampleC(
            is_valid = <int*>mem.alloc(nr_class, sizeof(int)),
            costs = <int*>mem.alloc(nr_class, sizeof(int)),
            scores = <weight_t*>mem.alloc(nr_class, sizeof(weight_t)),
            atoms = <atom_t*>mem.alloc(nr_atom, sizeof(atom_t)),
            features = <Feature*>mem.alloc(nr_feat, sizeof(Feature)),
            embeddings = <weight_t*>mem.alloc(nr_embed, sizeof(weight_t)),
            nr_class = nr_class,
            nr_atom = nr_atom,
            nr_feat = nr_feat,
            nr_embed = nr_embed,
            guess = 0,
            best = 0,
            cost = 0,
            loss = 0)
