from libc.string cimport memset
from cymem.cymem cimport Pool

from .typedefs cimport weight_t, atom_t


cdef class Example:
    def __init__(self, int nr_class, int nr_atom, int nr_feat, int nr_embed):
        self.mem = Pool()

        self.c = Example.init(self.mem, nr_class, nr_atom, nr_feat, nr_embed)

        self.is_valid = <int[:nr_class]>self.c.is_valid
        self.costs = <int[:nr_class]>self.c.costs
        self.atoms = <atom_t[:nr_atom]>self.c.atoms
        self.embeddings = <weight_t[:nr_embed]>self.c.embeddings
        self.scores = <weight_t[:nr_class]>self.c.scores

    def wipe(self):
        cdef int i
        for i in range(self.c.nr_class):
            self.c.is_valid[i] = 0
            self.c.costs[i] = 0
            self.c.scores[i] = 0
        for i in range(self.c.nr_atom):
            self.c.atoms[i] = 0
        for i in range(self.c.nr_feat):
            self.c.embeddings[i] = 0
