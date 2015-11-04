from libc.string cimport memset
from cymem.cymem cimport Pool

from .typedefs cimport weight_t, atom_t


cdef class Example:
    @classmethod
    def from_feats(cls, int nr_class, feats):
        nr_feat = len(feats)
        cdef Example self = cls(nr_class, nr_feat, nr_feat, nr_feat)
        for i, (key, value) in enumerate(feats):
            self.c.features[i].key = key
            self.c.features[i].value = value
        return self

    def __init__(self, int nr_class, int nr_atom, int nr_feat, int nr_embed):
        self.mem = Pool()
        self.c = Example.init(self.mem, nr_class, nr_atom, nr_feat, nr_embed)
        self.is_valid = <int[:nr_class]>self.c.is_valid
        self.costs = <int[:nr_class]>self.c.costs
        self.atoms = <atom_t[:nr_atom]>self.c.atoms
        self.embeddings = <weight_t[:nr_embed]>self.c.embeddings
        self.scores = <weight_t[:nr_class]>self.c.scores

    property guess:
        def __get__(self):
            return self.c.guess
        def __set__(self, int value):
            self.c.guess = value

    property best:
        def __get__(self):
            return self.c.best
        def __set__(self, int value):
            self.c.best = value
    
    property cost:
        def __get__(self):
            return self.c.cost
        def __set__(self, int value):
            self.c.cost = value
    
    property nr_class:
        def __get__(self):
            return self.c.nr_class
        def __set__(self, int value):
            self.c.nr_class = value
 
    property nr_atom:
        def __get__(self):
            return self.c.nr_atom
        def __set__(self, int value):
            self.c.nr_atom = value
 
    property nr_feat:
        def __get__(self):
            return self.c.nr_feat
        def __set__(self, int value):
            self.c.nr_feat = value
 
    property nr_embed:
        def __get__(self):
            return self.c.nr_embed
        def __set__(self, int value):
            self.c.nr_embed = value

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
