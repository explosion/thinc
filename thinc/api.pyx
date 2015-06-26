from libc.string cimport memset


cimport numpy as np
import numpy


cdef class Example:
    def __init__(self, int n_classes, int n_atoms, int n_embeddings):
        self.mem = Pool()

        self.n_classes = n_classes
        self.n_atoms = n_atoms
        
        #self.atoms    = <atom_t*>  self.mem.alloc(n_atoms,   sizeof(atom_t))
        self.atoms    = numpy.ndarray((n_atoms,), dtype=numpy.uint64)
        self.is_valid = numpy.ndarray((n_classes,), dtype=numpy.int32)
        self.costs = numpy.ndarray((n_classes,), dtype=numpy.int32)
        self.scores   = numpy.ndarray((n_classes,), dtype=numpy.float32)
        self.embeddings = numpy.ndarray((n_embeddings,), dtype=numpy.float32)

        self.guess = 0
        self.best = 0
        self.cost = 0
        self.loss = 0

    def wipe(self):
        self.atoms.fill(0)
        self.is_valid.fill(0)
        self.costs.fill(0)
        self.scores.fill(0)
        self.embeddings.fill(0)
