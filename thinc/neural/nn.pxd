from thinc.structs cimport NeuralNetC
from thinc.extra.eg cimport Example
from cymem.cymem cimport Pool


cdef class NeuralNet:
    cdef readonly Pool mem
    cdef readonly Example eg
    cdef NeuralNetC c


