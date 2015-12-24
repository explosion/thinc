from __future__ import print_function
cimport cython

from cymem.cymem cimport Pool
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter

from .typedefs cimport weight_t, atom_t, feat_t
from .blas cimport VecVec
from .eg cimport Example, Batch
from .structs cimport ExampleC, OptimizerC, MapC

import numpy


cdef class Embedding:
    def __init__(self, vector_widths, features, mem=None):
        if mem is None:
            mem = Pool()
        self.mem = mem
        self.c = <EmbeddingC*>self.mem.alloc(1, sizeof(EmbeddingC))
        Embedding.init(self.c, self.mem, vector_widths, features)


cdef class NeuralNet:
    def __init__(self, widths, embed=None, weight_t eta=0.005, weight_t eps=1e-6,
                 weight_t rho=1e-4):
        self.mem = Pool()
        self.c.eta = eta
        self.c.eps = eps
        self.c.rho = rho

        self.c.nr_layer = len(widths)
        self.c.widths = <int*>self.mem.alloc(self.c.nr_layer, sizeof(self.c.widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            self.c.widths[i] = width

        self.c.nr_weight = 0
        for i in range(self.c.nr_layer-1):
            self.c.nr_weight += self.c.widths[i+1] * self.c.widths[i] + self.c.widths[i+1]

        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        
        if embed is not None:
            table_widths, features = embed
            self.c.embeds = <EmbeddingC*>self.mem.alloc(1, sizeof(EmbeddingC))
            Embedding.init(self.c.embeds, self.mem,
                table_widths, features)

        self.c.opt = <OptimizerC*>self.mem.alloc(1, sizeof(OptimizerC))
        VanillaSGD.init(self.c.opt, self.mem,
            self.c.nr_weight, self.c.widths, self.c.nr_layer, eta, eps, rho)

        cdef weight_t* W = self.c.weights
        for i in range(self.c.nr_layer-2): # Don't init softmax weights
            W = _init_layer_weights(W, self.c.widths[i+1], self.c.widths[i])

    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        return Example(nn_shape=self.widths, features=input_, label=label)

    def Batch(self, inputs, costs=None):
        return Batch(self.widths, inputs, costs)
   
    def __call__(self, input_):
        cdef Example eg = self.Example(input_)
        NeuralNet.predictC(&eg.c,
            &self.c)
        return eg

    def train(self, Xs, ys):
        cdef Batch mb = self.Batch(Xs, ys)

        NeuralNet.trainC(&self.c, &mb.c)

        for i in range(mb.c.nr_eg):
            Example.set_scores(&mb.c.egs[i],
                mb.c.egs[i].fwd_state[self.c.nr_layer-1])
        return mb
    
    property weights:
        def __get__(self):
            return [self.c.weights[i] for i in range(self.c.nr_weight)]
        def __set__(self, weights):
            for i, weight in enumerate(weights):
                self.c.weights[i] = weight

    property widths:
        def __get__(self):
            return tuple(self.c.widths[i] for i in range(self.c.nr_layer))

    property nr_layer:
        def __get__(self):
            return self.c.nr_layer
    property nr_weight:
        def __get__(self):
            return self.c.nr_weight
    property nr_out:
        def __get__(self):
            return self.c.widths[self.c.nr_layer-1]
    property nr_in:
        def __get__(self):
            return self.c.widths[0]

    property eta:
        def __get__(self):
            return self.c.eta
        def __set__(self, eta):
            self.c.eta = eta
    property rho:
        def __get__(self):
            return self.c.rho
        def __set__(self, rho):
            self.c.rho = rho
    property eps:
        def __get__(self):
            return self.c.eps
        def __set__(self, eps):
            self.c.eps = eps


cdef weight_t* _init_layer_weights(weight_t* W, int nr_out, int nr_wide) except NULL:
    cdef int i
    std = numpy.sqrt(2.0) * numpy.sqrt(1.0 / nr_wide)
    values = numpy.random.normal(loc=0.0, scale=std, size=(nr_out * nr_wide))
    for i in range(nr_out * nr_wide):
        W[i] = values[i]
    W += nr_out * nr_wide
    for i in range(nr_out):
        W[i] = 0.2
    return W + nr_out
