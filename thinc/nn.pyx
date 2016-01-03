# cython: profile=True
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
                 weight_t rho=1e-4, weight_t bias=0.2, weight_t alpha=0.0):
        self.mem = Pool()
        self.c.eta = eta
        self.c.eps = eps
        self.c.rho = rho
        self.c.alpha = alpha

        self.c.nr_layer = len(widths)
        self.c.widths = <int*>self.mem.alloc(self.c.nr_layer, sizeof(self.c.widths[0]))
        #cdef int i
        #for i, width in enumerate(widths):
        #    self.c.widths[i] = width

        #self.c.nr_weight = 0
        #for i in range(1, self.c.nr_layer):
        #    self.c.nr_weight += NeuralNet.nr_weight(self.c.widths[i], self.c.widths[i-1])
        #self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        #
        #if embed is not None:
        #    table_widths, features = embed
        #    self.c.embeds = <EmbeddingC*>self.mem.alloc(1, sizeof(EmbeddingC))
        #    Embedding.init(self.c.embeds, self.mem,
        #        table_widths, features)

        #self.c.opt = <OptimizerC*>self.mem.alloc(1, sizeof(OptimizerC))
        #Adagrad.init(self.c.opt, self.mem,
        #    self.c.nr_weight, self.c.widths, self.c.nr_layer, eta, eps, rho)

        #cdef weight_t* W = self.c.weights
        #fan_in = 1.0
        #for i in range(1, self.c.nr_layer-1): # Don't init softmax weights
        #    Initializer.normal(W,
        #        0.0, numpy.sqrt(2.0 / fan_in), self.c.widths[i] * self.c.widths[i-1])
        #    Initializer.constant(W + self.c.widths[i] * self.c.widths[i-1],
        #        bias, self.c.widths[i])
        #    fan_in = self.c.widths[i]

    def __call__(self, input_):
        cdef Example eg = self.Example(input_)
        NeuralNet.predictC(&eg.c, 1,
            &self.c)
        return eg
   
    def train(self, Xs, ys=None):
        cdef Batch mb = self.Batch(Xs, ys)
        NeuralNet.predictC(mb.c.egs,
            mb.c.nr_eg, &self.c)
        NeuralNet.insert_embeddingsC(&self.c, self.mem,
            mb.c.egs, mb.c.nr_eg)
        NeuralNet.updateC(&self.c, mb.c.gradient, mb.c.egs,
            mb.c.nr_eg)
        return mb
 
    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        return Example(self.widths, features=input_, label=label)

    def Batch(self, inputs, costs=None):
        if isinstance(inputs, Batch):
            return inputs
        return Batch(self.widths, inputs, costs, self.c.nr_weight)
 
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
