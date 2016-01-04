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
                 weight_t rho=1e-4, weight_t bias=0.0, weight_t alpha=0.0):
        self.mem = Pool()
        self.c.alpha = alpha

        self.c.nr_layer = len(widths)
        self.c.widths = <int*>self.mem.alloc(self.c.nr_layer, sizeof(self.c.widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            self.c.widths[i] = width

        self.c.nr_weight = 0
        for i in range(self.c.nr_layer-1):
            self.c.nr_weight += NN.nr_weight(self.c.widths[i+1], self.c.widths[i])
        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))

        self.c.opt = <OptimizerC*>self.mem.alloc(1, sizeof(OptimizerC))
        Adagrad.init(self.c.opt, self.mem,
            self.c.nr_weight, self.c.widths, self.c.nr_layer, eta, eps, rho)

        if embed is not None:
            table_widths, features = embed
            self.c.embeds = <EmbeddingC*>self.mem.alloc(1, sizeof(EmbeddingC))
            Embedding.init(self.c.embeds, self.mem,
                table_widths, features)
            self.c.opt.embed_params = <EmbeddingC*>self.mem.alloc(1, sizeof(EmbeddingC))
            Embedding.init(self.c.opt.embed_params, self.mem,
                table_widths, features)
            for i in range(self.c.opt.embed_params.nr):
                # Ensure momentum terms start at zero
                memset(self.c.opt.embed_params.defaults[i],
                    0, sizeof(weight_t) * self.c.opt.embed_params.lengths[i])
        
        self.c.fwd_norms = <weight_t**>self.mem.alloc(self.c.nr_layer*2, sizeof(void*))
        self.c.bwd_norms = <weight_t**>self.mem.alloc(self.c.nr_layer*2, sizeof(void*))
        fan_in = 1.0
        cdef IteratorC it
        it.i = 0
        while NN.iter(&it, self.c.widths, self.c.nr_layer-1, 1):
            # Allocate arrays for the normalizers
            self.c.fwd_norms[it.Ex] = <weight_t*>self.mem.alloc(it.nr_out, sizeof(weight_t))
            self.c.fwd_norms[it.Vx] = <weight_t*>self.mem.alloc(it.nr_out, sizeof(weight_t))
            self.c.bwd_norms[it.E_dXh] = <weight_t*>self.mem.alloc(it.nr_out, sizeof(weight_t))
            self.c.bwd_norms[it.E_dXh_Xh] = <weight_t*>self.mem.alloc(it.nr_out, sizeof(weight_t))
            # Don't initialize the softmax weights
            if (it.i+1) >= self.c.nr_layer:
                break
            # Do He initialization, and allow bias to be initialized to a constant.
            # Initialize the batch-norm scale, gamma, to 1.
            Initializer.normal(&self.c.weights[it.W],
                0.0, numpy.sqrt(2.0 / fan_in), it.nr_out * it.nr_in)
            Initializer.constant(&self.c.weights[it.bias],
                bias, it.nr_out)
            Initializer.constant(&self.c.weights[it.gamma],
                1.0, it.nr_out)
            fan_in = it.nr_out

    def __call__(self, input_):
        cdef Example eg = self.Example(input_)
        NeuralNet.predictC(&eg.c, 1,
            &self.c)
        return eg
   
    def train(self, Xs, ys=None):
        cdef Batch mb = self.Batch(Xs, ys)
        NeuralNet.predictC(mb.c.egs,
            mb.c.nr_eg, &self.c)
        NeuralNet.insert_embeddingsC(self.c.embeds, self.mem,
            mb.c.egs, mb.c.nr_eg)
        NeuralNet.insert_embeddingsC(self.c.opt.embed_params, self.mem,
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
            assert len(weights) == self.c.nr_weight
            for i, weight in enumerate(weights):
                self.c.weights[i] = weight

    property layers:
        def __get__(self):
            weights = self.weights
            cdef IteratorC it
            it.i = 0
            while NN.iter(&it, self.c.widths, self.c.nr_layer-1, 1):
                yield (weights[it.W:it.bias], weights[it.bias:it.gamma])

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
