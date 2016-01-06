# cython: profile=True
from __future__ import print_function
cimport cython

from cymem.cymem cimport Pool
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t

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
                 weight_t mu=0.2, weight_t rho=1e-4, weight_t bias=0.0, weight_t alpha=0.0):
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
        self.c.gradient = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))

        self.c.opt = <OptimizerC*>self.mem.alloc(1, sizeof(OptimizerC))
        Adam.init(self.c.opt, self.mem,
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
        self.eg = Example(self.widths)

    def __call__(self, features):
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        NeuralNet.predictC(&eg.c, 1,
            &self.c)
        return eg
   
    def train(self, features, y):
        memset(self.c.gradient,
            0, sizeof(self.c.gradient[0]) * self.c.nr_weight)
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        eg.set_label(y)

        NeuralNet.predictC(&eg.c,
            1, &self.c)
        NeuralNet.insert_embeddingsC(self.c.embeds, self.mem,
            &eg.c, 1)
        Adadelta.insert_embeddings(self.c.opt.embed_params, self.mem,
            &eg.c, 1)
        NeuralNet.updateC(&self.c, self.c.gradient, &eg.c,
            1)
        return eg
 
    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        return Example(self.widths, input_, label)

    def Batch(self, inputs, labels=None):
        if isinstance(inputs, Batch):
            return inputs
        return Batch(self.widths, inputs, labels, self.c.nr_weight)
 
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

    property layer_l1s:
        def __get__(self):
            for W, bias in self.layers:
                w_l1 = sum(abs(w) for w in W) / len(W)
                bias_l1 = sum(abs(w) for w in W) / len(bias)
                yield w_l1, bias_l1

    property gradient:
        def __get__(self):
            return [self.c.gradient[i] for i in range(self.c.nr_weight)]

    property l1_gradient:
        def __get__(self):
            cdef int i
            cdef weight_t total = 0.0
            for i in range(self.c.nr_weight):
                if self.c.gradient[i] < 0:
                    total -= self.c.gradient[i]
                else:
                    total += self.c.gradient[i]
            return total / self.c.nr_weight

    property embeddings:
        def __get__(self):
            cdef int i = 0
            cdef int j = 0
            cdef int k = 0
            cdef key_t key
            cdef void* value
            for i in range(self.c.embeds.nr):
                j = 0
                while Map_iter(self.c.embeds.tables[i], &j, &key, &value):
                    emb = <weight_t*>value
                    yield key, [emb[k] for k in range(self.c.embeds.lengths[i])]

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
