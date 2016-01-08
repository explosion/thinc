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
from .eg cimport Example
from .structs cimport ExampleC, OptimizerC, MapC
from .funcs cimport NN

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
        self.eg = Example(self.widths)
        NN.init(&self.c, mem, widths, eta, eps, mu, rho, bias, alpha)

    def __call__(self, features):
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        NN.predict_examples(&eg.c, 1,
            &self.c)
        return eg
   
    def train(self, features, y):
        memset(self.c.gradient,
            0, sizeof(self.c.gradient[0]) * self.c.nr_weight)
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        eg.set_label(y)

        NN.predict_example(&eg.c, &self.c)
        NN.insert_embeddings(self.c.embeds, self.mem,
            &eg.c)
        NN.insert_embeddings(self.c.opt.embed_params, self.mem,
            &eg.c)
        NN.update_dense(&self.c, self.c.gradient, &eg.c)
        NN.update_sparse(&self.c, self.c.gradient, &eg.c)
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




@cython.cdivision(True)
cdef void __tmp(OptimizerC* opt, weight_t* moments, weight_t* weights,
        weight_t* gradient, weight_t scale, int nr_weight) nogil:
    cdef weight_t beta1 = 0.90
    cdef weight_t beta2 = 0.999
    cdef weight_t EPS = 1e-6
    Vec.mul_i(gradient,
        scale, nr_weight)
    # Add the derivative of the L2-loss to the gradient
    cdef int i
    if opt.rho != 0:
        VecVec.add_i(gradient,
            weights, opt.rho, nr_weight)
    # This is all vectorized and in-place, so it's hard to read. See the
    # paper.
    mom1 = moments
    mom2 = &moments[nr_weight]
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)
    Vec.mul_i(mom2,
        beta2, nr_weight)
    VecVec.mul_i(gradient,
        gradient, nr_weight)
    VecVec.add_i(mom2,
        gradient, 1-beta2, nr_weight)
    Vec.div(gradient,
        mom1, 1-beta1, nr_weight)
    for i in range(nr_weight):
        gradient[i] /= sqrtf(mom2[i] / (1-beta2)) + EPS
    Vec.mul_i(gradient,
        opt.eta, nr_weight)
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)


