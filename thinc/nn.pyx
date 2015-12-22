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
from .structs cimport ExampleC

import numpy


cdef class NeuralNet:
    def __init__(self, widths, weight_t eta=0.005, weight_t eps=1e-6, weight_t rho=1e-4):
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

        Map_init(self.mem, self.c.sparse, 8)

        cdef weight_t* W = self.c.weights
        for i in range(self.c.nr_layer-2): # Don't init softmax weights
            W = _init_layer_weights(W, self.c.widths[i+1], self.c.widths[i])

    def make_example(self, input_, label=None):
        cdef Example eg
        if isinstance(input_, Example):
            eg = input_
        else:
            eg = Example(self.nr_out)
        Example.init_nn(&eg.c, eg.mem, self.widths)
        #Example.set_dense(&eg.c, eg.mem, input_)
        #Example.set_label(&eg.c, eg.mem, label)
        return eg

    def make_batch(self, examples):
        if isinstance(examples, Batch):
            return examples
        return Batch(self.widths, examples)
   
    def predict(self, Example eg):
        with nogil:
            NeuralNet.forward(eg.c.fwd_state,
                self.c.weights, self.c.widths, self.c.nr_layer)
            Example.set_scores(&eg.c,
                eg.c.fwd_state[self.c.nr_layer-1])
        return eg

    def train(self, minibatch):
        cdef Batch mb = self.Batch(minibatch)
        with nogil:
            NeuralNet.trainC(&self.c, &mb.c)
            for i in range(mb.c.nr_eg):
                Example.set_scores(&mb.c.egs[i],
                    mb.c.egs[i].fwd_state[self.c.nr_layer-1])
        return mb

    def update(self, Batch mb):
        cdef int i
        # Get the averaged gradient for the minibatch
        for i in range(mb.c.nr_eg):
            eg = &mb.c.egs[i]
            NeuralNet.inc_gradient(mb.c.gradient,
                eg.fwd_state, eg.bwd_state, self.c.widths, self.model.c.nr_layer)
        # Vanilla SGD and L2 regularization (for now)
        VecVec.add_i(mb.c.gradient,
            self.c.weights, self.c.rho, self.c.nr_weight)
        VecVec.add_i(self.c.weights,
            mb.c.gradient, -self.c.eta, self.c.nr_weight)

        # Gather the per-feature gradient
        Batch.init_sparse_gradients(mb.c.sparse, mb.mem,
            mb.c.egs, mb.c.nr_eg)
        Batch.average_sparse_gradients(mb.c.sparse,
            mb.c.egs, mb.c.nr_eg)
        # Iterate over the sparse gradient, and update
        cdef feat_t key
        cdef void* addr
        i = 0
        while Map_iter(mb.c.sparse, &i, &key, &addr):
            feat_w = <weight_t*>Map_get(self.c.sparse, key)
            # This should never be null --- they should be preset.
            # Still, we check.
            if feat_w is not NULL and addr is not NULL:
                feat_g = <weight_t*>addr
                # Add the derivative of the L2-loss to the gradient
                VecVec.add_i(feat_g,
                    feat_w, self.c.rho, self.c.widths[0])
                # Vanilla SGD for now
                VecVec.add_i(feat_w,
                    feat_g, -self.c.eta, self.c.widths[0])



    def __call__(self, input_):
        return self.predict(self.make_example(input_))
    
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


#        # Pre-allocate and insert embedding vectors for any features we haven't
#        # seen. Do it now so we can release the GIL later.
#        for i in range(mb.c.nr_eg):
#            for j in range(mb.c.egs[i].nr_feat):
#                key = mb.c.egs[i].features[j].key
#                embed = <weight_t*>Map_get(&self.c.sparse_weights, key)
#                if embed is NULL:
#                    embed = <weight_t*>self.mem.alloc(self.c.widths[0], sizeof(weight_t))
#                    Map_set(self.mem, &self.c.sparse_weights,
#                        key, embed)
#
