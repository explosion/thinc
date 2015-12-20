from __future__ import print_function
cimport cython

from cymem.cymem cimport Pool
import numpy

from .typedefs cimport weight_t, atom_t, feat_t
from .blas cimport VecVec
from .eg cimport Example, Batch


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
        self.c.support = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))

        cdef weight_t* W = self.c.weights
        for i in range(self.c.nr_layer-2): # Don't init softmax weights
            W = _init_layer_weights(W, self.c.widths[i+1], self.c.widths[i])
   
    def __call__(self, input_):
        cdef Example eg = self.make_example(input_)

        NeuralNet.forward(eg.c.fwd_state,
            self.c.weights, self.c.widths, self.c.nr_layer)
        return eg

    def train(self, X_y_pairs):
        cdef Batch mb = self.make_batch(X_y_pairs)
        gradient = mb.c.egs[0].gradient
        # Compute the gradient
        for i in range(mb.c.nr):
            eg = &mb.c.egs[i]
            NeuralNet.forward_backward(gradient, eg.fwd_state, eg.bwd_state,
                eg.costs, &self.c)
        # L2 regularization
        VecVec.add_i(gradient,
            self.c.weights, self.c.rho, self.c.nr_weight)
        
        Adagrad.update(self.c.weights, gradient, self.c.support,
            self.c.nr_weight, self.c.eta, self.c.eps)
        return mb.loss

    def make_example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        return Example.for_dense_nn(self.widths, input_, label)

    def make_batch(self, inputs, labels=None):
        return Batch.for_dense_nn(self.widths, inputs, labels)

    property weights:
        def __get__(self):
            return [self.c.weights[i] for i in range(self.c.nr_weight)]
        def __set__(self, weights):
            for i, weight in enumerate(weights):
                self.c.weights[i] = weight
    property support:
        def __get__(self):
            return [self.c.support[i] for i in range(self.nr_weight)]
        def __set__(self, weights):
            for i, weight in enumerate(weights):
                self.c.support[i] = weight

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
