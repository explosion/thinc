# cython: profile=True
from __future__ import print_function

from libc.string cimport memset

cimport cython

from cymem.cymem cimport Pool
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t

from .typedefs cimport weight_t, atom_t, feat_t
from .typedefs cimport len_t, idx_t
from .blas cimport VecVec
from .structs cimport MapC
from .structs cimport NeuralNetC
from .structs cimport IteratorC
from .structs cimport ExampleC
from .structs cimport FeatureC

from .eg cimport Example

from .lvl0 cimport advance_iterator
from .lvl0 cimport forward
from .lvl0 cimport backward
from .lvl0 cimport set_input
from .lvl0 cimport insert_sparse
from .lvl0 cimport dense_update
from .lvl0 cimport sparse_update

import numpy


cdef class NN:
    @staticmethod
    cdef void init(
        NeuralNetC* nn,
        Pool mem,
            widths,
            float eta=0.005,
            float eps=1e-6,
            float mu=0.2,
            float rho=1e-4,
            float bias=0.0,
            float alpha=0.0
    ) except *:
        nn.hp.a = alpha
        nn.hp.b = bias
        nn.hp.r = rho
        nn.hp.m = mu
        nn.hp.e = eta
        nn.nr_layer = len(widths)
        nn.widths = <len_t*>mem.alloc(nn.nr_layer, sizeof(widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            nn.widths[i] = width

        nn.nr_weight = 0
        for i in range(nn.nr_layer-1):
            nn.nr_weight += NN.nr_weight(nn.widths[i+1], nn.widths[i])
        nn.weights = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.gradient = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.momentum = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.averages = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        
        nn.sparse_weights = <MapC**>mem.alloc(nn.nr_embed, sizeof(void*))
        nn.sparse_gradient = <MapC**>mem.alloc(nn.nr_embed, sizeof(void*))
        nn.sparse_momentum = <MapC**>mem.alloc(nn.nr_embed, sizeof(void*))
        nn.sparse_averages = <MapC**>mem.alloc(nn.nr_embed, sizeof(void*))

        nn.embed_offsets = <idx_t*>mem.alloc(nn.nr_embed, sizeof(nn.embed_offsets[0]))
        nn.embed_lengths = <len_t*>mem.alloc(nn.nr_embed, sizeof(nn.embed_offsets[0]))
        nn.embed_defaults = <float**>mem.alloc(nn.nr_embed, sizeof(nn.embed_offsets[0]))

        for i in range(nn.nr_embed):
            nn.embed_defaults[i] = <float*>mem.alloc(nn.embed_lengths[i],
                                                     sizeof(nn.embed_defaults[i][0]))
        
        cdef IteratorC it
        it.i = 0
        while advance_iterator(&it, nn.widths, nn.nr_layer-1, 1):
            # Allocate arrays for the normalizers
            # Don't initialize the softmax weights
            if (it.i+1) >= nn.nr_layer:
                break
            he_normal_initializer(&nn.weights[it.W],
                fan_in, it.nr_out * it.nr_in)
            constant_initializer(&nn.weights[it.bias],
                bias, it.nr_out)
            he_normal_initializer(&nn.weights[it.gamma],
               1, it.nr_out)
            fan_in = it.nr_out

    @staticmethod
    cdef int nr_weight(int nr_out, int nr_in) nogil:
        return nr_out * nr_in + nr_out * 3

    @staticmethod
    cdef void predict_example(ExampleC* eg, const NeuralNetC* nn) nogil:
        NN.forward(eg.scores, eg.fwd_state,
            eg.features, eg.nr_feat, nn)
        Example.set_scores(eg, eg.fwd_state[nn.nr_layer*2-2])

    @staticmethod
    cdef void train_example(NeuralNetC* nn, Pool mem, ExampleC* eg) except *:
        memset(nn.gradient,
            0, sizeof(nn.gradient[0]) * nn.nr_weight)
        NN.predict_example(eg,
            nn)
        insert_sparse(mem, nn.sparse_weights,
            nn.embed_lengths, nn.embed_offsets, nn.embed_defaults,
            eg.features, eg.nr_feat)
        # N.B. If we switch the insert_sparse API away from taking this
        # defaults argument, ensure that we allow zero-initialization option.
        insert_sparse(mem, nn.sparse_momentum,
            nn.embed_lengths, nn.embed_offsets, nn.embed_defaults,
            eg.features, eg.nr_feat)
        NN.update(nn, eg)
     
    @staticmethod
    cdef void forward(
        float* scores,
        float** fwd,
            const FeatureC* feats,
                int nr_feat,
            const NeuralNetC* nn
    ) nogil:
        set_input(fwd[0],
            feats, nr_feat, nn.embed_lengths, nn.embed_offsets,
            nn.embed_defaults, nn.sparse_weights) 
        forward(scores, fwd,
            nn.widths, nn.nr_layer, nn.weights, nn.nr_weight, &nn.hp,
            nn.iterate, nn.begin_fwd, nn.feed_fwd, nn.end_fwd)

    @staticmethod
    cdef void backward(
        float** bwd,
            const float* const* fwd,
            const float* costs,
            const NeuralNetC* nn
    ) nogil:
        backward(bwd,
            fwd, nn.widths, nn.nr_layer, nn.weights, nn.nr_weight,
            costs, nn.widths[nn.nr_layer-1], &nn.hp,
            nn.iterate, nn.begin_bwd, nn.feed_bwd, nn.end_bwd)

    @staticmethod
    cdef void update(
        NeuralNetC* nn,
            const ExampleC* eg
    ) nogil:
        dense_update(nn.weights, nn.gradient, nn.momentum,
            nn.nr_weight, eg.bwd_state, eg.fwd_state, nn.widths, nn.nr_layer,
            &nn.hp, nn.iterate, nn.update)
        sparse_update(
            nn.sparse_weights,
            nn.sparse_momentum,
            nn.gradient,
                eg.bwd_state[0],
                    nn.widths[0],
                nn.embed_lengths,
                nn.embed_offsets,
                nn.embed_defaults,
                    nn.nr_embed,
                eg.features,
                    eg.nr_feat,
                &nn.hp,
                nn.update)



cdef class NeuralNet:
    cdef readonly Pool mem
    cdef readonly Example eg
    cdef NeuralNetC c

    def __init__(self, widths, embed=None, weight_t eta=0.005, weight_t eps=1e-6,
                 weight_t mu=0.2, weight_t rho=1e-4, weight_t bias=0.0, weight_t alpha=0.0):
        self.mem = Pool()
        NN.init(&self.c, self.mem, widths, eta, eps, mu, rho, bias, alpha)
        self.eg = Example(self.widths)

    def predict_example(self, Example eg):
        NN.predict_example(&eg.c,
            &self.c)
        return eg

    def predict_sparse(self, features):
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        NN.predict_example(&eg.c,
            &self.c)
        return eg

    def predict_dense(self, features):
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_input(features)
        self.predict_example(eg)
        return eg
  
    def train(self, features, y):
        memset(self.c.gradient,
            0, sizeof(self.c.gradient[0]) * self.c.nr_weight)
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        eg.set_label(y)

        NN.predict_example(&eg.c, &self.c)
        NN.update(&self.c, &eg.c)
        return eg
 
    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        return Example(self.widths, input_, label)

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
            while self.c.iterate(&it, self.c.widths, self.c.nr_layer-1, 1):
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
            for i in range(self.c.nr_embed):
                j = 0
                while Map_iter(self.c.sparse_weights[i], &j, &key, &value):
                    emb = <weight_t*>value
                    yield key, [emb[k] for k in range(self.c.embed_lengths[i])]

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
            return self.c.hp.e
        def __set__(self, eta):
            self.c.hp.e = eta
    property rho:
        def __get__(self):
            return self.c.hp.rho
        def __set__(self, rho):
            self.c.hp.r = rho
    property eps:
        def __get__(self):
            return self.c.hp.p
        def __set__(self, eps):
            self.c.hp.p = eps




#@cython.cdivision(True)
#cdef void __tmp(OptimizerC* opt, weight_t* moments, weight_t* weights,
#        weight_t* gradient, weight_t scale, int nr_weight) nogil:
#    cdef weight_t beta1 = 0.90
#    cdef weight_t beta2 = 0.999
#    cdef weight_t EPS = 1e-6
#    Vec.mul_i(gradient,
#        scale, nr_weight)
#    # Add the derivative of the L2-loss to the gradient
#    cdef int i
#    if opt.rho != 0:
#        VecVec.add_i(gradient,
#            weights, opt.rho, nr_weight)
#    # This is all vectorized and in-place, so it's hard to read. See the
#    # paper.
#    mom1 = moments
#    mom2 = &moments[nr_weight]
#    Vec.mul_i(mom1,
#        beta1, nr_weight)
#    VecVec.add_i(mom1,
#        gradient, 1-beta1, nr_weight)
#    Vec.mul_i(mom2,
#        beta2, nr_weight)
#    VecVec.mul_i(gradient,
#        gradient, nr_weight)
#    VecVec.add_i(mom2,
#        gradient, 1-beta2, nr_weight)
#    Vec.div(gradient,
#        mom1, 1-beta1, nr_weight)
#    for i in range(nr_weight):
#        gradient[i] /= sqrtf(mom2[i] / (1-beta2)) + EPS
#    Vec.mul_i(gradient,
#        opt.eta, nr_weight)
#    VecVec.add_i(weights,
#        gradient, -1.0, nr_weight)
#
#


cdef void he_normal_initializer(float* weights, int fan_in, int n) except *:
    # See equation 10 here:
    # http://arxiv.org/pdf/1502.01852v1.pdf
    values = numpy.random.normal(loc=0.0, scale=numpy.sqrt(2.0 / float(fan_in)), size=n)
    for i, value in enumerate(values):
        weights[i] = value


cdef void constant_initializer(float* weights, float value, int n) nogil:
    for i in range(n):
        weights[i] = value
