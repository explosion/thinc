# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
from __future__ import print_function

from libc.string cimport memmove, memset, memcpy
from libc.stdint cimport uint64_t

cimport cython

from cymem.cymem cimport Pool
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t

from ..typedefs cimport weight_t, atom_t, feat_t
from ..typedefs cimport len_t, idx_t
from ..linalg cimport MatMat, MatVec, VecVec, Vec
from .. cimport prng
from ..structs cimport MapC
from ..structs cimport NeuralNetC
from ..structs cimport ExampleC
from ..structs cimport FeatureC
from ..structs cimport EmbedC
from ..structs cimport ConstantsC
from ..structs cimport do_update_t

from ..extra.eg cimport Example

from .solve cimport noisy_update, vanilla_sgd

from .forward cimport ELU_forward
from .forward cimport ReLu_forward
from .backward cimport ELU_backward
from .backward cimport ReLu_backward
from .backward cimport d_log_loss

from .embed cimport Embedding
from .initializers cimport he_normal_initializer, he_uniform_initializer

from libc.string cimport memcpy
from libc.math cimport isnan, sqrt

import random
import numpy


cdef class NN:
    @staticmethod
    cdef int nr_weight(int nr_out, int nr_in) nogil:
        return nr_out * nr_in + nr_out

    @staticmethod
    cdef void init(
        NeuralNetC* nn,
        Pool mem,
            widths,
            embed=None,
            update_step='sgd',
            weight_t eta=0.005,
            weight_t rho=1e-4,
    ) except *:
        prng.normal_setup()
        if update_step == 'sgd':
            nn.update = noisy_update
        else:
            raise NotImplementedError(update_step)
        nn.feed_fwd = ELU_forward
        nn.feed_bwd = ELU_backward

        nn.hp.t = 0
        nn.hp.r = rho
        nn.hp.e = eta

        nn.nr_layer = len(widths)
        nn.widths = <len_t*>mem.alloc(nn.nr_layer, sizeof(widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            nn.widths[i] = width
        nn.nr_weight = 0
        nn.nr_node = 0
        for i in range(nn.nr_layer-1):
            nn.nr_weight += NN.nr_weight(nn.widths[i+1], nn.widths[i])
            nn.nr_node += nn.widths[i]
        nn.weights = <weight_t*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.gradient = <weight_t*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        
        if embed is not None:
            vector_widths, features = embed
            Embedding.init(&nn.embed, mem, vector_widths, features)

        W = nn.weights
        for i in range(nn.nr_layer-2):
            he_normal_initializer(W,
                nn.widths[i+1], nn.widths[i+1] * nn.widths[i])
            W += NN.nr_weight(nn.widths[i+1], nn.widths[i])
    
    @staticmethod
    cdef void train_example(NeuralNetC* nn, Pool mem, ExampleC* eg) except *:
        nn.hp.t += 1
        if eg.nr_feat >= 1:
            Embedding.insert_missing(mem, &nn.embed,
                eg.features, eg.nr_feat)
            Embedding.set_input(eg.fwd_state[0],
                eg.features, eg.nr_feat, &nn.embed)
        NN.forward(eg.scores, eg.fwd_state,
            nn, True)
        NN.backward(eg.bwd_state, nn.gradient,
            eg.fwd_state, eg.costs, nn)
        if eg.nr_feat >= 1:
            Embedding.fine_tune(&nn.embed,
                eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)
        if not nn.hp.t % 100:
            nn.update(nn.weights, nn.gradient,
                nn.nr_weight, &nn.hp)
            Embedding.update_all(&nn.embed,
                &nn.hp, nn.update)
    
    @staticmethod
    cdef void train_batch(NeuralNetC* nn, Pool mem, ExampleC** egs, int nr_eg) except *:
        for eg in egs[:nr_eg]:
            nn.hp.t += 1
            if eg.nr_feat >= 1:
                Embedding.insert_missing(mem, &nn.embed,
                    eg.features, eg.nr_feat)
                Embedding.set_input(eg.fwd_state[0],
                    eg.features, eg.nr_feat, &nn.embed)
            NN.forward(eg.scores, eg.fwd_state,
                nn, True)
            NN.backward(eg.bwd_state, nn.gradient,
                eg.fwd_state, eg.costs, nn)
            if eg.nr_feat >= 1:
                Embedding.fine_tune(&nn.embed,
                    eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)
            if not nn.hp.t % 100:
                nn.update(nn.weights, nn.gradient,
                    nn.nr_weight, &nn.hp)
                Embedding.update_all(&nn.embed,
                    &nn.hp, nn.update)
 
    @staticmethod
    cdef void forward(weight_t* scores, weight_t** fwd,
            const NeuralNetC* nn, bint dropout) nogil:
        cdef const weight_t* W = nn.weights
        cdef uint64_t bit_mask
        cdef uint64_t j
        cdef uint64_t one = 1
        for i in range(nn.nr_layer-1):
            nn.feed_fwd(&fwd[i],
                W, &nn.widths[i], i, nn.nr_layer-(i+1), &nn.hp)
            W += NN.nr_weight(nn.widths[i+1], nn.widths[i])
        memcpy(scores,
            fwd[nn.nr_layer-1], sizeof(scores[0]) * nn.widths[nn.nr_layer-1])

    @staticmethod
    cdef void backward(weight_t** bwd, weight_t* gradient,
            const weight_t* const* fwd, const weight_t* costs, const NeuralNetC* nn) nogil:
        d_log_loss(bwd[nn.nr_layer-1],
            costs, fwd[nn.nr_layer-1], nn.widths[nn.nr_layer-1])
        cdef const weight_t* W = nn.weights + nn.nr_weight
        cdef weight_t* G = gradient + nn.nr_weight
        for i in range(nn.nr_layer-2, -1, -1):
            W -= NN.nr_weight(nn.widths[i+1], nn.widths[i])
            G -= NN.nr_weight(nn.widths[i+1], nn.widths[i])
            nn.feed_bwd(G, &bwd[i],
                W, &fwd[i], &nn.widths[i], nn.nr_layer-(i+1), i, &nn.hp)


cdef class NeuralNet:
    def __init__(self, widths, embed=None,
                 weight_t eta=0.005, weight_t rho=1e-4, update_step='sgd'):
        prng.normal_setup()
        self.mem = Pool()
        NN.init(&self.c, self.mem, widths, embed, update_step,
                eta, rho)
        self.eg = Example(nr_class=self.nr_class, widths=self.widths)

    def predict_example(self, Example eg):
        if eg.c.nr_feat >= 1:
            Embedding.insert_missing(self.mem, &self.c.embed,
                eg.c.features, eg.c.nr_feat)
            Embedding.set_input(eg.c.fwd_state[0],
                eg.c.features, eg.c.nr_feat, &self.c.embed)
        NN.forward(eg.c.scores, eg.c.fwd_state,
            &self.c, False)
        return eg

    def predict_dense(self, features):
        cdef Example eg = Example(nr_class=self.nr_class, widths=self.widths)
        cdef weight_t value
        for i, value in enumerate(features):
            eg.c.fwd_state[0][i] = value
        return self.predict_example(eg)

    def predict_sparse(self, features):
        cdef Example eg = self.Example(features)
        return self.predict_example(eg)
    
    def train_dense(self, features, y):
        cdef Example eg = Example(nr_class=self.nr_class, widths=self.widths)
        cdef weight_t value 
        for i, value in enumerate(features):
            eg.c.fwd_state[0][i] = value
        eg.costs = y
        NN.train_example(&self.c, self.mem, eg.c)
        return eg
  
    def train_sparse(self, features, label):
        cdef Example eg = self.Example(features, label=label)
        NN.train_example(&self.c, self.mem, eg.c)
        return eg
   
    def train_example(self, Example eg):
        NN.train_example(&self.c, self.mem, eg.c)
        return eg

    def train(self, egs):
        cdef ExampleC* minibatch[100]
        cdef Example eg
        for i, eg in enumerate(egs):
            if i and not i % 100:
                NN.train_batch(&self.c, self.mem, minibatch, 100)
            minibatch[i % 100] = eg.c
        if i % 100:
            NN.train_batch(&self.c, self.mem, minibatch, (i % 100) + 1)
 
    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        cdef Example eg = Example(nr_class=self.nr_class, widths=self.widths)
        eg.features = input_
        if label is not None:
            if isinstance(label, int):
                eg.costs = [i != label for i in range(eg.nr_class)]
            else:
                eg.costs = label
        return eg

    property weights:
        def __get__(self):
            return [self.c.weights[i] for i in range(self.c.nr_weight)]
        def __set__(self, weights):
            assert len(weights) == self.c.nr_weight
            for i, weight in enumerate(weights):
                self.c.weights[i] = weight

    property layers:
        def __get__(self):
            weights = list(self.weights)
            start = 0
            for i in range(self.c.nr_layer-1):
                nr_w = self.widths[i] * self.widths[i+1]
                nr_bias = self.widths[i] * self.widths[i+1] + self.widths[i+1]
                W = weights[start:start+nr_w]
                bias = weights[start+nr_w:start+nr_bias]
                yield W, bias
                start = start + NN.nr_weight(self.widths[i+1], self.widths[i])

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
            embeddings = []
            for i in range(self.c.embed.nr):
                j = 0
                table = []
                while Map_iter(self.c.embed.weights[i], &j, &key, &value):
                    emb = <weight_t*>value
                    table.append((key, [emb[k] for k in range(self.c.embed.lengths[i])]))
                embeddings.append(table)
            return embeddings

        def __set__(self, embeddings):
            cdef weight_t val
            for i, table in enumerate(embeddings):
                for key, value in table:
                    emb = <weight_t*>self.mem.alloc(self.c.embed.lengths[i], sizeof(emb[0]))
                    for j, val in enumerate(value):
                        emb[j] = val
                    Map_set(self.mem, self.c.embed.weights[i], <key_t>key, emb)

    property nr_layer:
        def __get__(self):
            return self.c.nr_layer
    property nr_weight:
        def __get__(self):
            return self.c.nr_weight
    property nr_class:
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
            return self.c.hp.r
        def __set__(self, rho):
            self.c.hp.r = rho
    property eps:
        def __get__(self):
            return self.c.hp.p
        def __set__(self, eps):
            self.c.hp.p = eps

    property tau:
        def __get__(self):
            return self.c.hp.t
        def __set__(self, tau):
            self.c.hp.t = tau
