# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
from __future__ import print_function

from libc.string cimport memmove, memset, memcpy
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, calloc, free

cimport cython
cimport numpy as np

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
from cytoolz import partition


prng.normal_setup()


cdef cppclass MinibatchC:
    weight_t** _fwd
    weight_t** _bwd
    len_t* widths
    int nr_layer
    int batch_size

    __init__(len_t* widths, int nr_layer, int batch_size) nogil:
        this.nr_layer = nr_layer
        this.batch_size = batch_size
        this.widths = <len_t*>calloc(nr_layer, sizeof(len_t))
        this._fwd = <weight_t**>calloc(nr_layer, sizeof(weight_t*))
        this._bwd = <weight_t**>calloc(nr_layer, sizeof(weight_t*))
        for i in range(nr_layer):
            this.widths[i] = widths[i]
            this._fwd[i] = <weight_t*>calloc(this.widths[i] * batch_size, sizeof(weight_t))
            this._bwd[i] = <weight_t*>calloc(this.widths[i] * batch_size, sizeof(weight_t))

    __dealloc__() nogil:
        for i in range(this.nr_layer):
            free(this._fwd[i])
            free(this._bwd[i])
        free(this._fwd)
        free(this._bwd)
        free(this.widths)

    weight_t* fwd(int i, int j) nogil:
        return this._fwd[i] + (j * this.widths[i])
 
    weight_t* bwd(int i, int j) nogil:
        return this._bwd[i] + (j * this.widths[i])
  
    weight_t* scores(int i) nogil:
        return this.fwd(this.nr_layer-1, i)

    weight_t* losses(int i) nogil:
        return this.bwd(this.nr_layer-1, i)


cdef class NN:
    @staticmethod
    cdef int nr_weight(int nr_out, int nr_in) nogil:
        return nr_out * nr_in + nr_out

    @staticmethod
    cdef void train_example(NeuralNetC* nn, Pool mem, ExampleC* eg) except *:
        nn.hp.t += 1
        if eg.nr_feat >= 1:
            Embedding.insert_missing(mem, &nn.embed,
                eg.features, eg.nr_feat)
            Embedding.set_input(eg.fwd_state[0],
                eg.features, eg.nr_feat, &nn.embed)
        nn.feed_fwd(eg.fwd_state,
            nn.weights, nn.widths, nn.nr_layer, 1, &nn.hp)
        d_log_loss(eg.bwd_state[nn.nr_layer-1],
            eg.costs, eg.fwd_state[nn.nr_layer-1], nn.widths[nn.nr_layer-1])
        nn.feed_bwd(nn.gradient + nn.nr_weight, eg.bwd_state,
            nn.weights + nn.nr_weight, eg.fwd_state, nn.widths, nn.nr_layer,
            1, &nn.hp)
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
        nn.hp.t += nr_eg
        cdef MinibatchC* mb = new MinibatchC(nn.widths, nn.nr_layer, nr_eg)
        nr_class = nn.widths[nn.nr_layer-1]
        for i in range(nr_eg):
            if egs[i].nr_feat >= 1:
                Embedding.insert_missing(mem, &nn.embed,
                    egs[i].features, egs[i].nr_feat)
                Embedding.set_input(egs[i].fwd_state[0],
                    egs[i].features, egs[i].nr_feat, &nn.embed)
            memcpy(mb.fwd(0, i),
                egs[i].fwd_state[0], sizeof(weight_t) * mb.widths[0])
        nn.feed_fwd(mb._fwd,
            nn.weights, nn.widths, nn.nr_layer, nr_eg, &nn.hp)
        # Set scores onto the ExampleC objects
        for i in range(mb.batch_size):
            memcpy(egs[i].scores,
                mb.scores(i), nr_class * sizeof(weight_t))
            # Set loss from the ExampleC costs 
            d_log_loss(mb.losses(i),
                egs[i].costs, mb.scores(i), nr_class)
        nn.feed_bwd(nn.gradient + nn.nr_weight, mb._bwd,
            nn.weights + nn.nr_weight, mb._fwd, nn.widths, nn.nr_layer,
            nr_eg, &nn.hp)
        nn.update(nn.weights, nn.gradient,
            nn.nr_weight, &nn.hp)

        # Set scores onto the ExampleC objects
        for i in range(mb.batch_size):
            memcpy(egs[i].scores,
                mb.scores(i), nr_class * sizeof(weight_t))
        for eg in egs[:mb.batch_size]:
            if eg.nr_feat >= 1:
                Embedding.fine_tune(&nn.embed,
                    eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)
        Embedding.update_all(&nn.embed,
            &nn.hp, nn.update)
        del mb


cdef class NeuralNet:
    def __init__(self, widths, embed=None,
                 weight_t eta=0.005, weight_t rho=1e-4, update_step='sgd'):
        self.mem = Pool()
        self.c.update = noisy_update
        self.c.feed_fwd = ELU_forward
        self.c.feed_bwd = ELU_backward

        self.c.hp.t = 0
        self.c.hp.r = rho
        self.c.hp.e = eta

        self.c.nr_layer = len(widths)
        self.c.widths = <len_t*>self.mem.alloc(self.c.nr_layer, sizeof(widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            self.c.widths[i] = width
        self.c.nr_weight = 0
        self.c.nr_node = 0
        for i in range(self.c.nr_layer-1):
            self.c.nr_weight += self.c.widths[i+1] * self.c.widths[i] + self.c.widths[i+1]
            self.c.nr_node += self.c.widths[i]
        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        self.c.gradient = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        
        if embed is not None:
            vector_widths, features = embed
            Embedding.init(&self.c.embed, self.mem, vector_widths, features)

        W = self.c.weights
        for i in range(self.c.nr_layer-2):
            he_normal_initializer(W,
                self.c.widths[i+1], self.c.widths[i+1] * self.c.widths[i])
            W += self.c.widths[i+1] * self.c.widths[i] + self.c.widths[i+1]

    def predict_batch(self, inputs):
        mb = new MinibatchC(self.c.widths, self.c.nr_layer, len(inputs))
        cdef weight_t[::1] input_
        for i, input_ in enumerate(inputs):
            memcpy(mb.fwd(0, i),
                &input_[0], sizeof(weight_t) * self.c.widths[0])
        cdef np.ndarray scores = numpy.zeros(shape=(mb.batch_size, self.nr_class),
                                               dtype='float64')
        self.c.feed_fwd(mb._fwd,
            self.c.weights, self.c.widths, self.c.nr_layer, mb.batch_size, &self.c.hp)
        memcpy(<weight_t*>scores.data,
            mb._fwd[self.c.nr_layer-1],
            sizeof(scores[0]) * self.c.widths[self.c.nr_layer-1] * mb.batch_size)
        scores.reshape((mb.batch_size, self.nr_class))
        del mb
        return scores

    def predict_example(self, Example eg):
        if eg.c.nr_feat >= 1:
            Embedding.insert_missing(self.mem, &self.c.embed,
                eg.c.features, eg.c.nr_feat)
            Embedding.set_input(eg.c.fwd_state[0],
                eg.c.features, eg.c.nr_feat, &self.c.embed)
        self.c.feed_fwd(eg.c.fwd_state,
            self.c.weights, self.c.widths, self.c.nr_layer, 1, &self.c.hp)
        memcpy(eg.c.scores,
            eg.c.fwd_state[self.c.nr_layer-1],
            sizeof(eg.c.scores[0]) * self.c.widths[self.c.nr_layer-1])
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

    def train(self, x_y, batch_size=50):
        minibatch = <ExampleC**>malloc(batch_size * sizeof(ExampleC*))
        correct = 0.0
        total = 0.0
        cdef Example eg
        for batch in partition(batch_size, x_y):
            egs = []
            for i, (x, y) in enumerate(batch):
                eg = Example(nr_class=self.nr_class, widths=self.widths)
                eg.set_input(x)
                eg.costs = [clas != y for clas in range(self.nr_class)]
                minibatch[i] = eg.c
                egs.append(eg)
            NN.train_batch(&self.c, self.mem, minibatch, len(batch))
            correct += sum(eg.guess == eg.best for eg in egs)
            total += len(batch)
        free(minibatch)
        return correct / total
 
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
            cdef weight_t weight
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
