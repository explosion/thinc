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

from ..base cimport Model
from ..typedefs cimport weight_t, atom_t, feat_t
from ..typedefs cimport len_t, idx_t
from ..linalg cimport MatMat, MatVec, VecVec, Vec
from .. cimport prng
from ..structs cimport MapC
from ..structs cimport NeuralNetC
from ..structs cimport ExampleC
from ..structs cimport MinibatchC
from ..structs cimport FeatureC
from ..structs cimport EmbedC
from ..structs cimport ConstantsC
from ..structs cimport do_update_t

from ..extra.eg cimport Example
from ..extra.mb cimport Minibatch

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


cdef int get_nr_weight(int nr_out, int nr_in) nogil:
    return nr_out * nr_in + nr_out


cdef class NeuralNet(Model):
    def __init__(self, widths, *args, **kwargs):
        self.mem = Pool()

        self.c.hp.e = kwargs.get('eta', 0.01)
        self.c.hp.r = kwargs.get('rho', 0.00)
        if kwargs.get('update_step') == 'sgd':
            self.c.update = vanilla_sgd
        else:
            self.c.update = noisy_update
        self.c.feed_fwd = ELU_forward
        self.c.feed_bwd = ELU_backward

        self.c.nr_layer = len(widths)
        self.c.widths = <len_t*>self.mem.alloc(self.c.nr_layer, sizeof(widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            self.c.widths[i] = width
        self.c.nr_weight = 0
        self.c.nr_node = 0
        for i in range(self.c.nr_layer-1):
            self.c.nr_weight += get_nr_weight(self.c.widths[i+1], self.c.widths[i])
            self.c.nr_node += self.c.widths[i]
        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        self.c.gradient = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        
        if kwargs.get('embed') is not None:
            vector_widths, features = kwargs['embed']
            Embedding.init(&self.c.embed, self.mem, vector_widths, features)

        W = self.c.weights
        for i in range(self.c.nr_layer-2):
            he_normal_initializer(W,
                self.c.widths[i+1], self.c.widths[i+1] * self.c.widths[i])
            W += get_nr_weight(self.c.widths[i+1], self.c.widths[i])
        self._mb = new MinibatchC(self.c.widths, self.c.nr_layer, 100)

    def __call__(self, Example eg):
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

    def train(self, examples):
        cdef Example eg
        for eg in examples:
            self.c.hp.t += 1
            if eg.c.nr_feat != 0:
                Embedding.insert_missing(self.mem, &self.c.embed,
                    eg.c.features, eg.c.nr_feat)
            is_full = self._mb.push_back(eg.c.features, eg.c.nr_feat, True,
                                         eg.c.costs, eg.c.is_valid)
            if is_full:
                self._updateC(self._mb)
                minibatch = Minibatch.take_ownership(self._mb)
                yield from minibatch
                self._mb = new MinibatchC(self.c.widths, self.c.nr_layer,
                                          minibatch.batch_size)

    def update(self, Example eg, force_update=False):
        return self.updateC(eg.c, force_update)

    def dump(self, loc):
        pass

    def load(self, loc):
        pass

    def end_training(self):
        pass

    @property
    def nr_feat(self):
        return self.c.widths[0]
   
    cdef void set_scoresC(self, weight_t* scores, const void* feats, int nr_feat,
            int is_sparse) nogil:
        fwd_state = <weight_t**>calloc(self.c.nr_layer, sizeof(void*))
        for i in range(self.c.nr_layer):
            fwd_state[i] = <weight_t*>calloc(self.c.widths[i], sizeof(weight_t))

        if is_sparse:
            Embedding.set_input(fwd_state[0],
                <const FeatureC*>feats, nr_feat, &self.c.embed)
        else:
            memcpy(fwd_state[0],
                <const weight_t*>feats, nr_feat * sizeof(fwd_state[0]))
        self.c.feed_fwd(fwd_state,
            self.c.weights, self.c.widths, self.c.nr_layer, 1, &self.c.hp)
        memcpy(scores,
            fwd_state[self.c.nr_layer-1], sizeof(scores[0]) * self.c.widths[self.c.nr_layer-1])
        for i in range(self.c.nr_layer):
            free(fwd_state[i])
        free(fwd_state)
 
    cdef weight_t updateC(self, const ExampleC* eg, int force_update) nogil:
        self.c.hp.t += 1
        is_full = self._mb.push_back(eg.features, eg.nr_feat, 1, eg.costs, eg.is_valid)
        if eg.nr_feat != 0:
            with gil:
                Embedding.insert_missing(self.mem, &self.c.embed,
                    eg.features, eg.nr_feat)

        cdef weight_t loss = 0.0
        cdef int i
        if is_full or force_update:
            self._updateC(self._mb)
            for i in range(self._mb.i):
                loss += 1.0 - self._mb.scores(i)[self._mb.best(i)]
            batch_size = self._mb.batch_size
            del self._mb
            self._mb = new MinibatchC(self.c.widths, self.c.nr_layer, batch_size)
        return loss

    cdef void _updateC(self, MinibatchC* mb) nogil:
        nr_class = self.c.widths[self.c.nr_layer-1]
        
        for i in range(mb.i):
            if mb.nr_feat(i) != 0:
                Embedding.set_input(mb.fwd(0, i),
                    mb.features(i), mb.nr_feat(i), &self.c.embed)
        self.c.feed_fwd(mb._fwd,
            self.c.weights, self.c.widths, self.c.nr_layer, mb.i, &self.c.hp)
        for i in range(mb.i):
            # Set loss from the costs 
            d_log_loss(mb.losses(i),
                mb.costs(i), mb.fwd(mb.nr_layer-1, i), nr_class)

        self.c.feed_bwd(self.c.gradient + self.c.nr_weight, mb._bwd,
            self.c.weights + self.c.nr_weight, mb._fwd, self.c.widths, self.c.nr_layer,
            mb.i, &self.c.hp)

        self.c.update(self.c.weights, self.c.gradient,
            self.c.nr_weight, &self.c.hp)

        for i in range(mb.i):
            if mb.nr_feat(i) != 0:
                Embedding.fine_tune(&self.c.embed,
                    mb.bwd(0, i), self.c.widths[0], mb.features(i), mb.nr_feat(i))
        for i in range(mb.i):
            for feat in mb.features(i)[:mb.nr_feat(i)]:
                Embedding.update(&self.c.embed,
                    feat.i, feat.key, &self.c.hp, self.c.update)

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        pass

    @property
    def weights(self):
        return [self.c.weights[i] for i in range(self.c.nr_weight)]
    
    @weights.setter
    def weights(self, weights):
        assert len(weights) == self.c.nr_weight
        cdef weight_t weight
        for i, weight in enumerate(weights):
            self.c.weights[i] = weight

    property layers:
        # TODO: Apparent Cython bug: @property syntax fails on generators?
        def __get__(self):
            weights = list(self.weights)
            start = 0
            for i in range(self.c.nr_layer-1):
                nr_w = self.widths[i] * self.widths[i+1]
                nr_bias = self.widths[i] * self.widths[i+1] + self.widths[i+1]
                W = weights[start:start+nr_w]
                bias = weights[start+nr_w:start+nr_bias]
                yield W, bias
                start = start + get_nr_weight(self.widths[i+1], self.widths[i])

    @property
    def widths(self):
        return tuple(self.c.widths[i] for i in range(self.c.nr_layer))

    property layer_l1s:
        # TODO: Apparent Cython bug: @property syntax fails on generators?
        def __get__(self):
            for W, bias in self.layers:
                w_l1 = sum(abs(w) for w in W) / len(W)
                bias_l1 = sum(abs(w) for w in W) / len(bias)
                yield w_l1, bias_l1

    @property
    def gradient(self):
        return [self.c.gradient[i] for i in range(self.c.nr_weight)]

    @property
    def l1_gradient(self):
        cdef int i
        cdef weight_t total = 0.0
        for i in range(self.c.nr_weight):
            if self.c.gradient[i] < 0:
                total -= self.c.gradient[i]
            else:
                total += self.c.gradient[i]
        return total / self.c.nr_weight

    @property
    def embeddings(self):
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

    @embeddings.setter
    def embeddings(self, embeddings):
        cdef weight_t val
        for i, table in enumerate(embeddings):
            for key, value in table:
                emb = <weight_t*>self.mem.alloc(self.c.embed.lengths[i], sizeof(emb[0]))
                for j, val in enumerate(value):
                    emb[j] = val
                Map_set(self.mem, self.c.embed.weights[i], <key_t>key, emb)

    @property
    def nr_layer(self):
        return self.c.nr_layer

    @property
    def nr_weight(self):
        return self.c.nr_weight

    @property
    def nr_class(self):
        return self.c.widths[self.c.nr_layer-1]

    @property
    def nr_in(self):
        return self.c.widths[0]

    @property
    def eta(self):
        return self.c.hp.e
    @eta.setter
    def eta(self, eta):
            self.c.hp.e = eta

    @property
    def rho(self):
        return self.c.hp.r
    @rho.setter
    def rho(self, rho):
        self.c.hp.r = rho
    
    @property
    def eps(self):
        return self.c.hp.p
    @eps.setter
    def eps(self, eps):
        self.c.hp.p = eps

    @property
    def tau(self):
        return self.c.hp.t
    @tau.setter
    def tau(self, tau):
        self.c.hp.t = tau
