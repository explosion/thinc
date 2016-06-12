# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
from __future__ import print_function

from libc.string cimport memmove, memset, memcpy
from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport malloc, calloc, free, rand

cimport cython
cimport numpy as np
from cpython.exc cimport PyErr_CheckSignals

from cymem.cymem cimport Pool
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t
from murmurhash.mrmr cimport hash64

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

from .solve cimport noisy_update, vanilla_sgd, adam, sgd_cm, asgd

from .forward cimport softmax
from .forward cimport ELU_forward
from .forward cimport ELU_batch_norm_forward
from .forward cimport ReLu_forward
from .backward cimport ELU_backward
from .backward cimport ReLu_backward
from .backward cimport ELU_batch_norm_backward
from .backward cimport d_log_loss, d_hinge_loss

from .embed cimport Embedding
from .initializers cimport he_normal_initializer, he_uniform_initializer, constant_initializer

from libc.string cimport memcpy
from libc.math cimport isnan, sqrt

import random
import numpy
from cytoolz import partition
import cPickle


prng.normal_setup()


cdef int get_nr_weight(int nr_out, int nr_in) nogil:
    return nr_out * nr_in + nr_out + nr_out


cdef class NeuralNet(Model):
    def __init__(self, widths, *args, **kwargs):
        self.mem = Pool()

        self.c.hp.e = kwargs.get('eta', 0.01)
        self.c.hp.r = kwargs.get('rho', 0.00)
        self.c.hp.m = kwargs.get('mu', 0.9)
        if kwargs.get('update_step') == 'sgd':
            self.c.update = vanilla_sgd
            nr_support = 1
        elif kwargs.get('update_step') == 'asgd':
            self.c.update = asgd
            nr_support = 2
        elif kwargs.get('update_step') == 'sgd_cm':
            self.c.update = sgd_cm
            nr_support = 3
        elif kwargs.get('update_step') == 'adam':
            self.c.update = adam
            nr_support = 4
        else:
            self.c.update = noisy_update
            nr_support = 1
        self.c.embed.nr_support = nr_support
        self.c.feed_fwd = ELU_batch_norm_forward
        self.c.feed_bwd = ELU_batch_norm_backward

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
        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight * nr_support,
                                                   sizeof(self.c.weights[0]))
        self.c.gradient = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))
        
        if kwargs.get('embed') is not None:
            vector_widths, features = kwargs['embed']
            Embedding.init(&self.c.embed, self.mem, vector_widths, features)

        W = self.c.weights
        for i in range(self.c.nr_layer-2):
            he_normal_initializer(W,
                self.c.widths[i+1], self.c.widths[i+1] * self.c.widths[i])
            nr_W = self.c.widths[i+1] * self.c.widths[i]
            nr_bias = self.c.widths[i+1]
            if get_nr_weight(self.c.widths[i+1], self.c.widths[i]) > (nr_W + nr_bias):
                # Initialise gamma terms
                constant_initializer(W + nr_W + nr_bias,
                    1.0, self.c.widths[i + 1])
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
        return self.updateC(eg.c.features, eg.c.nr_feat, True,
                            eg.c.costs, eg.c.is_valid, force_update)
    
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        pass

    def set_embedding(self, int i, key_t key, values):
        if len(values) != self.c.embed.lengths[i]:
            msg_vals = (i, self.c.embed.lengths[i], len(values))
            raise ValueError(
                "set_embedding %d expected embedding of length %d. Got length %d." % msg_vals)
        emb = <weight_t*>Map_get(self.c.embed.weights[i], key)
        grad = <weight_t*>Map_get(self.c.embed.gradients[i], key)
        if emb is NULL or grad is NULL:
            # If one is null both should be. But free just in case, to avoid mem
            # leak.
            free(emb)
            free(grad)
            emb = <weight_t*>self.mem.alloc(self.c.embed.lengths[i] * self.c.embed.nr_support,
                    sizeof(emb[0]))
            grad = <weight_t*>self.mem.alloc(self.c.embed.lengths[i], sizeof(emb[0]))
            Map_set(self.mem, self.c.embed.weights[i],
                key, emb)
            Map_set(self.mem, self.c.embed.gradients[i],
                key, grad)
 
        for j, value in enumerate(values):
            emb[j] = value
            emb[len(values) + j] = value # For average

    def dump(self, loc):
        data = (list(self.embeddings), self.weights, dict(self.c.hp)) 
        with open(loc, 'wb') as file_:
            cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)

    def load(self, loc):
        with open(loc, 'rb') as file_:
            embeddings, weights, hp = cPickle.load(file_)
        self.embeddings = embeddings
        self.weights = weights
        self.c.hp = hp

    def end_training(self):
        acc = self.c.weights + self.c.nr_weight
        for i in range(self.c.nr_weight):
            self.c.weights[i] = acc[i]
        Embedding.average(&self.c.embed, self.c.hp.t)

    @property
    def nr_feat(self):
        return self.c.widths[0]
   
    cdef void set_scoresC(self, weight_t* scores, const void* feats, int nr_feat,
            int is_sparse) nogil:
        fwd_state = <weight_t**>calloc(self.c.nr_layer, sizeof(void*))
        for i in range(self.c.nr_layer):
            fwd_state[i] = <weight_t*>calloc(self.c.widths[i], sizeof(weight_t))

        self._extractC(fwd_state[0], feats, nr_feat, is_sparse)

        self.c.feed_fwd(fwd_state,
            self.c.weights, self.c.widths, self.c.nr_layer, 1, &self.c.hp)
        memcpy(scores,
            fwd_state[self.c.nr_layer-1], sizeof(scores[0]) * self.c.widths[self.c.nr_layer-1])
        for i in range(self.c.nr_layer):
            free(fwd_state[i])
        free(fwd_state)
 
    cdef weight_t updateC(self, const void* feats, int nr_feat, int is_sparse,
            const weight_t* costs, const int* is_valid, int force_update) nogil:
        is_full = self._mb.push_back(feats, nr_feat, is_sparse, costs, is_valid)
        if nr_feat > 0 and is_sparse:
            with gil:
                Embedding.insert_missing(self.mem, &self.c.embed,
                    <const FeatureC*>feats, nr_feat)

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
        for i in range(mb.i):
            self._dropoutC(mb.features(i), mb.nr_feat(i), 1)
            self._extractC(mb.fwd(0, i), mb.features(i), mb.nr_feat(i), 1)
        
        self.c.feed_fwd(mb._fwd,
            self.c.weights, self.c.widths, self.c.nr_layer, mb.i, &self.c.hp)
        
        for i in range(mb.i):
            self._set_delta_lossC(mb.losses(i),
                mb.costs(i), mb.fwd(mb.nr_layer-1, i))
        
        self.c.feed_bwd(self.c.gradient + self.c.nr_weight, mb._bwd,
            self.c.weights + self.c.nr_weight, mb._fwd, self.c.widths, self.c.nr_layer,
            mb.i, &self.c.hp)

        self.c.hp.t += 1
        self.c.update(self.c.weights, self.c.gradient,
            self.c.nr_weight, &self.c.hp)
        for i in range(mb.i):
            self._backprop_extracterC(mb.bwd(0, i), mb.features(i), mb.nr_feat(i), 1)
        for i in range(mb.i):
            self._update_extracterC(mb.features(i), mb.nr_feat(i), 1)

    cdef void _dropoutC(self, void* _feats, int nr_feat, int is_sparse) nogil:
        cdef int dropout = 7
        cdef weight_t drop_score = 7.0 / 6.0
        if is_sparse:
            feats = <FeatureC*>_feats
            for i in range(nr_feat):
                if rand() % dropout:
                    # Preserve the mean activation, by increasing the activation
                    # of the non-dropped units. This way, we don't have to
                    # re-balance the weights.
                    # I think I read this somewhere.
                    # If not...well, it makes sense right?
                    # Right?
                    feats[i].value *= drop_score
                else:
                    feats[i].value = 0
    
    cdef void _extractC(self, weight_t* input_,
            const void* feats, int nr_feat, int is_sparse) nogil:
        if is_sparse:
            Embedding.set_input(input_,
                <const FeatureC*>feats, nr_feat, &self.c.embed)
        else:
            memcpy(input_,
                <const weight_t*>feats, nr_feat * sizeof(input_))

    cdef void _set_delta_lossC(self, weight_t* delta_loss,
            const weight_t* costs, const weight_t* scores) nogil:
        d_log_loss(delta_loss,
            costs, scores, self.c.widths[self.c.nr_layer-1])

    cdef void _backprop_extracterC(self, const weight_t* deltas, const void* feats,
            int nr_feat, int is_sparse) nogil:
        if nr_feat < 1 or not is_sparse:
            return
        Embedding.fine_tune(&self.c.embed,
            deltas, self.c.widths[0], <const FeatureC*>feats, nr_feat)

    cdef void _update_extracterC(self, const void* _feats,
            int nr_feat, int is_sparse) nogil:
        if nr_feat < 1 or not is_sparse:
            return
        feats = <const FeatureC*>_feats
        for feat in feats[:nr_feat]:
            Embedding.update(&self.c.embed,
                feat.i, feat.key, &self.c.hp, self.c.update)

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
        seen_tables = {}
        for i in range(self.c.embed.nr):
            addr = <uintptr_t>self.c.weights[i]
            if addr in seen_tables:
                table = seen_tables[addr]
            else:
                j = 0
                table = []
                length = self.c.embed.lengths[i]
                while Map_iter(self.c.embed.weights[i], &j, &key, &value):
                    emb = <weight_t*>value
                    table.append((key, [val for val in emb[:length]]))
                seen_tables[addr] = table
            embeddings.append(table)
        return embeddings

    @embeddings.setter
    def embeddings(self, embeddings):
        cdef weight_t val
        uniq_tables = {}
        for i, table in enumerate(embeddings):
            if id(table) in uniq_tables:
                self.c.weights[i] = self.c.weights[uniq_tables[id(table)]]
                continue
            uniq_tables[id(table)] = i
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
