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
from ..structs cimport LayerC
from ..structs cimport EmbedC
from ..structs cimport ConstantsC
from ..structs cimport do_update_t

from ..extra.eg cimport Example
from ..extra.mb cimport Minibatch

from .solve cimport vanilla_sgd, sgd_cm, nag, adagrad, adadelta, adam

from .forward cimport softmax
from .forward cimport ELU_forward
from .forward cimport ReLu_forward
from .forward cimport ReLu
from .backward cimport ReLu_backward
from .backward cimport ELU_backward
from .backward cimport d_ReLu, d_softmax

#from .forward cimport ELU_batch_norm_residual_forward
#from .forward cimport ReLu_layer_norm_forward
#from .backward cimport ELU_batch_norm_residual_backward
#from .backward cimport ReLu_layer_norm_backward

from .embed cimport Embedding
from .initializers cimport he_normal_initializer, he_uniform_initializer, constant_initializer

from libc.string cimport memcpy
from libc.math cimport isnan, sqrt

import random
import numpy
import cPickle


cdef int get_nr_weight(int nr_out, int nr_in, int batch_norm) nogil:
    if batch_norm:
        return nr_out * nr_in + nr_out * 5
    else:
        return nr_out * nr_in + nr_out


cdef class NeuralNet(Model):
    def __init__(self, widths, *args, **kwargs):
        self.mem = kwargs.get('mem') or Pool()
        self.c.embed = <EmbedC*>self.mem.alloc(sizeof(EmbedC), 1)

        # Learning rate
        self.c.hp.e = kwargs.get('eta', 0.001)
        # Regularization
        self.c.hp.r = kwargs.get('rho', 0.00)
        # Momentum
        self.c.hp.m = kwargs.get('mu', 0.9)
        # Gradient noise
        self.c.hp.w = kwargs.get('noise', 0.0)
        # Dropout
        self.c.hp.d = kwargs.get('dropout', 0.0)
        if kwargs.get('update_step') == 'sgd':
            self.c.update = vanilla_sgd
            nr_support = 2
        elif kwargs.get('update_step', 'sgd_cm') == 'sgd_cm':
            self.c.update = sgd_cm
            nr_support = 3
        elif kwargs.get('update_step') == 'nag':
            self.c.update = nag
            nr_support = 3
        elif kwargs.get('update_step') == 'adagrad':
            self.c.update = adagrad
            nr_support = 3
        elif kwargs.get('update_step') == 'adadelta':
            self.c.update = adadelta
            nr_support = 4
        elif kwargs.get('update_step') == 'adam':
            self.c.update = adam
            nr_support = 4
        else:
            raise ValueError(kwargs.get('update_step'))
        self.c.embed.nr_support = nr_support
        norm_type = kwargs.get('norm_type', None)
        #if norm_type == 'layer':
        #    self.c.feed_fwd = ReLu_layer_norm_forward
        #    self.c.feed_bwd = ReLu_layer_norm_backward
        #elif norm_type == 'batch':
        #    raise NotImplementedError
        #else:
        self.c.feed_fwd = ReLu_forward
        self.c.feed_bwd = ReLu_backward

        self.c.nr_layer = len(widths)
        self.c.widths = <len_t*>self.mem.alloc(self.c.nr_layer, sizeof(widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            self.c.widths[i] = width
        self.c.nr_weight = 0
        self.c.nr_node = 0
        for i in range(self.c.nr_layer-1):
            self.c.nr_weight += get_nr_weight(self.c.widths[i+1], self.c.widths[i],
                                              norm_type != None)
            self.c.nr_node += self.c.widths[i]
        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_weight * nr_support,
                                                   sizeof(self.c.weights[0]))
        self.c.gradient = <weight_t*>self.mem.alloc(self.c.nr_weight, sizeof(self.c.weights[0]))

        if kwargs.get('embed') is not None:
            vector_widths, features = kwargs['embed']
            print("Make embed", vector_widths, features)
            Embedding.init(self.c.embed, self.mem, vector_widths, features)

        self.c.layers = <LayerC*>self.mem.alloc(self.c.nr_layer, sizeof(LayerC))
        self.c.d_layers = <LayerC*>self.mem.alloc(self.c.nr_layer, sizeof(LayerC))
        W = self.c.weights
        G = self.c.gradient
        for i in range(self.c.nr_layer-2):
            he_normal_initializer(W,
                self.c.widths[i+1], self.c.widths[i+1] * self.c.widths[i])
            nr_W = self.c.widths[i+1] * self.c.widths[i]
            self.c.layers[i].activate = ReLu
            self.c.layers[i].sparse = NULL
            self.c.layers[i].dense = W
            self.c.layers[i].bias = W+nr_W
            self.c.d_layers[i].activate = NULL
            self.c.d_layers[i].sparse = NULL
            self.c.d_layers[i].dense = G
            self.c.d_layers[i].bias = G+nr_W
            for j in range(nr_W):
                if random.random() < 0.4:
                    W[j] = 0.0
            nr_bias = self.c.widths[i+1]
            constant_initializer(W+nr_W,
                -0.000001, self.c.widths[i+1] * self.c.widths[i])
            if norm_type != None:
                # Initialise gamma terms
                constant_initializer(W + nr_W + nr_bias,
                    1.0, self.c.widths[i + 1])
                # Initialize variance
                constant_initializer(W + nr_W + self.c.widths[i+1] * 4,
                    1.0, self.c.widths[i+1])
            W += get_nr_weight(self.c.widths[i+1], self.c.widths[i], norm_type != None)
            G += get_nr_weight(self.c.widths[i+1], self.c.widths[i], norm_type != None)
        i = self.c.nr_layer - 2
        constant_initializer(W,
            1e-6, self.c.widths[i+1] * self.c.widths[i])
        constant_initializer(W+self.c.widths[i+1]*self.c.widths[i],
            1e-6, self.c.widths[i+1])
        self.c.layers[i].activate = softmax
        self.c.layers[i].dense = W
        self.c.layers[i].bias = W+(self.c.widths[i]*self.c.widths[i+1])
        self.c.d_layers[i].activate = NULL
        self.c.d_layers[i].dense = G
        self.c.d_layers[i].bias = G+(self.c.widths[i]*self.c.widths[i+1])
 
        self._mb = Minibatch(self.widths, kwargs.get('batch_size', 200))

    def __call__(self, eg_or_mb):
        cdef Example eg
        cdef Minibatch mb
        if isinstance(eg_or_mb, Example):
            eg = eg_or_mb
            self.set_scoresC(eg.c.scores, eg.c.features, eg.c.nr_feat, eg.c.is_sparse)
        elif isinstance(eg_or_mb, Minibatch):
            mb = eg_or_mb
            for i in range(mb.c.i):
                self._extractC(mb.c.fwd(0, i),
                    mb.c.features(i), mb.c.nr_feat(i), mb.c.is_sparse(i))
            self.c.feed_fwd(mb.c._fwd,
                self.c.layers, NULL, self.c.widths, self.c.nr_layer, mb.c.i, &self.c.hp)
        return eg_or_mb

    def train(self, examples):
        cdef Example eg
        for eg in examples:
            is_full = self._mb.c.push_back(eg.c.features, eg.c.nr_feat, eg.c.is_sparse,
                                           eg.c.costs, eg.c.is_valid, 0)
            if is_full:
                self._updateC(self._mb.c)
                yield from self._mb

    def update(self, Example eg, force_update=False):
        loss = self.updateC(eg.c.features, eg.c.nr_feat, eg.c.is_sparse,
                            eg.c.costs, eg.c.is_valid, force_update, 0)
        return loss
    
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        pass

    def has_embedding(self, int i, key_t key):
        emb = <weight_t*>Map_get(self.c.embed.weights[i], key)
        return True if emb is not NULL else False

    def default_embedding(self, int i):
        return [self.c.embed.defaults[i][j] for j in range(self.c.embed.lengths[i])]

    def set_embedding(self, int i, key_t key, values):
        '''Insert an embedding for a given key.'''
        if len(values) != self.c.embed.lengths[i]:
            msg_vals = (i, self.c.embed.lengths[i], len(values))
            raise ValueError(
                "set_embedding %d expected embedding of length %d. Got length %d." % msg_vals)
        emb = <weight_t*>Map_get(self.c.embed.weights[i], key)
        grad = <weight_t*>Map_get(self.c.embed.gradients[i], key)
        if emb is NULL or grad is NULL:
            # If one is null both should be. But free just in case, to avoid mem
            # leak.
            if emb is not NULL:
                self.mem.free(emb)
            if grad is not NULL:
                self.mem.free(grad)
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

    def sparsify_embeddings(self, weight_t threshold, int use_infinity_norm=True):
        '''Prune all embeddings where:
        
        | embed - default |_infinity < threshold

        That is, if max(abs(embed - default)) < threshold, allow the embedding to
        be represented by the default.
        '''
        cdef key_t key
        cdef void* value
        cdef int i, j
        cdef weight_t infinity_norm
        cdef weight_t nr_trimmed = 0
        cdef weight_t total = 0.0
        for i in range(self.c.embed.nr):
            j = 0
            length = self.c.embed.lengths[i]
            default = self.c.embed.defaults[i]
            while Map_iter(self.c.embed.weights[i], &j, &key, &value):
                if value == NULL:
                    continue # Shouldn't happen! Raise...
                emb = <weight_t*>value
                if use_infinity_norm:
                    for i in range(length):
                        if abs(emb[i]-default[i]) >= threshold:
                            break
                    else:
                        memcpy(emb, default, sizeof(emb[0]) * length * self.c.embed.nr_support)
                        nr_trimmed += 1
                else:
                    norm = 0
                    for i in range(length):
                        norm += abs(emb[i]-default[i])
                    if norm < threshold:
                        memcpy(emb, default, sizeof(emb[0]) * length * self.c.embed.nr_support)
                        nr_trimmed += 1
                total += 1
        return nr_trimmed / total

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
        Embedding.average(self.c.embed)

    @property
    def nr_feat(self):
        return self.c.widths[0]
   
    cdef void set_scoresC(self, weight_t* scores,
                          const void* feats, int nr_feat, int is_sparse) nogil:
        fwd_state = <weight_t**>calloc(self.c.nr_layer, sizeof(void*))
        for i in range(self.c.nr_layer):
            fwd_state[i] = <weight_t*>calloc(self.c.widths[i], sizeof(weight_t))

        self._extractC(fwd_state[0], feats, nr_feat, is_sparse)
        self.c.feed_fwd(fwd_state,
            self.c.layers, NULL, self.c.widths, self.c.nr_layer, 1, &self.c.hp)
 
        memcpy(scores,
            fwd_state[self.c.nr_layer-1], sizeof(scores[0]) * self.c.widths[self.c.nr_layer-1])
        for i in range(self.c.nr_layer):
            free(fwd_state[i])
        free(fwd_state)
 
    cdef weight_t updateC(self, const void* feats, int nr_feat, int is_sparse,
                          const weight_t* costs, const int* is_valid,
                          int force_update, uint64_t key) except -1:
        is_full = self._mb.c.push_back(feats, nr_feat, is_sparse, costs, is_valid, key)
        cdef weight_t acc = 0.0
        cdef int i
        if is_full or force_update:
            self._updateC(self._mb.c)
            for i in range(self._mb.c.i):
                acc += self._mb.c.guess(i) == self._mb.c.best(i)
            for i in range(self._mb.c.i):
                if self._mb.c.is_sparse(i):
                    Embedding.insert_missing(self.mem, self.c.embed,
                        <FeatureC*>self._mb.c.features(i), self._mb.c.nr_feat(i))
            PyErr_CheckSignals()
        return acc

    cdef void _updateC(self, MinibatchC* mb) except *:
        for i in range(mb.i):
            self._extractC(mb.fwd(0, i), mb.features(i), mb.nr_feat(i), mb.is_sparse(i))
        
        randoms = <weight_t*>calloc(self.c.nr_node * mb.i, sizeof(weight_t))
        for i in range(self.c.nr_node * mb.i):
            randoms[i] = prng.get_uniform()
        self.c.feed_fwd(mb._fwd,
            self.c.layers, randoms, self.c.widths, self.c.nr_layer, mb.i, &self.c.hp)

        for i in range(mb.i):
            self._set_delta_lossC(mb.losses(i),
                mb.costs(i), mb.fwd(mb.nr_layer-1, i))
        
        self.c.feed_bwd(self.c.d_layers, mb._bwd,
            self.c.layers, mb._fwd, randoms, self.c.widths, self.c.nr_layer,
            mb.i, &self.c.hp)
        free(randoms)

        self.c.hp.t += 1
        self.c.update(self.c.weights, self.c.gradient,
            self.c.nr_weight, &self.c.hp)
        for i in range(mb.i):
            self._backprop_extracterC(mb.bwd(0, i),
                mb.features(i), mb.nr_feat(i), mb.is_sparse(i))
        for i in range(mb.i):
            self._update_extracterC(mb.features(i),
                mb.nr_feat(i), mb.i, mb.is_sparse(i))

    cdef void _extractC(self, weight_t* input_,
            const void* feats, int nr_feat, int is_sparse) nogil:
        if is_sparse:
            Embedding.set_input(input_,
                <const FeatureC*>feats, nr_feat, self.c.embed)
        else:
            memcpy(input_, feats, nr_feat * sizeof(input_[0]))
    
    cdef void _set_delta_lossC(self, weight_t* delta_loss,
            const weight_t* costs, const weight_t* scores) nogil:
        d_softmax(delta_loss,
            costs, scores, self.c.widths[self.c.nr_layer-1])

    cdef void _backprop_extracterC(self, const weight_t* deltas,
            const void* feats, int nr_feat, int is_sparse) nogil:
        if nr_feat < 1:
            return
        if is_sparse:
            Embedding.fine_tune(self.c.embed,
                deltas, self.c.widths[0], <const FeatureC*>feats, nr_feat)

    cdef void _update_extracterC(self, const void* _feats,
            int nr_feat, int batch_size, int is_sparse) nogil:
        if nr_feat < 1:
            return
        if not is_sparse:
            return
        feats = <const FeatureC*>_feats
        for feat in feats[:nr_feat]:
            Embedding.update(self.c.embed,
                feat.i, feat.key, batch_size, &self.c.hp, self.c.update)
        # Additionally, update defaults
        for i in range(self.c.embed.nr):
            length = self.c.embed.lengths[i]
            gradient = self.c.embed.d_defaults[i]
            emb = self.c.embed.defaults[i]
            for weight in gradient[:length]:
                if weight != 0.0:
                    self.c.update(emb, gradient,
                        length, &self.c.hp)
                    break

    property weights:
        def __get__(self):
            nr_weight = self.c.nr_weight * self.c.embed.nr_support
            return [self.c.weights[i] for i in range(nr_weight)]
            #cdef np.ndarray weights = np.ndarray(shape=(nr_weight,), dtype='float64')
            #for i in range(nr_weight):
            #    weights[i] = self.c.weights[i]
            #return weights
    
        def __set__(self, weights):
            for i, weight in enumerate(weights):
                self.c.weights[i] = weight
            #assert len(weights) == self.c.nr_weight * self.c.embed.nr_support
            #for i in range(weights.shape[0]):
            #    self.c.weights[i] = weights[i]

    property layers:
        # TODO: Apparent Cython bug: @property syntax fails on generators?
        def __get__(self):
            weights = list(self.weights)
            start = 0
            for i in range(self.c.nr_layer-1):
                nr_w = self.widths[i] * self.widths[i+1]
                nr_bias = self.widths[i+1]
                W = weights[start:start+nr_w]
                bias = weights[start+nr_w:start+nr_w+nr_bias]
                yield W, bias
                start += get_nr_weight(self.widths[i+1], self.widths[i],
                                       False) # TODO

    @property
    def time(self):
        return self.c.hp.t

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

    property layer_sparsity:
        def __get__(self):
            for W, bias in self.layers:
                w_sparsity = sum(w == 0 for w in W) / float(len(W))
                bias_sparsity = sum(w == 0 for w in bias) / float(len(bias))
                yield w_sparsity, bias_sparsity

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

    property eta:
        def __get__(self):
            return self.c.hp.e
        def __set__(self, eta):
            self.c.hp.e = eta

    property mu:
        def __get__(self):
            return self.c.hp.m
        def __set__(self, mu):
            self.c.hp.m = mu

    property rho:
        def __get__(self):
            return self.c.hp.r
        def __set__(self, rho):
            self.c.hp.r = rho

    property noise:
        def __get__(self):
            return self.c.hp.w
 
    property dropout:
        def __get__(self):
            return self.c.hp.d
        def __set__(self, drop_prob):
            self.c.hp.d = drop_prob
   
    @property
    def eps(self):
        return self.c.hp.p
    @eps.setter
    def eps(self, eps):
        self.c.hp.p = eps

    property tau:
        def __get__(self):
            return self.c.hp.t
        def __set__(self, tau):
            self.c.hp.t = tau
