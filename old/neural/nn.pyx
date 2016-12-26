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

from .solve cimport vanilla_sgd, sgd_cm, nag, adagrad, adadelta, adam

from .forward cimport softmax
from .forward cimport ELU_forward
from .forward cimport ELU_batch_norm_residual_forward
from .forward cimport ReLu_forward
from .backward cimport ELU_backward
from .backward cimport ReLu_backward
from .backward cimport ELU_batch_norm_residual_backward
from .backward cimport d_log_loss, d_hinge_loss

from .embed cimport Embedding
from .initializers cimport he_normal_initializer, he_uniform_initializer, constant_initializer

from libc.string cimport memcpy
from libc.math cimport isnan, sqrt

import random
import numpy
import cPickle


prng.normal_setup()


cdef NeuralNetC init_network(Pool mem, widths, *args, **kwargs) except *:
    cdef NeuralNetC nn
    nn.hp.e = kwargs.get('learn_rate', 0.001)
    nn.hp.r = kwargs.get('l2', 1e-6)
    nn.hp.m = kwargs.get('momentum', 0.9)

    set_update_func(&nn, kwargs)
    nr_support = set_activation_func(&nn, kwargs)

    nn.nr_layer = len(widths)
    nn.widths = <len_t*>mem.alloc(nn.nr_layer, sizeof(widths[0]))
    for i, width in enumerate(widths):
        nn.widths[i] = width
    nn.nr_weight = 0
    nn.nr_node = 0
    for i in range(nn.nr_layer-1):
        nn.nr_weight += get_nr_weight(nn.widths[i+1], nn.widths[i])
        nn.nr_node += nn.widths[i]
    nn.weights = <weight_t*>mem.alloc(nr_weight * nr_support,
                                      sizeof(nn.weights[0]))
    nn.gradient = <weight_t*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
    
    W = nn.weights
    for i in range(nn.nr_layer-2):
        he_normal_initializer(W,
            nn.widths[i+1], nn.widths[i+1] * nn.widths[i])
        nr_W = nn.widths[i+1] * nn.widths[i]
        nr_bias = nn.widths[i+1]
        W += get_nr_weight(nn.widths[i+1], nn.widths[i])


cdef int set_update_func(NeuralNetC* nn, kwargs) except -1:
    if kwargs.get('update_step') == 'sgd':
        nn.update = vanilla_sgd
        return 1
    elif kwargs.get('update_step') == 'asgd':
        nn.update = asgd
        return 2
    elif kwargs.get('update_step') == 'sgd_cm':
        nn.update = sgd_cm
        return 3
    elif kwargs.get('update_step') == 'adam':
        nn.update = adam
        return 4
    else:
        nn.update = noisy_update
        return 1


cdef int set_activation_func(NeuralNetC* nn, kwargs) except -1:
    self.c.embed.nr_support = nr_support
    use_batch_norm = kwargs.get('batch_norm', False)
    if use_batch_norm:
        nn.feed_fwd = ELU_batch_norm_residual_forward
        nn.feed_bwd = ELU_batch_norm_residual_backward
    else:
        nn.feed_fwd = ELU_forward
        nn.feed_bwd = ELU_backward


cdef int get_nr_weight(int nr_out, int nr_in, int batch_norm) nogil:
    if batch_norm:
        return nr_out * nr_in + nr_out * 5
    else:
        return nr_out * nr_in + nr_out


cdef class NeuralNet(Model):
    def __init__(self, widths, *args, **kwargs):
        self.mem = Pool()
        self.c = init_network(self.mem, widths, *args, **kwargs)
        self._mb = Minibatch(self.widths, kwargs.get('batch_size', 100))

    def __call__(self, eg_or_mb):
        cdef Example eg
        cdef Minibatch mb
        if isinstance(eg_or_mb, Example):
            eg = eg_or_mb
            self.set_scoresC(eg.c.scores, eg.c.atoms, eg.c.nr_atom)
        elif isinstance(eg_or_mb, Minibatch):
            mb = eg_or_mb
            self.c.feed_fwd(mb.c._fwd,
                self.c.weights, self.c.widths, self.c.nr_layer, mb.c.i, &self.c.hp)
            for i in range(mb.c.i):
                self._softmaxC(mb.c.fwd(self.c.nr_layer-1, i))
        return eg_or_mb

    def train(self, examples):
        cdef Example eg
        for eg in examples:
            is_full = self._mb.c.push_back(eg.c.features, eg.c.nr_feat,
                                         eg.c.costs, eg.c.is_valid, 0)
            if is_full:
                self._updateC(self._mb.c)
                yield from self._mb

    def update(self, Example eg, force_update=False):
        return self.updateC(eg.c.atoms, eg.c.nr_atom,
                            eg.c.costs, eg.c.is_valid, force_update, 0)
    
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        raise NotImplementedError

    def dump(self, loc):
        data = (self.weights, dict(self.c.hp)) 
        with open(loc, 'wb') as file_:
            cPickle.dump(data, file_, cPickle.HIGHEST_PROTOCOL)

    def load(self, loc):
        with open(loc, 'rb') as file_:
            weights, hp = cPickle.load(file_)
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
   
    cdef void set_scoresC(self, weight_t* scores, weight_t* dense_input,
            const void* sparse_input, int nr_sparse, const NeuralNetC* nn) nogil:
        if nn.prev != NULL:
            self.set_scoresC(dense_input, dense_input,
                sparse_input, nr_sparse, nn.prev)
        nn.feed_fwd(scores,
            nn.weights, nn.widths, nn.nr_layer, 1,
            dense_input, sparse_input, &nr_sparse, &nn.hp)
 
    cdef weight_t updateC(self, weight_t* d_dense_input, void* d_sparse_input,
            const weight_t* dense_input, const void* sparse_input, int nr_sparse,
            const weight_t* gradient, const int* is_valid, int force_update) nogil:
        is_full = self._mb.push_back(dense_input, sparse_input, nr_sparse, gradient, is_valid)
        if is_full or force_update:
            self._updateC(self.c, d_dense_input, d_sparse_input,
                self._mb.c.d_loss(), self._mb.c.dense_inputs(),
                self._mb.c.sparse_inputs(), self._mb.c.nr_sparses(),
                self._mb.c.i)
            return self._mb.loss()
        else:
            return 0.0
 
    cdef void _updateC(self, MinibatchC* mb) nogil:
        self.c.feed_fwd(mb._fwd,
            self.c.weights, self.c.widths, self.c.nr_layer, mb.i, &self.c.hp)

        for i in range(mb.i):
            self.outputC(mb.fwd(self.c.nr_layer-1, i),
                mb.fwd(self.c.nr_layer-1, i))
        for i in range(mb.i):
            self.set_d_lossC(mb.losses(i),
                mb.costs(i), mb.fwd(mb.nr_layer-1, i))
        
        self.c.feed_bwd(self.c.gradient, mb._bwd,
            self.c.weights, mb._fwd, self.c.widths, self.c.nr_layer,
            mb.i, &self.c.hp)

        self.c.hp.t += 1
        self.c.update(self.c.weights, self.c.gradient,
            self.c.nr_weight, &self.c.hp)

    cdef void extractC(self, weight_t* input_, const FeatureC* feats, int nr_feat) nogil:
        pass
    
    cdef void outputC(self, weight_t* output, const weight_t* last_layer) nogil:
        softmax(output, self.c.widths[self.c.nr_layer-1])

    cdef void set_d_lossC(self, weight_t* delta_loss,
            const weight_t* costs, const weight_t* scores) nogil:
        d_log_loss(delta_loss,
            costs, scores, self.c.widths[self.c.nr_layer-1])

    @property
    def weights(self):
        return [self.c.weights[i] for i in range(self.c.nr_weight)]
    
    @weights.setter
    def weights(self, weights):
        assert len(weights) == self.c.nr_weight
        cdef weight_t weight
        for i, weight in enumerate(weights):
            self.c.weights[i] = weight

    @property
    def widths(self):
        return tuple(self.c.widths[i] for i in range(self.c.nr_layer))

    @property
    def momentum(self):
        return ptr2np(&self.c.weights[self.c.nr_weight*2], self.c.nr_weight)
    @weights.setter
    def weights(self, weights):
        return np2ptr(&self.c.weights[self.c.nr_weight*2], self.c.nr_weight, weights)

    @property
    def widths(self):
        return ptr2np(self.c.widths, self.c.nr_layer)

    @property
    def gradient(self):
        cdef np.ndarray out = np.zeros(shape=(self.c.nr_weight,), dtype='float64')
        for i in range(self.c.nr_weight):
            out[i] = self.c.gradient[i]
        return out

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
    def nr_update(self):
        return self.c.hp.t
    @tau.setter
    def nr_update(self, tau):
        self.c.hp.t = tau

    @property
    def learn_rate(self):
        return self.c.hp.e
    @eta.setter
    def learn_rate(self, eta):
            self.c.hp.e = eta

    @property
    def l2(self):
        return self.c.hp.r
    @rho.setter
    def l2(self, rho):
        self.c.hp.r = rho
    
    @property
    def eps(self):
        return self.c.hp.p
    @eps.setter
    def eps(self, eps):
        self.c.hp.p = eps
