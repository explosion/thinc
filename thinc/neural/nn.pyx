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

from .solve cimport vanilla_sgd, sgd_cm, adam, adagrad

from .solve cimport adam
from .solve cimport adadelta
from .solve cimport adagrad
from .solve cimport vanilla_sgd

from .forward cimport dot_plus__ELU
from .forward cimport dot_plus__ReLu
from .backward cimport d_ELU__dot
from .backward cimport d_ReLu__dot

from .backward cimport d_log_loss

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
            update_step='adam',
            weight_t eta=0.005,
            weight_t eps=1e-6,
            weight_t mu=0.9,
            weight_t rho=1e-4,
            weight_t alpha=0.5
    ) except *:
        prng.normal_setup()
        if update_step == 'sgd':
            nn.update = vanilla_sgd
        elif update_step == 'sgd_cm':
            nn.update = sgd_cm
        elif update_step == 'adadelta':
            nn.update = adadelta
        elif update_step == 'adagrad':
            nn.update = adagrad
        else:
            nn.update = adam
        nn.feed_fwd = dot_plus__ELU
        nn.feed_bwd = d_ELU__dot

        nn.hp.t = 0
        nn.hp.a = alpha
        nn.hp.r = rho
        nn.hp.m = mu
        nn.hp.e = eta

        nn.nr_layer = len(widths)
        nn.widths = <len_t*>mem.alloc(nn.nr_layer, sizeof(widths[0]))
        nn.averages = <weight_t**>mem.alloc(nn.nr_layer, sizeof(void*))
        cdef int i
        for i, width in enumerate(widths):
            nn.widths[i] = width
            nn.averages[i] = <weight_t*>mem.alloc(width*4, sizeof(nn.averages[i][0]))
        nn.nr_weight = 0
        nn.nr_node = 0
        for i in range(nn.nr_layer-1):
            nn.nr_weight += NN.nr_weight(nn.widths[i+1], nn.widths[i])
            nn.nr_node += nn.widths[i]
        nn.weights = <weight_t*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.gradient = <weight_t*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.momentum = <weight_t*>mem.alloc(nn.nr_weight * 2, sizeof(nn.weights[0]))
        
        if embed is not None:
            vector_widths, features = embed
            Embedding.init(&nn.embed, mem, vector_widths, features)

        W = nn.weights
        for i in range(nn.nr_layer-2):
            he_uniform_initializer(W,
                nn.widths[i+1] * nn.widths[i])
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
        nn.update(nn.weights, nn.momentum, nn.gradient,
            nn.nr_weight, &nn.hp)
        if eg.nr_feat >= 1:
            Embedding.fine_tune(&nn.embed, nn.gradient,
                eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat,
                &nn.hp, nn.update)
    
    @staticmethod
    cdef void forward(weight_t* scores, weight_t** fwd,
            const NeuralNetC* nn, bint dropout) nogil:
        cdef const weight_t* W = nn.weights
        cdef uint64_t bit_mask
        cdef uint64_t j
        cdef uint64_t one = 1
        for i in range(nn.nr_layer-1):
            nn.feed_fwd(&fwd[i], nn.averages[i+1],
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
            nn.feed_bwd(G, &bwd[i], nn.averages[i+1],
                W, &fwd[i], &nn.widths[i], nn.nr_layer-(i+1), i, &nn.hp)


cdef class Embedding:
    cdef Pool mem
    cdef EmbedC* c

    @staticmethod
    cdef void init(EmbedC* self, Pool mem, vector_widths, features) except *: 
        assert max(features) < len(vector_widths)
        # Create tables, which may be shared between different features
        # e.g., we might have a feature for this word, and a feature for next
        # word. These occupy different parts of the input vector, but draw
        # from the same embedding table.
        self.nr = len(features)
        uniq_weights = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        uniq_momentum = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        for i, width in enumerate(vector_widths):
            Map_init(mem, &uniq_weights[i], 8)
            Map_init(mem, &uniq_momentum[i], 8)
        self.offsets = <idx_t*>mem.alloc(len(features), sizeof(len_t))
        self.lengths = <len_t*>mem.alloc(len(features), sizeof(len_t))
        self.weights = <MapC**>mem.alloc(len(features), sizeof(void*))
        self.momentum = <MapC**>mem.alloc(len(features), sizeof(void*))
        offset = 0
        for i, table_id in enumerate(features):
            self.weights[i] = &uniq_weights[table_id]
            self.momentum[i] = &uniq_momentum[table_id]
            self.lengths[i] = vector_widths[table_id]
            self.offsets[i] = offset
            offset += vector_widths[table_id]

    @staticmethod
    cdef void set_input(weight_t* out,
            const FeatureC* features, len_t nr_feat, const EmbedC* embed) nogil:
        for feat in features[:nr_feat]:
            emb = <const weight_t*>Map_get(embed.weights[feat.i], feat.key)
            if emb is not NULL:
                VecVec.add_i(&out[embed.offsets[feat.i]], 
                    emb, feat.value, embed.lengths[feat.i])

    @staticmethod
    cdef void insert_missing(Pool mem, EmbedC* embed,
            const FeatureC* features, len_t nr_feat) except *:
        for feat in features[:nr_feat]:
            if feat.i >= embed.nr:
                continue
            emb = <weight_t*>Map_get(embed.weights[feat.i], feat.key)
            if emb is NULL:
                emb = <weight_t*>mem.alloc(embed.lengths[feat.i], sizeof(emb[0]))
                he_uniform_initializer(emb, embed.lengths[feat.i])
                Map_set(mem, embed.weights[feat.i],
                    feat.key, emb)
                # Need 2x length for momentum. Need to centralize this somewhere =/
                mom = <weight_t*>mem.alloc(embed.lengths[feat.i] * 2, sizeof(mom[0]))
                Map_set(mem, embed.momentum[feat.i],
                    feat.key, mom)
    
    @staticmethod
    cdef inline void fine_tune(EmbedC* layer, weight_t* fine_tune,
            const weight_t* delta, int nr_delta, const FeatureC* features, int nr_feat,
            const ConstantsC* hp, do_update_t do_update) nogil:
        for feat in features[:nr_feat]:
            # Reset fine_tune, because we need to modify the gradient
            memcpy(fine_tune, delta, sizeof(weight_t) * nr_delta)
            weights = <weight_t*>Map_get(layer.weights[feat.i], feat.key)
            gradient = &fine_tune[layer.offsets[feat.i]]
            mom = <weight_t*>Map_get(layer.momentum[feat.i], feat.key)
            # None of these should ever be null
            do_update(weights, mom, gradient,
                layer.lengths[feat.i], hp)


cdef class NeuralNet:
    def __init__(self, widths, embed=None,
                 weight_t eta=0.005, weight_t eps=1e-6, weight_t mu=0.2,
                 weight_t rho=1e-4, weight_t alpha=0.99,
                 update_step='adam'):
        prng.normal_setup()
        self.mem = Pool()
        NN.init(&self.c, self.mem, widths, embed, update_step,
                eta, eps, mu, rho, alpha)
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
        NN.train_example(&self.c, self.mem, &eg.c)
        return eg
  
    def train_sparse(self, features, label):
        cdef Example eg = self.Example(features, label=label)
        NN.train_example(&self.c, self.mem, &eg.c)
        return eg
   
    def train_example(self, Example eg):
        NN.train_example(&self.c, self.mem, &eg.c)
        return eg
 
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

    property averages:
        def __get__(self):
            for i, width in enumerate(self.widths):
                yield [self.c.averages[i][j] for j in range(width*4)]

    property layers:
        def __get__(self):
            weights = list(self.weights)
            start = 0
            for i in range(self.c.nr_layer-1):
                nr_w = self.widths[i] * self.widths[i+1]
                nr_bias = self.widths[i] * self.widths[i+1] + self.widths[i+1]
                W = weights[start:start+nr_w]
                bias = weights[start+nr_w:start+nr_w+nr_bias]
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
                    mom = <weight_t*>self.mem.alloc(self.c.embed.lengths[i], sizeof(emb[0]))
                    Map_set(self.mem, self.c.embed.momentum[i], <key_t>key, mom)

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



cdef void he_normal_initializer(weight_t* weights, int fan_in, int n) except *:
    # See equation 10 here:
    # http://arxiv.org/pdf/1502.01852v1.pdf
    values = numpy.random.normal(loc=0.0, scale=numpy.sqrt(2.0 / float(fan_in)), size=n)
    cdef weight_t value
    for i, value in enumerate(values):
        weights[i] = value


cdef void he_uniform_initializer(weight_t* weights, int n) except *:
    # See equation 10 here:
    # http://arxiv.org/pdf/1502.01852v1.pdf
    values = numpy.random.randn(n) * numpy.sqrt(2.0/n)
    cdef weight_t value
    for i, value in enumerate(values):
        weights[i] = value


cdef void constant_initializer(weight_t* weights, weight_t value, int n) nogil:
    for i in range(n):
        weights[i] = value
