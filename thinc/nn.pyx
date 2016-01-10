# cython: profile=True
from __future__ import print_function

from libc.string cimport memmove, memset, memcpy

cimport cython

from cymem.cymem cimport Pool
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t

from .typedefs cimport weight_t, atom_t, feat_t
from .typedefs cimport len_t, idx_t
from .blas cimport MatMat, MatVec, VecVec, Vec
from .structs cimport MapC
from .structs cimport NeuralNetC
from .structs cimport IteratorC
from .structs cimport ExampleC
from .structs cimport FeatureC
from .structs cimport EmbedC

from .eg cimport Example

from .lvl0 cimport advance_iterator
from .lvl0 cimport adam_update_step
from .lvl0 cimport vanilla_sgd_update_step
from .lvl0 cimport advance_iterator
from .lvl0 cimport dot_plus__ELU
from .lvl0 cimport dot_plus
from .lvl0 cimport softmax
from .lvl0 cimport d_log_loss
from .lvl0 cimport d_dot
from .lvl0 cimport d_ELU

import numpy


cdef class NN:
    @staticmethod
    cdef void init(
        NeuralNetC* nn,
        Pool mem,
            widths,
            embed=None,
            update_step='adam',
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

        nn.iterate = advance_iterator
        if update_step == 'adam':
            nn.update = vanilla_sgd_update_step
        else:
            nn.update = adam_update_step

        nn.nr_weight = 0
        for i in range(nn.nr_layer-1):
            nn.nr_weight += NN.nr_weight(nn.widths[i+1], nn.widths[i])
        nn.weights = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.gradient = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.momentum = <float*>mem.alloc(nn.nr_weight*2, sizeof(nn.weights[0]))
        nn.averages = <float*>mem.alloc(nn.nr_weight*2, sizeof(nn.weights[0]))
        
        if embed is not None:
            vector_widths, features = embed
            Embedding.init(&nn.embed, mem, vector_widths, features)

        W = nn.weights
        fan_in = 1
        for i in range(nn.nr_layer-1):
            he_normal_initializer(W,
                fan_in, nn.widths[i+1] * nn.widths[i])
            constant_initializer(W,
                bias, nn.widths[i+1])
            W += nn.widths[i+1] * nn.widths[i] + nn.widths[i+1]
            fan_in = nn.widths[i]

    @staticmethod
    cdef int nr_weight(int nr_out, int nr_in) nogil:
        return nr_out * nr_in + nr_out

    @staticmethod
    cdef void predict_example(ExampleC* eg, const NeuralNetC* nn) nogil:
        NN.forward(eg.scores, eg.fwd_state,
            eg.features, eg.nr_feat, nn)

    @staticmethod
    cdef void train_example(NeuralNetC* nn, Pool mem, ExampleC* eg) except *:
        memset(nn.gradient, 0, sizeof(nn.gradient[0]) * nn.nr_weight)
        insert_embeddings(mem, nn.embed.weights, nn.embed.momentum,
            nn.embed.lengths, nn.embed.offsets, nn.embed.defaults,
            eg.features, eg.nr_feat)
        NN.predict_example(eg, nn)
        NN.backward(eg.bwd_state,
            eg.fwd_state, eg.costs, nn)
        NN.update(nn, eg)
     
    @staticmethod
    cdef void forward(float* scores, float** fwd, const FeatureC* feats,
                      int nr_feat, const NeuralNetC* nn) nogil:
        NN.set_input(fwd[0],
            feats, nr_feat, nn.embed.lengths, nn.embed.offsets,
            nn.embed.defaults, nn.embed.weights) 
        cdef int i
        cdef const float* W = nn.weights
        cdef const float* bias = W + (nn.widths[0] * nn.widths[1])
        for i in range(nn.nr_layer-2): # Save last layer for softmax
            dot_plus__ELU(fwd[i+1],
                bias, nn.widths[i+1], fwd[i], nn.widths[i], W)
            W += nn.widths[i+1] * nn.widths[i] + nn.widths[i+1]
            bias += nn.widths[i+1] * nn.widths[i] + nn.widths[i+1]
        dot_plus(fwd[nn.nr_layer-1],
            bias, nn.widths[nn.nr_layer-1], fwd[nn.nr_layer-2], nn.widths[nn.nr_layer-2],
            W)
        softmax(fwd[nn.nr_layer-1],
            nn.widths[nn.nr_layer-1])
        memcpy(scores,
            fwd[nn.nr_layer-1], sizeof(scores[0]) * nn.widths[nn.nr_layer-1])

    @staticmethod
    cdef void backward(float** bwd,
            const float* const* fwd, const float* costs, const NeuralNetC* nn) nogil:
        d_log_loss(bwd[nn.nr_layer-1],
            costs, fwd[nn.nr_layer-1], nn.widths[nn.nr_layer-1])
        cdef const float* W = nn.weights + nn.nr_weight
        cdef int i
        for i in range(nn.nr_layer-2, 0, -1):
            W -= nn.widths[i+1] * nn.widths[i] + nn.widths[i+1]
            d_ELU(bwd[i+1],
                fwd[i+1], nn.widths[i+1])
            d_dot(bwd[i],
                nn.widths[i], bwd[i+1], nn.widths[i+1], W)
        W -= nn.widths[1] * nn.widths[0] + nn.widths[1]
        d_dot(bwd[0],
            nn.widths[0], bwd[1], nn.widths[1], W)
    
    @staticmethod
    cdef void update(NeuralNetC* nn, const ExampleC* eg) nogil:
        cdef int i
        memset(nn.gradient, 0, sizeof(nn.gradient[0]) * nn.nr_weight)
        cdef float* G = nn.gradient
        for i in range(nn.nr_layer-1):
            MatMat.add_outer_i(G,
                eg.bwd_state[i+1], eg.fwd_state[i], nn.widths[i+1], nn.widths[i])
            VecVec.add_i(G + (nn.widths[i+1] * nn.widths[i]),
                eg.bwd_state[i+1], 1.0, nn.widths[i+1])
            G += (nn.widths[i+1] * nn.widths[i]) + nn.widths[i+1]
        nn.update(nn.weights, nn.momentum, nn.gradient,
            nn.nr_weight, &nn.hp)
        cdef idx_t f
        cdef idx_t idx
        cdef idx_t os
        cdef float* emb
        cdef float* mom
        cdef float* upd = nn.gradient
        for f in range(eg.nr_feat):
            # Copy the fine-tuning into the temp buffer, so we can modify it in-place
            memcpy(upd, eg.bwd_state[0], sizeof(upd[0]) * nn.widths[0])
            idx = eg.features[f].i
            os = nn.embed.offsets[idx]
            with gil:
                print(f, idx, os, eg.features[f].value, nn.embed.lengths[idx])
            emb = <float*>Map_get(nn.embed.weights[idx], eg.features[f].key)
            mom = <float*>Map_get(nn.embed.momentum[idx], eg.features[f].key)
            # These should never be null.
            if emb is not NULL and mom is not NULL:
                Vec.mul_i(&upd[os],
                    eg.features[f].value, nn.embed.lengths[idx])
                nn.update(emb, mom, &upd[os],
                    nn.embed.lengths[idx], &nn.hp)


    @staticmethod
    cdef void set_input(
        float* out,
            const FeatureC* feats,
                len_t nr_feat,
            len_t* lengths,
            idx_t* offsets,
            const float* const* defaults,
            const MapC* const* tables) nogil:
        for f in range(nr_feat):
            emb = <const float*>Map_get(tables[feats[f].i], feats[f].key)
            if emb == NULL:
                emb = defaults[feats[f].i]
            VecVec.add_i(&out[offsets[feats[f].i]], 
                emb, feats[f].value, lengths[feats[f].i])



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
        uniq_weights = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        uniq_momentum = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        uniq_defaults = <float**>mem.alloc(len(vector_widths), sizeof(void*))
        for i, width in enumerate(vector_widths):
            Map_init(mem, &uniq_weights[i], 8)
            Map_init(mem, &uniq_momentum[i], 8)
            uniq_defaults[i] = <float*>mem.alloc(width, sizeof(float))
            he_normal_initializer(uniq_defaults[i],
                1, width)
        self.offsets = <idx_t*>mem.alloc(len(features), sizeof(len_t))
        self.lengths = <len_t*>mem.alloc(len(features), sizeof(len_t))
        self.weights = <MapC**>mem.alloc(len(features), sizeof(void*))
        self.momentum = <MapC**>mem.alloc(len(features), sizeof(void*))
        self.defaults = <float**>mem.alloc(len(features), sizeof(void*))
        offset = 0
        for i, table_id in enumerate(features):
            self.weights[i] = &uniq_weights[table_id]
            self.momentum[i] = &uniq_momentum[table_id]
            self.lengths[i] = vector_widths[table_id]
            self.defaults[i] = uniq_defaults[table_id]
            self.offsets[i] = offset
            offset += vector_widths[table_id]


cdef class NeuralNet:
    cdef readonly Pool mem
    cdef readonly Example eg
    cdef NeuralNetC c

    def __init__(self, widths, embed=None, weight_t eta=0.005, weight_t eps=1e-6,
                 weight_t mu=0.2, weight_t rho=1e-4, weight_t bias=0.0, weight_t alpha=0.0,
                 update_step='adam'):
        self.mem = Pool()
        NN.init(&self.c, self.mem, widths, embed, update_step, eta, eps, mu, rho, bias, alpha)
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

    def train_dense(self, features, y):
        memset(self.c.gradient,
            0, sizeof(self.c.gradient[0]) * self.c.nr_weight)
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_input(features)
        eg.set_label(y)
        NN.train_example(&self.c, self.mem, &eg.c)
        return eg
  
    def train_sparse(self, features, y):
        memset(self.c.gradient,
            0, sizeof(self.c.gradient[0]) * self.c.nr_weight)
        cdef Example eg = self.Example(features)
        eg.wipe(self.widths)
        eg.set_label(y)
        NN.train_example(&self.c, self.mem, &eg.c)
        return eg
 
    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        cdef Example eg = Example(self.widths)
        eg.set_features(input_)
        if label is not None:
            eg.set_label(label)
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
            for i in range(self.c.embed.nr):
                j = 0
                while Map_iter(self.c.embed.weights[i], &j, &key, &value):
                    emb = <weight_t*>value
                    yield key, [emb[k] for k in range(self.c.embed.lengths[i])]

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


cdef void insert_embeddings(
    Pool mem,
    MapC** weights,
    MapC** momentum,
        const len_t* lengths, 
        const idx_t* offsets,
        const float* const* defaults,
        const FeatureC* feats,
        int nr_feat
) except *:
    for f in range(nr_feat):
        emb = <float*>Map_get(weights[feats[f].i], feats[f].key)
        if emb is NULL:
            emb = <float*>mem.alloc(lengths[feats[f].i], sizeof(emb[0]))
            # TODO: Which is better here???
            he_normal_initializer(emb, 1, lengths[feats[f].i])
            # We initialize with the defaults here so that we only have
            # to insert during training --- on the forward pass, we can
            # set default. But if we're doing that, the back pass needs
            # to be dealing with the same representation.
            Map_set(mem, weights[feats[f].i],
                feats[f].key, emb)
            mom = <float*>mem.alloc(lengths[feats[f].i], sizeof(mom[0]))
            Map_set(mem, momentum[feats[f].i],
                feats[f].key, mom)


cdef void he_normal_initializer(float* weights, int fan_in, int n) except *:
    # See equation 10 here:
    # http://arxiv.org/pdf/1502.01852v1.pdf
    values = numpy.random.normal(loc=0.0, scale=numpy.sqrt(2.0 / float(fan_in)), size=n)
    for i, value in enumerate(values):
        weights[i] = value


cdef void constant_initializer(float* weights, float value, int n) nogil:
    for i in range(n):
        weights[i] = value
