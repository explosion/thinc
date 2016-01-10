# cython: profile=True
# cython: cdivision=True
from libc.string cimport memset, memcpy

cimport numpy as np

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t

from .eg cimport Example

from .structs cimport NeuralNetC
from .structs cimport IteratorC
from .structs cimport FeatureC
from .structs cimport ExampleC

from .blas cimport MatVec, VecVec, Vec

from .structs cimport do_iter_t
from .structs cimport do_feed_fwd_t
from .structs cimport do_end_fwd_t
from .structs cimport do_begin_fwd_t
from .structs cimport do_begin_bwd_t
from .structs cimport do_end_bwd_t
from .structs cimport do_feed_bwd_t
from .structs cimport do_update_t

from .blas cimport MatMat


import numpy


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


DEF EPS = 0.000001 
DEF ALPHA = 1.0


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
        nn.nr_layer = len(widths)
        nn.widths = <int*>mem.alloc(nn.nr_layer, sizeof(widths[0]))
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

        nn.embed_offsets = <int*>mem.alloc(nn.nr_embed, sizeof(nn.embed_offsets[0]))
        nn.embed_lengths = <int*>mem.alloc(nn.nr_embed, sizeof(nn.embed_offsets[0]))
        nn.embed_defaults = <float**>mem.alloc(nn.nr_embed, sizeof(nn.embed_offsets[0]))

        for i in range(nn.nr_embed):
            nn.embed_defaults[i] = <float*>mem.alloc(nn.embed_lengths[i],
                                                     sizeof(nn.embed_defaults[i][0]))
        
        cdef IteratorC it
        it.i = 0
        while NN.iter(&it, nn.widths, nn.nr_layer-1, 1):
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
    cdef inline int iter(IteratorC* it, const int* widths, int nr_layer, int inc) nogil:
        it.nr_out = widths[it.i+1]
        it.nr_in = widths[it.i]
        it.W = 0
        cdef int i
        for i in range(it.i):
            it.W += NN.nr_weight(widths[i+1], widths[i])
        it.bias = it.W + (it.nr_out * it.nr_in)
        it.gamma = it.bias + it.nr_out
        it.beta = it.gamma + it.nr_out

        it.below = it.i * 2
        it.here = it.below + 1
        it.above = it.below + 2

        it.Ex = it.here
        it.Vx = it.above
        it.E_dXh = it.here
        it.E_dXh_Xh = it.above
        it.i += inc
        if nr_layer >= it.i and it.i >= 0:
            return True
        else:
            return False

    @staticmethod
    cdef int nr_weight(int nr_out, int nr_in) nogil:
        return nr_out * nr_in + nr_out * 3

    @staticmethod
    cdef void predict_example(ExampleC* eg, const NeuralNetC* nn) nogil:
        NN.forward(eg.fwd_state,
            eg.features, eg.nr_feat, nn)
        Example.set_scores(eg, eg.fwd_state[nn.nr_layer*2-2])

    @staticmethod
    cdef void train_example(NeuralNetC* nn, Pool mem, ExampleC* eg) except *:
        memset(nn.gradient,
            0, sizeof(nn.gradient[0]) * nn.nr_weight)
        NN.predict_example(eg,
            nn)
        insert_sparse(nn.sparse_weights, mem,
            nn.embed_lengths, nn.embed_offsets, nn.embed_defaults,
            eg.features, eg.nr_feat)
        # N.B. If we switch the insert_sparse API away from taking this
        # defaults argument, ensure that we allow zero-initialization option.
        insert_sparse(nn.sparse_momentum, mem,
            nn.embed_lengths, nn.embed_offsets, nn.embed_defaults,
            eg.features, eg.nr_feat)
        NN.update(nn, eg)
     
    @staticmethod
    cdef void forward(
        float** fwd,
            const FeatureC* feats,
            int nr_feat,
            const NeuralNetC* nn
    ) nogil:
        set_input(fwd[0],
            feats, nr_feat, nn.embed_lengths, nn.embed_offsets,
            nn.embed_defaults, nn.sparse_weights) 
        forward(fwd,
            nn.widths, nn.nr_layer, nn.weights, nn.nr_weight, &nn.alpha, nn.iterate,
            nn.begin_fwd, nn.feed_fwd, nn.end_fwd)

    @staticmethod
    cdef void backward(
        float* bwd,
            const float* fwd,
            const float* costs,
            const NeuralNetC* nn
    ) nogil:
        backward(bwd,
            fwd, nn.widths, nn.nr_layer, nn.weights, nn.nr_weight, costs,
            &nn.alpha, nn.iterate, nn.begin_bwd, nn.feed_bwd, nn.end_bwd)

    @staticmethod
    cdef void update(
        NeuralNetC* nn,
            const ExampleC* eg
    ) nogil:
        dense_update(nn.weights, nn.gradient, nn.momentum,
            nn.nr_weight, eg.bwd_state, eg.fwd_state, nn.widths, nn.nr_layer,
            &nn.alpha, nn.iterate, nn.update)
        sparse_update(
            nn.sparse_weights,
            nn.sparse_momentum,
            nn.gradient,
                eg.bwd_state[0],
                nn.embed_lengths,
                nn.embed_offsets,
                eg.bwd_state,
                eg.features,
                eg.nr_feat,
                &nn.alpha,
                nn.update)


cdef void set_input(
    float* out,
        const FeatureC* feats,
        int nr_feat,
        int* lengths,
        int* offsets,
        const float* const* defaults,
        const MapC* const* tables,
) nogil:
    for f in range(nr_feat):
        emb = <const float*>Map_get(tables[feats[f].i], feats[f].key)
        if emb == NULL:
            emb = defaults[feats[f].i]
        VecVec.add_i(out, 
            emb, 1.0, lengths[feats[f].i])


