# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True

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
from ..structs cimport dense_update_t
from ..structs cimport const_weights_ft, const_dense_weights_t, const_sparse_weights_t
from ..structs cimport weights_ft, dense_weights_t, sparse_weights_t

from ..extra.eg cimport Example

from .initializers cimport he_normal_initializer, he_uniform_initializer

from libc.string cimport memcpy
from libc.math cimport isnan, sqrt

import random
import numpy


cdef class Embedding:
    cdef Pool mem
    cdef EmbedC* c

    @staticmethod
    cdef inline void init(EmbedC* self, Pool mem, vector_widths, features) except *: 
        assert max(features) < len(vector_widths), repr((features, vector_widths))
        # Create tables, which may be shared between different features
        # e.g., we might have a feature for this word, and a feature for next
        # word. These occupy different parts of the input vector, but draw
        # from the same embedding table.
        self.nr = len(features)
        uniq_weights = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        uniq_gradients = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        uniq_defaults = <weight_t**>mem.alloc(len(vector_widths), sizeof(void*))
        uniq_d_defaults = <weight_t**>mem.alloc(len(vector_widths), sizeof(void*))
        cdef int width
        cdef int i
        for i, width in enumerate(vector_widths):
            Map_init(mem, &uniq_weights[i], 8)
            Map_init(mem, &uniq_gradients[i], 8)
            # Note that we need the support parameters, because we plan to
            # learn good defaults.
            uniq_defaults[i] = <weight_t*>mem.alloc(width * self.nr_support, sizeof(weight_t))
            he_uniform_initializer(uniq_defaults[i], 0.5, -0.5, width)
            uniq_d_defaults[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
        self.offsets = <idx_t*>mem.alloc(len(features), sizeof(idx_t))
        self.lengths = <len_t*>mem.alloc(len(features), sizeof(len_t))
        self.defaults = <weight_t**>mem.alloc(len(features), sizeof(void*))
        self.d_defaults = <weight_t**>mem.alloc(len(features), sizeof(void*))
        self.weights = <MapC**>mem.alloc(len(features), sizeof(void*))
        self.gradients = <MapC**>mem.alloc(len(features), sizeof(void*))
        offset = 0
        for i, table_id in enumerate(features):
            self.weights[i] = &uniq_weights[table_id]
            self.gradients[i] = &uniq_gradients[table_id]
            self.lengths[i] = vector_widths[table_id]
            self.offsets[i] = offset
            self.defaults[i] = uniq_defaults[table_id]
            self.d_defaults[i] = uniq_d_defaults[table_id]
            offset += vector_widths[table_id]

    @staticmethod
    cdef inline void set_input(weight_t* out,
            const FeatureC* features, len_t nr_feat, const EmbedC* embed) nogil:
        for feat in features[:nr_feat]:
            if feat.value == 0 or embed.lengths[feat.i] == 0:
                continue
            emb = <const weight_t*>Map_get(embed.weights[feat.i], feat.key)
            if emb is NULL:
                # If feature is missing, we use the default values. These defaults
                # should be back-propped appropriately.
                VecVec.add_i(&out[embed.offsets[feat.i]],
                    embed.defaults[feat.i], feat.value, embed.lengths[feat.i])
            else:
                VecVec.add_i(&out[embed.offsets[feat.i]], 
                    emb, feat.value, embed.lengths[feat.i])

    @staticmethod
    cdef inline void insert_missing(Pool mem, EmbedC* embed,
            const FeatureC* features, len_t nr_feat) except *:
        cdef weight_t* grad
        cdef weight_t add_prob = 1.0
        for feat in features[:nr_feat]:
            if feat.i >= embed.nr or feat.value == 0:
                continue
            emb = <weight_t*>Map_get(embed.weights[feat.i], feat.key)
            if emb is NULL and numpy.random.random() < add_prob:
                emb = <weight_t*>mem.alloc(embed.lengths[feat.i] * embed.nr_support,
                                           sizeof(emb[0]))
                # Inherit default, including averages
                memcpy(emb,
                    embed.defaults[feat.i],
                    sizeof(emb[0]) * embed.lengths[feat.i] * embed.nr_support)
                Map_set(mem, embed.weights[feat.i],
                    feat.key, emb)
                grad = <weight_t*>mem.alloc(embed.lengths[feat.i], sizeof(grad[0]))
                Map_set(mem, embed.gradients[feat.i],
                    feat.key, grad)
    
    @staticmethod
    cdef inline void fine_tune(EmbedC* layer,
            const weight_t* delta, int nr_delta, const FeatureC* features, int nr_feat) nogil:
        cdef size_t last_update
        for feat in features[:nr_feat]:
            if feat.value == 0 or layer.lengths[feat.i] == 0:
                continue
            gradient = <weight_t*>Map_get(layer.gradients[feat.i], feat.key)
            if gradient is NULL:
                # This means the feature was missing, so we update the default
                # This allows us to learn a good default representation.
                gradient = layer.d_defaults[feat.i]
            VecVec.add_i(gradient,
                &delta[layer.offsets[feat.i]], feat.value, layer.lengths[feat.i])

    @staticmethod
    cdef inline void update(EmbedC* layer, int i, key_t key, int batch_size,
            const ConstantsC* hp, dense_update_t do_update) nogil:
        length = layer.lengths[i]
        if length == 0:
            return
        emb = <weight_t*>Map_get(layer.weights[i], key)
        gradient = <weight_t*>Map_get(layer.gradients[i], key)
        if emb is not NULL and gradient is not NULL:
            for weight in gradient[:length]:
                if weight != 0.0:
                    break
            else:
                return
            do_update(emb, gradient,
                length, hp)

    @staticmethod
    cdef inline void update_all(EmbedC* layer,
            int batch_size, const ConstantsC* hp, dense_update_t do_update) nogil:
        cdef key_t key
        cdef void* value
        cdef int i, j
        for i in range(layer.nr):
            j = 0
            length = layer.lengths[i]
            while Map_iter(layer.gradients[i], &j, &key, &value):
                grad = <weight_t*>value
                for weight in grad[:length]:
                    if weight != 0.0:
                        break
                else:
                    continue
                emb = <weight_t*>Map_get(layer.weights[i], key)
                if emb is not NULL:
                    do_update(emb, grad,
                        length, hp)
        # Additionally, update defaults
        for i in range(layer.nr):
            length = layer.lengths[i]
            for weight in layer.d_defaults[i][:length]:
                if weight != 0.0:
                    do_update(layer.defaults[i], layer.d_defaults[i],
                        layer.lengths[i], hp)
                    break

    @staticmethod
    cdef inline void average(EmbedC* layer) nogil:
        cdef key_t key
        cdef void* value
        cdef int i, j
        for i in range(layer.nr):
            j = 0
            while Map_iter(layer.weights[i], &j, &key, &value):
                emb = <weight_t*>value
                avg = emb + layer.lengths[i]
                for k in range(layer.lengths[i]):
                    emb[k] = avg[k]
        # Additionally, average defaults
        for i in range(layer.nr):
            emb = layer.defaults[i]
            avg = emb + layer.lengths[i]
            for k in range(layer.lengths[i]):
                emb[k] = avg[k]
