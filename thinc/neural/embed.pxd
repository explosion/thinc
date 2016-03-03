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
    cdef inline void set_input(weight_t* out,
            const FeatureC* features, len_t nr_feat, const EmbedC* embed) nogil:
        for feat in features[:nr_feat]:
            emb = <const weight_t*>Map_get(embed.weights[feat.i], feat.key)
            if emb is not NULL:
                VecVec.add_i(&out[embed.offsets[feat.i]], 
                    emb, feat.value, embed.lengths[feat.i])

    @staticmethod
    cdef inline void insert_missing(Pool mem, EmbedC* embed,
            const FeatureC* features, len_t nr_feat) except *:
        for feat in features[:nr_feat]:
            if feat.i >= embed.nr:
                continue
            emb = <weight_t*>Map_get(embed.weights[feat.i], feat.key)
            if emb is NULL:
                emb = <weight_t*>mem.alloc(embed.lengths[feat.i], sizeof(emb[0]))
                he_uniform_initializer(emb, -0.1, 0.1, embed.lengths[feat.i])
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
