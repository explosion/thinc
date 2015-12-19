from libc.stdint cimport int32_t
from libc.math cimport M_E
from libc.string cimport memset

from preshed.maps cimport MapStruct, map_get

from .structs cimport LayerC, FeatureC
from .typedefs cimport weight_t
from .blas cimport Vec, VecVec, MatVec, MatMat
from .api cimport arg_max_if_zero


cdef class Embedding:
    @staticmethod
    cdef inline void set_layer(weight_t* output, const MapStruct* map_,
                               const FeatureC* feats, int32_t nr_feat) nogil:
        cdef int32_t i, j
        for i in range(nr_feat):
            feat = feats[i]
            feat_embed = <const weight_t*>map_get(map_, feat.key)
            if feat_embed is not NULL:
                VecVec.add_i(&output[feat.i], feat_embed, feat.val, feat.length)

    @staticmethod
    cdef inline void fine_tune(weight_t* delta, int32_t length,
                        const FeatureC* feats, int32_t nr_feat,
                        const MapStruct* embed_map, const MapStruct* support_map) nogil:
        pass
        # tuning = weights.T.dot(delta)
        
        #for w, freq in ids.items():
        #    if w < gradient.E.shape[0]:
        #        gradient.E[w] += tuning * freq


        #cdef weight_t total = 0.0 
        #for i in range(nr_feat):
        #    total += feats[i].val
        #Vec.div_i(gradient, total, length)
        ## What do we do about regularization for these updates? Nothing?
        #for i in range(nr_feat):
        #    feat = feats[i]
        #    embed = <weight_t*>self.weights.get(feat.key)
        #    support = <weight_t*>self.train_weights.get(feat.key)
        #    if embed is not NULL and support is not NULL:
        #        # This is hardly ideal, but it lets us support different values
        #        # for now
        #        Vec.mul_i(gradient, feat.val, length)
        #        adagrad(
        #            embed,
        #            gradient,
        #            support,
        #            length,
        #            self.c.hyper_params
        #        )
        #        Vec.div_i(gradient, feat.val, length)


