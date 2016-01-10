# cython: profile=True
# cython: cdivision=True
cimport cython
from libc.string cimport memcpy

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_set as Map_set

from .structs cimport IteratorC
from .structs cimport FeatureC
from .structs cimport ConstantsC

from .typedefs cimport len_t
from .typedefs cimport idx_t

from .blas cimport MatMat, MatVec, VecVec, Vec

from .structs cimport do_iter_t
from .structs cimport do_feed_fwd_t
from .structs cimport do_end_fwd_t
from .structs cimport do_begin_fwd_t
from .structs cimport do_begin_bwd_t
from .structs cimport do_end_bwd_t
from .structs cimport do_feed_bwd_t
from .structs cimport do_update_t


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


DEF EPS = 0.000001 
DEF ALPHA = 1.0


cdef int advance_iterator(
    IteratorC* it,
        const len_t* widths,
            len_t nr_layer,
        int inc) nogil:
    it.nr_out = widths[it.i+1]
    it.nr_in = widths[it.i]
    it.W = 0
    cdef int i
    for i in range(it.i):
        it.W += widths[i+1] * widths[i]
        it.W += widths[i+1]
        it.W += widths[i+1]
        it.W += widths[i+1]
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


cdef IteratorC default_begin_fwd(
    float** fwd,
        const len_t* widths,
        len_t nr_layer,
        const float* weights,
            len_t nr_weight
) nogil:
    cdef IteratorC it
    it.i = 0
    return it


cdef void default_feed_fwd(
    float** fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const IteratorC* it,
        const ConstantsC* hp
) nogil:
    dot_plus__ELU(
        fwd[it.above],
            &weights[it.bias],
                it.nr_out,
            fwd[it.below],
                it.nr_in,
            &weights[it.W])
            

cdef void default_end_fwd(
    IteratorC* it,
    float* scores,
    float** fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight) nogil:
    dot_plus(fwd[it.above],
        &weights[it.bias], it.nr_out, fwd[it.below], it.nr_in, &weights[it.W])
    softmax(fwd[it.above],
       it.nr_out)
    memcpy(scores,
        fwd[it.above], sizeof(scores[0]) * it.nr_out)


cdef IteratorC default_begin_bwd(
    float** bwd,
        const float* const* fwd,
        const len_t* widths,
        len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const float* costs
) nogil:
    cdef IteratorC it
    it.i = nr_layer-1
    advance_iterator(&it,
        widths, nr_layer, -1)
    d_log_loss(bwd[it.below],
        costs, fwd[it.below], widths[nr_layer-1])
    return it


cdef void default_end_bwd(
    IteratorC* it,
    float** bwd,
        const float* const* fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight
) nogil:
    pass


cdef void dot_plus__ELU(
    float* output,
        const float* bias,
        len_t nr_out,
        const float* input_,
            len_t nr_in,
        const float* W
) nogil:
    dot_plus(output,
        bias, nr_out, input_, nr_in, W)
    ELU(output, nr_out)


cdef void dense_update(
    float* weights,
    float* momentum,
    float* gradient,
        len_t nr_weight,
        const float* const* bwd,
        const float* const* fwd,
        const len_t* widths,
            len_t nr_layer,
        const ConstantsC* hp,
        do_iter_t iterate,
        do_update_t do_update
) nogil:
    cdef IteratorC it
    it.i = 0
    while iterate(&it, widths, nr_layer, 1):
        MatMat.add_outer_i(&gradient[it.W], # Gradient of synapse weights
            bwd[it.above], fwd[it.below], it.nr_out, it.nr_in)
        VecVec.add_i(&gradient[it.bias], # Gradient of bias weights
            bwd[it.above], 1.0, it.nr_out)
    do_update(weights, momentum, gradient,
        nr_weight, hp)


cdef void sparse_update(
    MapC** weights_tables,
    MapC** moments_tables,
    float* tmp,
        const float* gradient,
            len_t nr_grad,
        const len_t* lengths,
        const idx_t* offsets,
        const float* const* defaults,
            len_t nr_table,
        const FeatureC* feats,
            len_t nr_feat,
        const ConstantsC* hp,
        do_update_t do_update,
) nogil:
    cdef idx_t f
    cdef idx_t idx
    for f in range(nr_feat):
        idx = feats[f].i
        weights = <float*>Map_get(weights_tables[idx], feats[f].key)
        moments = <float*>Map_get(moments_tables[idx], feats[f].key)
        # These should never be null.
        if weights is not NULL and moments is not NULL:
            # Copy the gradient into the temp buffer, so we can modify it in-place
            memcpy(&tmp[offsets[idx]],
                &gradient[offsets[idx]], sizeof(float) * lengths[idx])
            Vec.mul_i(&tmp[offsets[idx]],
                feats[f].value, lengths[idx])
            do_update(&weights[offsets[idx]], &moments[offsets[idx]], &tmp[offsets[idx]],
                lengths[idx], hp)


cdef void default_feed_bwd(
    float** bwd,
        const float* const* fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const IteratorC* it,
        const ConstantsC* hp,
) nogil:
    dELU__dDot(bwd[it.below], bwd[it.above],
        it.nr_in, fwd[it.above], it.nr_out, weights)


cdef void dELU__dDot(
    float* dX,
    float* dY,
        len_t nr_wide,
        const float* Y,
        len_t nr_above,
        const float* W
) nogil:
    d_ELU(dY,
        Y, nr_above)
    d_dot(dX,
        nr_above, dY, nr_wide, W)


cdef void dot_plus(
    float* out,
        const float* bias,
            len_t nr_out,
        const float* in_,
            len_t nr_in,
        const float* W
) nogil:
    MatVec.dot(out,
        W, in_, nr_out, nr_in)
    VecVec.add_i(out,
        bias, 1.0, nr_out)


cdef void sparse_dot_plus(
    float* out,
        const float* bias,
            len_t nr_out,
        const FeatureC* feats,
            len_t nr_feat,
        const MapC* const* Ws
) nogil:
    for i in range(nr_feat):
        W = Ws[feats[i].i]
        if W is not NULL: # Shouldn't be NULL
            row = <const float*>Map_get(W, feats[i].key)
            if row is not NULL: # Can be NULL
                VecVec.add_i(out,
                    row, feats[i].value, nr_out)
    VecVec.add_i(out,
        bias, 1.0, nr_out)


cdef void d_dot(
    float* btm_diff,
        len_t nr_btm,
        const float* top_diff,
        len_t nr_top,
        const float* W,
) nogil:
    MatVec.T_dot(btm_diff,
        W, top_diff, nr_top, nr_btm)


cdef void ELU(float* out, len_t nr_out) nogil:
    cdef idx_t i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = ALPHA * (expf(out[i]) - 1)


cdef void d_ELU(float* delta, const float* signal_out, int n) nogil:
    # Backprop the ELU transformation
    # Note that this is over the function _output_, not the function
    # _input_!
    for i in range(n):
        if signal_out[i] < 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void softmax(
    float* out,
        len_t nr_out
) nogil:
    #w = exp(w - max(w))
    Vec.add_i(out,
        -Vec.max(out, nr_out), nr_out)
    Vec.exp_i(out,
        nr_out)
    #w = w / sum(w)
    cdef float norm = Vec.sum(out, nr_out)
    if norm != 0:
        Vec.div_i(out,
            norm, nr_out)


cdef void d_log_loss(
    float* loss,
        const float* costs,
        const float* scores,
            len_t nr_out
) nogil:
    # This assumes only one true class
    cdef idx_t i
    for i in range(nr_out):
        loss[i] = scores[i] - (costs[i] == 0)


cdef void set_input(
    float* out,
        const FeatureC* feats,
            len_t nr_feat,
        len_t* lengths,
        idx_t* offsets,
        const float* const* defaults,
        const MapC* const* tables,
) nogil:
    for f in range(nr_feat):
        emb = <const float*>Map_get(tables[feats[f].i], feats[f].key)
        if emb == NULL:
            emb = defaults[feats[f].i]
        VecVec.add_i(out, 
            emb, 1.0, lengths[feats[f].i])


cdef void insert_sparse(
    Pool mem,
    MapC** tables,
        const len_t* lengths, 
        const idx_t* offsets,
        const float* const* defaults,
        const FeatureC* feats,
        int nr_feat
) except *:
    for f in range(nr_feat):
        emb = <float*>Map_get(tables[feats[f].i], feats[f].key)
        if emb is NULL:
            emb = <float*>mem.alloc(lengths[feats[f].i], sizeof(emb[0]))
            # TODO: Which is better here???
            # N.B.: Careful enabling this. It can break use of this function to
            # initialize things that should be zeroed.
            #Initializer.normal(emb,
            #    0.0, 1.0, length)
            # We initialize with the defaults here so that we only have
            # to insert during training --- on the forward pass, we can
            # set default. But if we're doing that, the back pass needs
            # to be dealing with the same representation.
            memcpy(emb,
                defaults[feats[f].i], sizeof(emb[0]) * lengths[feats[f].i])
            Map_set(mem, tables[feats[f].i],
                feats[f].key, emb)


@cython.cdivision(True)
cdef void adam_update_step(
    float* weights,
    float* moments,
    float* gradient,
        len_t nr_weight,
        const ConstantsC* hp
) nogil:
    cdef float beta1 = 0.90
    cdef float beta2 = 0.999
    # Add the derivative of the L2-loss to the gradient
    cdef idx_t i
    if hp.r != 0:
        VecVec.add_i(gradient,
            weights, hp.r, nr_weight)
    # This is all vectorized and in-place, so it's hard to read. See the
    # paper.
    mom1 = moments
    mom2 = &moments[nr_weight]
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)
    Vec.mul_i(mom2,
        beta2, nr_weight)
    VecVec.mul_i(gradient,
        gradient, nr_weight)
    VecVec.add_i(mom2,
        gradient, 1-beta2, nr_weight)
    Vec.div(gradient,
        mom1, 1-beta1, nr_weight)
    for i in range(nr_weight):
        gradient[i] /= sqrtf(mom2[i] / (1-beta2)) + EPS
    Vec.mul_i(gradient,
        hp.e, nr_weight)
    VecVec.add_i(weights,
        gradient, -1.0, nr_weight)


########
# Batch Normalization, non-functional draft

#cdef void normalize(
#    float* x,
#    float* Ex,
#    float* Vx,
#        len_t nr_x,
#        float alpha
#) nogil:
#    # Upd EMA estimate of mean and variance
#    # See eq at the end of this:
#    # http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
#    cdef idx_t i
#    cdef float diff
#    cdef float incr
#    for i in range(nr_x):
#        diff = x[i] - Ex[i]
#        incr = alpha * diff
#        Vx[i] = (1.0 - alpha) * (Vx[i] + diff * incr)
#        Ex[i] += incr
#    # Normalize
#    for i in range(n):
#        x[i] = (x[i] - Ex[i]) / sqrtf(Vx[i] + EPS)
#
#
#cdef void d_normalize(
#    float* bwd,
#    float* E_dEdXh,
#    float* E_dEdXh_dot_Xh,
#        const float* Xh,
#        const float* Vx,
#            len_t n,
#        float alpha
#) nogil:
#    # Update EMA estimate of mean(dL/dX_hat)
#    Vec.mul_i(E_dEdXh,
#        alpha, n)
#    VecVec.add_i(E_dEdXh,
#        bwd, 1-alpha, n)
#    # Update EMA estimate of mean(dE/dX_hat \cdot X_hat)
#    Vec.mul_i(E_dEdXh_dot_Xh,
#        alpha, n)
#    for i in range(n):
#        E_dEdXh_dot_Xh[i] += (1-alpha) * bwd[i] * Xh[i]
#    # Simplification taken from Caffe, I think by cdoersch
#    # if X' = (X-mean(X))/sqrt(var(X)+eps), then
#    # dE/dX =
#    #   (dE/dXh - mean(dE/dXh) - mean(dE/dXh * Xh) * Xh)
#    #     ./ sqrt(var(X) + eps)
#    # bwd is dE/dXh to start with. We change it to dE/dX in-place.
#    for i in range(n):
#        bwd[i] -= E_dEdXh[i] - E_dEdXh_dot_Xh[i] * Xh[i]
#        bwd[i] /= sqrtf(Vx[i] + EPS)
#
#
#
#
#cdef void dot_plus__normalize__dot_plus__ELU(
#    float* output,
#    float* normed,
#    float* Ex,
#    float* Vx,
#        const float* bias,
#        const float* gamma,
#        len_t nr_out,
#        const float* input_,
#            len_t nr_in,
#        const weight_t* W,
#        float ema_stickiness
#) nogil:
#    dot_plus(output,
#        input_, W, bias, nr_out, nr_in)
#    normalize(normed, Ex, Vx,
#        nr_out, ema_stickiness) 
#    dot_plus(output,
#        normed, gamma, beta, nr_out, 1)
#    ELU(x_dotPlus_normalize_dotPlus_ELU,
#        nr_out)
#
#
#cdef void dELU_dDot_dNormalize_dDot(
#    float* dY,
#    float* dXh,
#    float* dX,
#    float* E_dXh,
#    float* E_dXh_Xh,
#        const float* Xh,
#        const float* Vx,
#        len_t nr_out,
#        len_t nr_in,
#        float ema_speed
#) nogil:
#    d_ELU(dY,
#        Y, nr_out) # Y = ELU(dot(G, BN(W*x+b))), i.e. our layer's final output
#    d_dot(dXh,
#        dY, gamma, nr_out, 1)
#    d_normalize(dXh, E_dXh, E_dXh_Xh,
#        Xh, Vx, nr_out, ema_speed)
#    d_dot(dX,
#        dXh, W, nr_out, nr_in)
#
#
#
