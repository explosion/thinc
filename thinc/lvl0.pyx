cdef void forward(
    float** fwd,
    float** averages,
        const len_t* widths,
        len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const void* _ext,
        do_iter_t iterate,
        do_begin_fwd_t begin_fwd,
        do_feed_fwd_t feed_fwd,
        do_end_fwd_t end_fwd
) nogil:
    cdef IteratorC it
    it.i = 0
    while iterate(&it, widths, nr_layer-2, 1):
        feed_fwd(fwd,
            widths, nr_layer, weights, nr_weight, &it)
    end_fwd(&it, scores, fwd,
        widths, nr_layer, weights, nr_weight, _ext)


cdef IteratorC begin_fwd(
    float** fwd,
        const len_t* widths,
        len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const void* _ext
) nogil:
    cdef IteratorC it
    it.i = 0
    return it


cdef void dot_plus__normalize__dot_plus__ELU(
    float** fwd,
    float** averages,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const IteratorC* it,
        const ConstantsC* hp,
        const void* _,
) nogil:
    dot_plus(fwd[it.here],
        fwd[it.below], &weights[it.W], &weights[it.bias], it.nr_out, it.nr_in)
    normalize(fwd[it.here], avg[it.Ex], avg[it.Vx],
        it.nr_out, hp.alpha) 
    dot_plus(fwd[it.above],
        fwd[it.here], &weights[it.gamma], &weights[it.beta], it.nr_out, 1)
    ELU(fwd[it.above], fwd[it.here])


cdef void backward(
    float** bwd,
        const float** fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const float* costs,
            len_t nr_cost,
        const void* _ext,
        do_iter_t iterate,
        do_begin_bwd_t begin_bwd,
        do_feed_bwd_t feed_bwd,
        do_end_bwd_t end_bwd
) nogil:
    '''Iteratatively apply the step_bwd function, to back-prop through the network.
    Fills partial derivatives for each layer into bwd, so that the gradient can
    be computed. Updates estimates of normalization parameters in b_norms.'''
    cdef IteratorC it = begin_bwd(bwd,
            fwd, widths, nr_layer, weights, nr_weight, _ext)
    while iterate(&it, widths, nr_layer, -1):
        feed_bwd(bwd,
            fwd, widths, nr_layer, weights, nr_weight, &it, _ext)
    end_bwd(&it, bwd,
        fwd, widths, nr_layer, weights, nr_weight, _ext)


cdef void dense_update(
    float* weights,
    float* gradient,
    float* moments,
        len_t nr_weight,
        const float* const* bwd,
        const float* const* fwd,
        const int* widths,
            len_t nr_layer,
        const void* _ext,
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
        MatMat.add_outer_i(&gradient[it.gamma], # Gradient of gammas
            bwd[it.here], fwd[it.here], it.nr_out, 1)
        VecVec.add_i(&gradient[it.beta], # Gradient of betas
            bwd[it.here], 1.0, it.nr_out)
    do_update(weights, gradient, moments,
        nr_weight, _ext)


cdef void sparse_update(
    MapC** weights_tables,
    MapC** moments_tables,
    float* tmp,
        const float* gradient,
            len_t nr_grad,
        const len_tt* lengths,
        const idx_t* offsets,
        const float* const* defaults,
        len_t nr_table,
        const FeatureC* feats,
            len_t nr_feat,
        const void* _ext,
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
                lengths[idx], _ext)



cdef void default_feed_bwd(
    float** bwd,
    float* averages,
        const float** fwd,
        const len_t* widths,
            len_t nr_layer,
        const float* weights,
            len_t nr_weight,
        const IteratorC* _it,
        const ConstantsC* hp,
        const void* _ext
) nogil:
    dELU__dDot__dNormalize__dDot(
        bwd[it.below],
        bwd[it.here],
        bwd[it.above],
        averages[it.E_dXh],
        averages[it.E_dXh_Xh],
            &fwd[it.above],
            &fwd[it.here],
            &fwd[it.Vx],
            it.nr_out,
            it.nr_in,
            hp.a)


cdef void dELU_dDot_dNormalize_dDot(
    float* dY,
    float* dXh,
    float* dX,
    float* E_dXh,
    float* E_dXh_Xh,
        const float* Xh,
        const float* Vx,
        len_t nr_out,
        len_t nr_in,
        float ema_speed
) nogil:
    d_ELU(dY,
        Y, nr_out) # Y = ELU(dot(G, BN(W*x+b))), i.e. our layer's final output
    d_dot(dXh,
        dY, gamma, nr_out, 1)
    d_normalize(dXh, E_dXh, E_dXh_Xh,
        Xh, Vx, nr_out, ema_speed)
    d_dot(dX,
        dXh, W, nr_out, nr_in)


cdef void dot_plus(
    float* out,
        const float* bias,
            len_t nr_out,
        const float* in_,
            len_t nr_in
        const float* W,
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
            len_t nr_feat
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
        W, top_diff, nr_out, nr_wide)


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


cdef void normalize(
    float* x,
    float* Ex,
    float* Vx,
        len_t nr_x,
        float alpha
) nogil:
    # Upd EMA estimate of mean and variance
    # See eq at the end of this:
    # http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    cdef idx_t i
    cdef float diff
    cdef float incr
    for i in range(nr_x):
        diff = x[i] - Ex[i]
        incr = alpha * diff
        Vx[i] = (1.0 - alpha) * (Vx[i] + diff * incr)
        Ex[i] += incr
    # Normalize
    for i in range(n):
        x[i] = (x[i] - Ex[i]) / sqrtf(Vx[i] + EPS)


cdef void d_normalize(
    float* bwd,
    float* E_dEdXh,
    float* E_dEdXh_dot_Xh,
        const float* Xh,
        const float* Vx,
            len_t n,
        float alpha
) nogil:
    # Update EMA estimate of mean(dL/dX_hat)
    Vec.mul_i(E_dEdXh,
        alpha, n)
    VecVec.add_i(E_dEdXh,
        bwd, 1-alpha, n)
    # Update EMA estimate of mean(dE/dX_hat \cdot X_hat)
    Vec.mul_i(E_dEdXh_dot_Xh,
        alpha, n)
    for i in range(n):
        E_dEdXh_dot_Xh[i] += (1-alpha) * bwd[i] * Xh[i]
    # Simplification taken from Caffe, I think by cdoersch
    # if X' = (X-mean(X))/sqrt(var(X)+eps), then
    # dE/dX =
    #   (dE/dXh - mean(dE/dXh) - mean(dE/dXh * Xh) * Xh)
    #     ./ sqrt(var(X) + eps)
    # bwd is dE/dXh to start with. We change it to dE/dX in-place.
    for i in range(n):
        bwd[i] -= E_dEdXh[i] - E_dEdXh_dot_Xh[i] * Xh[i]
        bwd[i] /= sqrtf(Vx[i] + EPS)


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
