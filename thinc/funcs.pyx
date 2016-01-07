from .structs cimport NeuralNetC, FeatureC
from .structs cimport IteratorC


cdef class NN:
    @staticmethod
    cdef void forward(
        float* fwd,
            const FeatureC* feats,
            int nr_feat,
            const NeuralNetC* nn
    ) nogil:
        forward(fwd,
            nn.widths, nn.nr_layer, nn.weights, nn.nr_weight, feats, nr_feat,
            nn.hyper_params, nn.iterate, nn.begin_fwd, nn.end_fwd, nn.feed_fwd)

    @staticmethod
    cdef void backward(
        float* bwd,
            const float* fwd,
            const NeuralNetC* nn
    ) nogil:
        backward(bwd,
            fwd, nn.widths, nn.nr_layer, nn.weights, nn.nr_weight, costs,
            nn.hyper_params, nn.iterate, nn.begin_bwd, nn.end_bwd, nn.feed_bwd)


cdef void forward(
    float* fwd,
        const int* widths,
        int nr_layer,
        const FeatureC* feats,
        int nr_feat,
        const float* weights,
        int nr_weight,
        const ConstantsC* hyper_params,
        do_iter_t iterate,
        do_begin_fwd_t begin_fwd,
        do_end_fwd_t end_fwd
        do_feed_fwd_t feed_fwd,
) nogil:
    cdef IteratorC it = begin_fwd(fwd,
            widths, nr_layer, weights, nr_weight, features, nr_feat, hyper_params)
    while iterate(<void*>&it, widths, nr_layer-2, 1):
        feed_fwd(fwd,
            widths, nr_layer, weights, nr_weight, hyper_params, &it)
    end_fwd(&it, fwd,
        widths, nr_layer, weights, nr_weight, hyper_params)


cdef void backward(
    float* bwd,
        const float* fwd,
        const int* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const float* costs,
        const ConstantsC* hyper_params,
        do_iter_t iterate,
        do_begin_bwd_t begin_bwd,
        do_feed_bwd_t step_bwd,
        do_end_bwd_t end_bwd
) nogil:
    '''Iteratatively apply the step_bwd function, to back-prop through the network.
    Fills partial derivatives for each layer into bwd, so that the gradient can
    be computed. Updates estimates of normalization parameters in b_norms.'''
    cdef IteratorC it = begin_bwd(bwd,
                            fwd, widths, nr_layer, weights, nr_weight, hyper_params)
    while iterate(&it, widths, nr_layer, -1):
        feed_bwd(bwd,
            fwd, widths, nr_layer, weights, nr_weight, hyper_params, &it)
    end_bwd(&it, bwd,
        fwd, widths, nr_layer, weights, nr_weight, hyper_params)


cdef void dotPlus_normalize_dotPlus_ELU(
    float* fwd,
        const int* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const ConstantsC* hyper_params,
        const void* _it
) nogil:
    it = <IteratorC*>_it
    cdef float* here = fwd + it.here
    cdef float* above = fwd + it.above
    cdef const float* below = fwd + it.below
    cdef const float* W = weights + it.W
    cdef const float* bias = weights + it.bias
    cdef const float* gamma = weights + it.gamma
    cdef const float* beta = weights + it.beta
    cdef const float* Ex = fwd + it.Ex
    cdef const float* Vx = fwd + it.Vx
    cdef int nr_in = it.nr_in
    cdef int nr_out = it.nr_out
    cdef float ema_speed = hyper_params.alpha
    dot_plus(here,
        below, W, bias, nr_out, nr_in)
    normalize(here, Ex, Vx,
        here, nr_out, ema_speed) 
    dot_plus(above,
        here, gamma, beta, nr_out, 1)
    ELU(above,
        nr_out)


cdef void dELU_dDot_dNormalize_dDot(
    float* bwd,
        const float* fwd,
        const int* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const ConstantsC* hyper_params
        const void* _it
) nogil:
    it = <IteratorC*>it
    cdef float* dEdX = &bwd[it.below]
    cdef float* dEdXh = &bwd[it.here]
    cdef float* dEdY = &bwd[it.above]
    cdef float* E_dEdXh = &bwd[it.E_dEdXh]
    cdef float* E_dEdXh_dot_Xh = &bwd[it.E_dEdXh_dot_Xh]
    cdef const float* Y = &fwd[it.above]
    cdef const float* Xh = &fwd[it.here]
    cdef const float* Vx = &fwd[it.Vx]
    cdef const float* W = &weights[it.W]
    cdef const float* gamma = &weights[it.gamma]
    cdef int nr_out = it.nr_out
    cdef int nr_in = it.nr_in
    cdef float ema_speed = hyper_params.ema_speed
    d_ELU(dEdY,
        Y, nr_out) # Y = ELU(dot(G, BN(W*x+b))), i.e. our layer's final output
    d_dot(dEdXh,
        dEdY, gamma, nr_out, 1)
    d_normalize(dEdXh, E_dEdXh, E_dEdXh_dot_Xh,
        Xh, Vx, nr_out, ema_speed)
    d_dot(dEdX,
        dEdXh, W, nr_out, nr_in)


cdef void dot_plus(
    float* out,
        const float* in_,
        const float* W,
        const float* bias,
        int nr_out,
        int nr_wide
) nogil:
    MatVec.dot(out,
        W, in_, nr_out, nr_wide)
    VecVec.add_i(out,
        bias, 1.0, nr_out)


cdef void d_dot(
    float* d_out,
        const float*,
        const float* d_in,
        int nr_out,
        int nr_wide
) nogil:
    MatVec.T_dot(delta_out,
        W, delta_in, nr_out, nr_wide)


cdef void ELU(weight_t* out, int nr_out) nogil:
    cdef int i
    for i in range(nr_out):
        if out[i] < 0:
            out[i] = ALPHA * (expf(out[i])-1)


cdef void d_ELU(float* delta, const float* signal_out, int n) nogil:
    # Backprop the ELU transformation
    for i in range(n):
        if signal_out[i] < 0:
            delta[i] *= signal_out[i] + ALPHA


cdef void normalize(
    float* x,
    float* Ex,
    float* Vx,
        int n,
        float alpha
) nogil:
    # Upd EMA estimate of mean and variance
    # See eq at the end of this:
    # http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    cdef int i
    cdef float diff
    cdef float incr
    for i in range(n):
        diff = x[i] - E_x[i]
        incr = alpha * diff
        V_x[i] = (1.0 - alpha) * (V_x[i] + diff * incr)
        E_x[i] += incr
    # Normalize
    for i in range(n):
        x[i] = (x[i] - Ex[i]) / sqrf(V_x[i] + EPS)


cdef void d_normalize(
    float* dE,
    float* E_dEdXh,
    float* E_dEdXh_dot_Xh,
        const float* Xh,
        const float* Vx,
        int n,
        float alpha
) nogil:
    # Update EMA estimate of mean(dL/dX_hat)
    Vec.mul_i(E_bwd,
        alpha, n)
    VecVec.add_i(E_bwd,
        bwd, 1-alpha, n)
    # Update EMA estimate of mean(dE/dX_hat \cdot X_hat)
    Vec.mul_i(E_bwd_dot_fwd,
        alpha, n)
    for i in range(n):
        E_bwd_dot_fwd[i] += (1-alpha) * bwd[i] * fwd[i]
    # Simplification taken from Caffe, I think by cdoersch
    # if X' = (X-mean(X))/sqrt(var(X)+eps), then
    # dE/dX =
    #   (dE/dX' - mean(dE/dX') - mean(dE/dX' * X') * X')
    #     ./ sqrt(var(X) + eps)
    # bwd is dE/dX' to start with. We change it to dE/dX in-place.
    for i in range(n):
        bwd[i] -= E_dEdXh[i] - E_dEdXh_dot_Xh[i] * Xh[i]
        bwd[i] /= c_sqrt(V_x[i] + EPS)


cdef void softmax(
    float* out,
        int nr_out
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
        int nr_out
) nogil:
    # This assumes only one true class
    cdef int i
    for i in range(nr_out):
        loss[i] = scores[i] - (costs[i] == 0)

#
#cdef void set_gradient(
#    float* gradient,
#        const float* const* bwd,
#        const float* const* fwd,
#        const NeuralNetC* nn
#) nogil:
#    cdef IteratorC it
#    it.i = 0
#    while nn.iter(&it, nn.widths, nn.nr_layer-1, 1):
#        MatMat.add_outer_i(&gradient[it.W], # Gradient of synapse weights
#            bwd[it.above], fwd[it.below], it.nr_out, it.nr_in)
#        VecVec.add_i(&gradient[it.bias], # Gradient of bias weights
#            bwd[it.above], 1.0, it.nr_out)
#        MatMat.add_outer_i(&gradient[it.gamma], # Gradient of gammas
#            bwd[it.here], fwd[it.here], it.nr_out, 1)
#        VecVec.add_i(&gradient[it.beta], # Gradient of betas
#            bwd[it.here], 1.0, it.nr_out)
#
#
#cdef void update(
#    NeuralNetC* nn,
#    float* gradient,
#        const float* const* bwd,
#        FeatureC* features,
#        int nr_feat
#) nogil: 
#    nn.opt.update(nn.opt, nn.opt.params, nn.weights, gradient,
#        1.0, nn.nr_weight)
#    # Fine-tune the embeddings
#    if nn.embeds is not NULL and features is not NULL:
#        Embedding.fine_tune(nn.opt, nn.embeds, eg.fine_tune,
#            eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)
# 
#
#cdef void fine_tune(
#    OptimizerC* opt,
#    EmbeddingC* layer,
#    weight_t* fine_tune,
#        const weight_t* delta,
#        int nr_delta,
#        const FeatureC* features,
#        int nr_feat
#) nogil:
#    for i in range(nr_feat):
#        # Reset fine_tune, because we need to modify the gradient
#        memcpy(fine_tune, delta, sizeof(weight_t) * nr_delta)
#        feat = features[i]
#        gradient = &fine_tune[layer.offsets[feat.i]]
#        weights = <weight_t*>Map_get(layer.tables[feat.i], feat.key)
#        params = <weight_t*>Map_get(opt.embed_params.tables[feat.i], feat.key)
#        ## These should never be null.
#        if weights is not NULL and params is not NULL:
#            opt.update(opt, params, weights, gradient,
#                feat.val, layer.lengths[feat.i])
#
#
#cdef void insert_embeddingsC(
#    EmbeddingC* layer,
#    Pool mem,
#        const ExampleC* egs,
#        int nr_eg)
#    except *:
#
#    for i in range(nr_eg):
#        eg = &egs[i]
#        for j in range(eg.nr_feat):
#            feat = eg.features[j]
#            emb = <float*>Map_get(layer.tables[feat.i], feat.key)
#            if emb is NULL:
#                emb = <float*>mem.alloc(layer.lengths[feat.i], sizeof(float))
#                Initializer.normal(emb,
#                    0.0, 1.0, layer.lengths[feat.i])
#                # We initialize with the defaults here so that we only have
#                # to insert during training --- on the forward pass, we can
#                # set default. But if we're doing that, the back pass needs
#                # to be dealing with the same representation.
#                #memcpy(emb,
#                #    layer.defaults[feat.i], sizeof(float) * layer.lengths[feat.i])
#                Map_set(mem, layer.tables[feat.i], feat.key, emb)
