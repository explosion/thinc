cdef void feed_forward(float** activity, float** normalizers,
        const FeatureC* features, int nr_feat, const NeuralNetC* nn) nogil:
    cdef IteratorC it
    nn.begin_fwd(&it, activity, normalizers,
        features, nr_feat, nn)
    while nn.iter(&it, nn.widths, nn.nr_layer-2, 1):
        nn.activate(activity, normalizers,
            &it, nn.weights, nn.widths, nn.nr_layer, nn.alpha)
    nn.end_fwd(&it, activity, normalizers,
        &it, nn)


cdef void back_propagate(float** bwd, float** b_norms,
        const float* costs, const float* const* fwd, const float* const* f_norms,
        const NeuralNetC* nn) nogil:
    cdef IteratorC it
    nn.begin_bwd(&it, bwd,
        costs, fwd, nn)
    while nn.iter(&it, nn.widths, nn.nr_layer, -1):
        nn.backprop(bwd, b_norms,
            &it, fwd, f_norms, nn.weights, nn.widths, nn.nr_layer, nn.alpha)
    nn.end_bwd(&it, bwd, b_norms,
        fwd, f_norms, nn.weights, nn.widths, nn.layers, nn.alpha)
 

cdef void activate(float** fwd, float** for_bn,
        const float* weights, int n, float alpha, IteratorC it) nogil:
    above = fwd[it.above]
    here = fwd[it.here]
    below = fwd[it.below]
    Ex = for_bn[it.Ex]
    Vx = for_bn[it.Vx]
    W = &weights[it.W]
    bias = &weights[it.bias]
    gamma = &weights[it.gamma]
    nr_in = it.nr_in
    nr_out = it.nr_out

    linear(here,
        below, W, bias, nr_out, nr_in)
    update_normalizers(Ex, Vx,
        here, nr_out, alpha)
    normalize(here, Ex, Vx,
        nr_out, alpha) 
    linear(above,
        here, gamma, beta, nr_out, 1)
    ELU(above,
        nr_out)


cdef void backprop(float** bwd, float** for_bn,
        const float* weights, int n, float alpha, IteratorC it) nogil:
    d_in = bwd[it.above]
    d_out = bwd[it.below]
    W = &weights[it.W]
    bias = &weights[it.bias]
    beta = &weights[it.beta]
    gamma = &weights[it.gamma]
    
    d_ELU(bwd[it.above],
        fwd[it.above], it.nr_out)
    d_linear(bwd[it.here],
        fwd[it.here], &weights[it.gamma], it.nr_out, 1)
    d_normalize(bwd[it.below], for_bn[it.E_dEdXh], for_bn[it.E_dEdXh_dot_Xh],
        fwd[it.here], for_bn[it.Vx], it.nr_out, alpha)
    # Stash dE/dY for backprop to gamma and beta
    memcpy(b_here,
        d_out, sizeof(b_here[0]) * it.nr_wide)
    d_linear(b_below
        f_below, &weights[it.W], it.nr_out, it.nr_in)


cdef void linear(float* out,
        const float* in_, const float* W, const float* bias,
        int nr_out, int nr_wide) nogil:
    MatVec.dot(out,
        W, in_, nr_out, nr_wide)
    VecVec.add_i(out,
        bias, 1.0, nr_out)


cdef void normalize(float* x, float* Ex, float* Vx,
        int n, float alpha) nogil:
    # Upd EMA estimate of mean and variance
    # See eq at the end of this:
    # http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    cdef int i
    cdef float diff, incr
    for i in range(n):
        diff = x[i] - E_x[i]
        incr = alpha * diff
        V_x[i] = (1.0 - alpha) * (V_x[i] + diff * incr)
        E_x[i] += incr
    for i in range(n):
        x[i] = (x[i] - Ex[i]) / sqrf(V_x[i] + EPS)


cdef void softmax(float* out,
        int nr_out) nogil:
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


cdef void d_log_loss(float* loss,
        const float* costs, const float* scores, int nr_out) nogil:
    # This assumes only one true class
    cdef int i
    for i in range(nr_out):
        loss[i] = scores[i] - (costs[i] == 0)


cdef void d_linear(float* d_out,
        const float*, const float* d_in, int nr_out, int nr_wide) nogil:
    MatVec.T_dot(delta_out,
        W, delta_in, nr_out, nr_wide)


cdef void d_ELU() nogil:
    # Backprop the ELU transformation
    for i in range(nr_wide):
        if x_norm[i] < 0:
            delta_out[i] *= signal_in[i] + ALPHA


cdef void d_normalize(float* dE, float* E_dEdXh, float* E_dEdXh_dot_Xh,
        const float* Xh, const float* Vx, int n, weight_t alpha) nogil:
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


cdef void set_gradient(float* gradient,
        const float* const* bwd, const float* const* fwd, const NeuralNetC* nn) nogil:
    cdef IteratorC it
    it.i = 0
    while nn.iter(&it, nn.widths, nn.nr_layer-1, 1):
        MatMat.add_outer_i(&gradient[it.W], # Gradient of synapse weights
            bwd[it.above], fwd[it.below], it.nr_out, it.nr_in)
        VecVec.add_i(&gradient[it.bias], # Gradient of bias weights
            bwd[it.above], 1.0, it.nr_out)
        MatMat.add_outer_i(&gradient[it.gamma], # Gradient of gammas
            bwd[it.here], fwd[it.here], it.nr_out, 1)
        VecVec.add_i(&gradient[it.beta], # Gradient of betas
            bwd[it.here], 1.0, it.nr_out)


cdef void update(NeuralNetC* nn, float* gradient,
        const float* bwd float*, FeatureC* features, int nr_feat) nogil: 
    nn.opt.update(nn.opt, nn.opt.params, nn.weights, gradient,
        1.0, nn.nr_weight)
    # Fine-tune the embeddings
    if nn.embeds is not NULL and features is not NULL:
        Embedding.fine_tune(nn.opt, nn.embeds, eg.fine_tune,
            eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)
 

cdef void fine_tune(OptimizerC* opt, EmbeddingC* layer, weight_t* fine_tune,
    const weight_t* delta, int nr_delta, const FeatureC* features, int nr_feat) nogil:
    for i in range(nr_feat):
        # Reset fine_tune, because we need to modify the gradient
        memcpy(fine_tune, delta, sizeof(weight_t) * nr_delta)
        feat = features[i]
        gradient = &fine_tune[layer.offsets[feat.i]]
        weights = <weight_t*>Map_get(layer.tables[feat.i], feat.key)
        params = <weight_t*>Map_get(opt.embed_params.tables[feat.i], feat.key)
        ## These should never be null.
        if weights is not NULL and params is not NULL:
            opt.update(opt, params, weights, gradient,
                feat.val, layer.lengths[feat.i])


cdef void insert_embeddingsC(EmbeddingC* layer, Pool mem,
        const ExampleC* egs, int nr_eg) except *:
    for i in range(nr_eg):
        eg = &egs[i]
        for j in range(eg.nr_feat):
            feat = eg.features[j]
            emb = <float*>Map_get(layer.tables[feat.i], feat.key)
            if emb is NULL:
                emb = <float*>mem.alloc(layer.lengths[feat.i], sizeof(float))
                Initializer.normal(emb,
                    0.0, 1.0, layer.lengths[feat.i])
                # We initialize with the defaults here so that we only have
                # to insert during training --- on the forward pass, we can
                # set default. But if we're doing that, the back pass needs
                # to be dealing with the same representation.
                #memcpy(emb,
                #    layer.defaults[feat.i], sizeof(float) * layer.lengths[feat.i])
                Map_set(mem, layer.tables[feat.i], feat.key, emb)


