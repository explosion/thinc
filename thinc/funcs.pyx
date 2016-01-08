# cython: profile=True
# cython: cdivision=True

from .structs cimport NeuralNetC
from .structs cimport IteratorC

from .blas cimport MatVec, VecVec, Vec

from .structs cimport do_iter_t
from .structs cimport do_feed_fwd_t
from .structs cimport do_end_fwd_t
from .structs cimport do_begin_fwd_t
from .structs cimport do_begin_bwd_t
from .structs cimport do_end_bwd_t
from .structs cimport do_feed_bwd_t


cdef extern from "math.h" nogil:
    float expf(float x)
    float sqrtf(float x)


DEF EPS = 0.000001 
DEF ALPHA = 1.0


cdef class NN:
    @staticmethod
    def void init(
        NeuralNetC* nn,
            widths,
            embed=None,
            float eta=0.005,
            float eps=1e-6,
            float mu=0.2,
            float rho=1e-4,
            float bias=0.0,
            float alpha=0.0
    ) except *:
        nn.nr_layer = len(widths)
        nn.widths = <int*>mem.alloc(nr_layer, sizeof(widths[0]))
        cdef int i
        for i, width in enumerate(widths):
            nn.widths[i] = width

        nn.nr_weight = 0
        for i in range(nn.nr_layer-1):
            nn.nr_weight += NN.nr_weight(nn.widths[i+1], nn.widths[i])
        nn.weights = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.gradient = <float*>mem.alloc(nn.nr_weight, sizeof(nn.weights[0]))
        nn.opt = <OptimizerC*>self.mem.alloc(1, sizeof(OptimizerC))
        Adam.init(nn.opt, mem,
            nn.nr_weight, nn.widths, nn.nr_layer, eta, eps, rho)
        if embed is not None:
            table_widths, features = embed
            nn.embeds = <EmbeddingC*>mem.alloc(1, sizeof(EmbeddingC))
            Embedding.init(nn.embeds, mem,
                table_widths, features)
            nn.opt.embed_params = <EmbeddingC*>mem.alloc(1, sizeof(EmbeddingC))
            Embedding.init(nn.opt.embed_params, mem,
                table_widths, features)
            for i in range(nn.opt.embed_params.nr):
                # Ensure momentum terms start at zero
                memset(nn.opt.embed_params.defaults[i],
                    0, sizeof(float) * nn.opt.embed_params.lengths[i])
        
        nn.fwd_norms = <float**>mem.alloc(self.c.nr_layer*2, sizeof(void*))
        nn.bwd_norms = <float**>mem.alloc(self.c.nr_layer*2, sizeof(void*))
        fan_in = 1.0
        cdef IteratorC it
        it.i = 0
        while NN.iter(&it, nn.widths, nn.nr_layer-1, 1):
            # Allocate arrays for the normalizers
            nn.fwd_norms[it.Ex] = <float*>self.mem.alloc(it.nr_out, sizeof(float))
            nn.fwd_norms[it.Vx] = <float*>self.mem.alloc(it.nr_out, sizeof(float))
            nn.bwd_norms[it.E_dXh] = <float*>self.mem.alloc(it.nr_out, sizeof(float))
            nn.bwd_norms[it.E_dXh_Xh] = <float*>self.mem.alloc(it.nr_out, sizeof(float))
            # Don't initialize the softmax weights
            if (it.i+1) >= self.c.nr_layer:
                break
            # Do He initialization, and allow bias to be initialized to a constant.
            # Initialize the batch-norm scale, gamma, to 1.
            Initializer.normal(&self.c.weights[it.W],
                0.0, numpy.sqrt(2.0 / fan_in), it.nr_out * it.nr_in)
            Initializer.constant(&self.c.weights[it.bias],
                bias, it.nr_out)
            Initializer.constant(&self.c.weights[it.gamma],
                1.0, it.nr_out)
            fan_in = it.nr_out
        self.eg = Example(self.widths)

    @staticmethod
    cdef void predict_example(ExampleC* eg, const NeuralNetC* nn) nogil:
        set_input(eg.fwd_state[0],
            eg.atoms, eg.nr_atom, nn)
        NN.forward(eg.fwd_state,
            eg.features, eg.nr_feat, nn)
        set_scores(eg, nn)

    @staticmethod
    cdef void train_example(NeuralNetC* nn, Pool mem ExampleC* eg) except *:
        memset(nn.gradient,
            0, sizeof(nn.gradient[0]) * nn.nr_weight)
        NN.predict_example(eg,
            nn)
        for i in range(nn.embeds.nr):
            insert_sparse(nn.embeds.tables[i], mem,
                nn.embeds.defaults[i], nn.embeds.lengths[i], eg.atoms, eg.nr_atom)
            # N.B. If we switch the insert_sparse API away from taking this
            # defaults argument, ensure that we allow zero-initialization option.
            insert_sparse(nn.opt.embed_params.tables[i], mem,
                nn.opt.embed_params.defaults[i], nn.opt.embed_params.lengths[i],
                eg.atoms, eg.nr_atom)
        NN.update(nn,
            nn.gradient, eg)
     
    @staticmethod
    cdef void forward(
        float* fwd,
            const FeatureC* feats,
            int nr_feat,
            const NeuralNetC* nn
    ) nogil:
        forward(fwd,
            nn.widths, nn.nr_layer, nn.weights, nn.nr_weight, feats, nr_feat,
            &nn.alpha, nn.iterate, nn.begin_fwd, nn.feed_fwd, nn.end_fwd)

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
        dense_update(nn.weights, nn.gradient, nn.opt.params,
            nn.nr_weight, eg.bwd_state, eg.fwd_state, nn.widths, nn.nr_layer,
            nn.opt.update, nn.opt)
        for i in range(nn.embeds.nr):
            sparse_update(
                nn.embeds.tables[i],
                nn.opt.embed_params.tables[i],
                nn.gradient,
                    nn.embed.lengths,
                    nn.embed.offsets,
                    eg.bwd_state,
                    eg.atoms,
                    eg.nr_atom,
                    nn.opt,
                    nn.opt.update)


cdef void forward(
    float* fwd,
        const int* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const FeatureC* feats,
        int nr_feat,
        const void* _ext,
        do_iter_t iterate,
        do_begin_fwd_t begin_fwd,
        do_feed_fwd_t feed_fwd,
        do_end_fwd_t end_fwd
) nogil:
    cdef IteratorC it = begin_fwd(fwd,
            widths, nr_layer, weights, nr_weight, feats, nr_feat, _ext)
    while iterate(<void*>&it, widths, nr_layer-2, 1):
        feed_fwd(fwd,
            widths, nr_layer, weights, nr_weight, _ext, &it)
    end_fwd(&it, fwd,
        widths, nr_layer, weights, nr_weight, _ext)


cdef void backward(
    float* bwd,
        const float* fwd,
        const int* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const float* costs,
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
        const float* const* bwd,
        const float* const* fwd,
        const float* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const void* _ext,
        do_update_t do_update
) nogil:
    cdef IteratorC it
    it.i = 0
    while iterate(&it, widths, nr_layer):
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
    MapC* weights_table,
    MapC* moments_table,
    float* tmp,
        float* gradient,
        int length,
        uint64_t* keys,
        float* values,
        int nr_feat,
        const void* _ext
        do_update_t do_update,
) nogil:
    for i in range(nr_feat):
        weights = <float*>Map_get(weights_table, key)
        moments = <float*>Map_get(moments_table, key)
        # These should never be null.
        if weights is not NULL and moments is not NULL:
            # Copy the gradient into the temp buffer, so we can modify it in-place
            memcpy(tmp,
                gradient, sizeof(float) * length)
            Vec.mul_i(tmp,
                value, length)
            do_update(weights, moments, tmp,
                length)


cdef void dotPlus_normalize_dotPlus_ELU(
    float* fwd,
        const int* widths,
        int nr_layer,
        const float* weights,
        int nr_weight,
        const void* _it,
        const void* _ext
) nogil:
    it = <const IteratorC*>_it
    hyper_params = <const float*>_ext
    cdef float* here = fwd + it.here
    cdef float* above = fwd + it.above
    cdef float* Ex = fwd + it.Ex
    cdef float* Vx = fwd + it.Vx
    cdef const float* below = fwd + it.below
    cdef const float* W = weights + it.W
    cdef const float* bias = weights + it.bias
    cdef const float* gamma = weights + it.gamma
    cdef const float* beta = weights + it.beta
    cdef int nr_in = it.nr_in
    cdef int nr_out = it.nr_out
    cdef float ema_speed = hyper_params[0]
    dot_plus(here,
        below, W, bias, nr_out, nr_in)
    normalize(here, Ex, Vx,
        nr_out, ema_speed) 
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
        const void* _it,
        const void* _ext
) nogil:
    it = <const IteratorC*>_it
    hyper_params = <const float*>_ext
    cdef float* dX = &bwd[it.below]
    cdef float* dXh = &bwd[it.here]
    cdef float* dY = &bwd[it.above]
    cdef float* E_dXh = &bwd[it.E_dXh]
    cdef float* E_dXh_Xh = &bwd[it.E_dXh_Xh]
    cdef const float* Y = &fwd[it.above]
    cdef const float* Xh = &fwd[it.here]
    cdef const float* Vx = &fwd[it.Vx]
    cdef const float* W = &weights[it.W]
    cdef const float* gamma = &weights[it.gamma]
    cdef int nr_out = it.nr_out
    cdef int nr_in = it.nr_in
    cdef float ema_speed = hyper_params[0]
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
    float* btm_diff,
        const float* top_diff,
        const float* W,
        int nr_out,
        int nr_wide
) nogil:
    MatVec.T_dot(btm_diff,
        W, top_diff, nr_out, nr_wide)


cdef void ELU(float* out, int nr_out) nogil:
    cdef int i
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
        int n,
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


cdef void set_input(
    float* out,
    const uint64_t* keys,
    int nr_key,
        const float* defaults,
        int length,
        const MapC* table
) nogil:
    for i in range(nr_key):
        emb = <const float*>Map_get(table, keys[i])
        if emb == NULL:
            emb = defaults
        VecVec.add_i(out, 
            emb, 1.0, length)


cdef void insert_sparse(
    MapC* table,
    Pool mem,
        const float* default,
        int length, 
        const uint64_t* keys,
        int nr_feat
) except *:
    for i in range(nr_feat):
        emb = <float*>Map_get(table, keys[i])
        if emb is NULL:
            emb = <float*>mem.alloc(length, sizeof(emb[0]))
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
                default, sizeof(emb[0]) * length)
            Map_set(mem, table,
                keys[i], emb)
