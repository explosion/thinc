cimport cython
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt
from libc.stdint cimport int32_t
import numpy
import numpy.random

from cymem.cymem cimport Pool

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_set as Map_set

from .structs cimport NeuralNetC, OptimizerC, FeatureC, BatchC, ExampleC, EmbeddingC, MapC
from .typedefs cimport weight_t
from .blas cimport Vec, MatMat, MatVec, VecVec
from .eg cimport Batch, Example

# The input/output of the fwd/bwd pass can be confusing. Some notes.
#
# Forward pass. in0 is at fwd_state[0]. Activation of layer 1 is
# at fwd_state[1]
# 
# in0 = input_
# in1 = act0 = ReLu(in0 * W0 + b0)
# in2 = act1 = ReLu(in1 * W1 + b1)
# out = act2 = Softmax(in2 * W2 + b2)

# Okay so our scores are at fwd_state[3]. Our loss will live there too.
# The loss will then be used to calculate the gradient for layer 2.
# We now sweep backward, and calculate the next loss, which will be used
# to calculate the gradient for layer 1, etc.
#
# So, the total loss is at bwd_state[3]
# 
# g2 = d3 = out - target
# g1 = d2 = Back(d3, in2, w2, b2)
# g0 = d1 = Back(d2, in1, w1, b1)
# gE = d0 = Back(d1, in0, w0, b0)
# 
# gE here refers to the 'fine tuning' vector, for word embeddings


cdef class NeuralNet:
    cdef Pool mem
    cdef NeuralNetC c

    @staticmethod
    cdef inline void predictC(ExampleC* egs, int nr_eg, const NeuralNetC* nn) nogil:
        for i in range(nr_eg):
            eg = &egs[i]
            if nn.embeds is not NULL and eg.features is not NULL:
                Embedding.set_input(eg.fwd_state[0],
                    eg.features, eg.nr_feat, nn.embeds)
            NeuralNet.forward(eg.fwd_state,
                nn.weights, nn.widths, nn.nr_layer)
            Example.set_scores(eg,
                eg.fwd_state[nn.nr_layer-1])
     
    @staticmethod
    cdef inline void updateC(NeuralNetC* nn, weight_t* gradient,
                             ExampleC* egs, int nr_eg) nogil:
        for i in range(nr_eg):
            eg = &egs[i]
            NeuralNet.backward(eg.bwd_state,
                eg.costs, eg.fwd_state, nn.weights + nn.nr_weight, nn.widths,
                nn.nr_layer)
        # Get the averaged gradient for the minibatch
        for i in range(nr_eg):
            NeuralNet.set_gradient(gradient,
                egs[i].fwd_state, egs[i].bwd_state, nn.widths, nn.nr_layer)
        Vec.div_i(gradient,
            nr_eg, nn.nr_weight)
        nn.opt.update(nn.opt, nn.weights, gradient,
            1.0, nn.nr_weight)
        # Fine-tune the embeddings
        # This is sort of wrong --- we're supposed to average over the minibatch.
        # However, most words are rare --- so most words will only have non-zero
        # gradient for 1 or 2 examples anyway.
        if nn.embeds is not NULL:
            for i in range(nr_eg):
                eg = &egs[i]
                if eg.features is not NULL:
                    Embedding.fine_tune(nn.opt, nn.embeds, eg.fine_tune,
                        eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)

    @staticmethod
    cdef inline void batch_norm_training(NeuralNetC* nn, BatchC* mb) except *:
        # Allocate. All this gets freed when mem gets cleaned up.
        cdef Pool mem = Pool()
        eg_fwd = <weight_t***>mem.alloc(nn.nr_layer, sizeof(void*))
        eg_bwd = <weight_t***>mem.alloc(nn.nr_layer, sizeof(void*))
        bn_fwd = <weight_t***>mem.alloc(nn.nr_layer, sizeof(void*))
        bn_bwd = <weight_t***>mem.alloc(nn.nr_layer, sizeof(void*))
        for i in range(nn.nr_layer):
            eg_fwd[i] = <weight_t**>mem.alloc(mb.nr_eg, sizeof(void*))
            eg_bwd[i] = <weight_t**>mem.alloc(mb.nr_eg, sizeof(void*))
            bn_fwd[i] = <weight_t**>mem.alloc(mb.nr_eg, sizeof(void*))
            bn_bwd[i] = <weight_t**>mem.alloc(mb.nr_eg, sizeof(void*))
            for j in range(mb.nr_eg):
                eg_fwd[i][j] = mb.egs[j].fwd_state[i]
                eg_bwd[i][j] = mb.egs[j].bwd_state[i]
                bn_fwd[i][j] = <weight_t*>mem.alloc(nn.widths[i], sizeof(weight_t))
                bn_bwd[i][j] = <weight_t*>mem.alloc(nn.widths[i], sizeof(weight_t))
        fwd_avg = <weight_t**>mem.alloc(nn.nr_layer, sizeof(void*))
        bwd_avg = <weight_t**>mem.alloc(nn.nr_layer, sizeof(void*))
        fwd_var = <weight_t**>mem.alloc(nn.nr_layer, sizeof(void*))
        bwd_var = <weight_t**>mem.alloc(nn.nr_layer, sizeof(void*))
        for i in range(nn.nr_layer):
            fwd_avg[i] = <weight_t*>mem.alloc(nn.widths[i], sizeof(weight_t))
            bwd_avg[i] = <weight_t*>mem.alloc(nn.widths[i], sizeof(weight_t))
            fwd_var[i] = <weight_t*>mem.alloc(nn.widths[i], sizeof(weight_t))
            bwd_var[i] = <weight_t*>mem.alloc(nn.widths[i], sizeof(weight_t))
        
        # TODO
        cdef weight_t* bn_W
        cdef weight_t* d_bn_W
        cdef weight_t* tmp
        # Embed
        for i in range(mb.nr_eg):
            if nn.embeds is not NULL and mb.egs[i].features is not NULL:
                Embedding.set_input(eg_fwd[i][0],
                    mb.egs[i].features, mb.egs[i].nr_feat, nn.embeds)

        # Forward
        cdef weight_t* W = nn.weights
        for i in range(nn.nr_layer): # Save last layer for softmax
            BN.mean(fwd_avg[i],
                eg_fwd[i], mb.nr_eg, nn.widths[i])
            BN.variance(fwd_var[i],
                eg_fwd[i], fwd_avg[i], mb.nr_eg, nn.widths[i])
            for j in range(mb.nr_eg):
                BN.forward(bn_fwd[i][j],
                    eg_fwd[i][j], bn_W, fwd_avg[i], fwd_var[i], nn.widths[i])
                Rectifier.forward(eg_fwd[i+1][j],
                    bn_fwd[i][j], W, nn.widths[i+1], nn.widths[i])
            W += nn.widths[i+1] * nn.widths[i] + nn.widths[i+1]
            bn_W += nn.widths[i+1]
        i_n1 = nn.nr_layer-1
        i_n2 = nn.nr_layer-2
        BN.mean(fwd_avg[i_n1],
            eg_fwd[i_n1], mb.nr_eg, nn.widths[i_n1])
        BN.variance(fwd_var[i_n1],
            eg_fwd[i_n1], fwd_avg[i_n1], mb.nr_eg, nn.widths[i_n1])
        for i in range(mb.nr_eg):
            BN.forward(bn_fwd[i_n1][i],
                eg_fwd[i_n1][i], bn_W, fwd_avg[i_n1], fwd_var[i_n1], nn.widths[i_n1])
            Softmax.forward(eg_fwd[i_n1][i],
                eg_fwd[i_n2][i], W, nn.widths[i_n1], nn.widths[i_n2])
        # Backward
        for i in range(nn.nr_layer-2, 0, -1):
            W -= nn.widths[i+1] * nn.widths[i] + nn.widths[i+1]
            BN.delta_variance(bwd_var[i],
                fwd_avg[i], fwd_var[i], bn_bwd[i+1], eg_fwd[i], mb.nr_eg, nn.widths[i])
            BN.delta_mean(bwd_avg[i], tmp,
                bwd_var[i], bn_bwd[i],
                eg_fwd[i],
                fwd_avg[i], fwd_var[i], mb.nr_eg, nn.widths[i])
            for j in range(mb.nr_eg):
                BN.backward(bn_bwd[i][j],
                    d_bn_W, eg_bwd[i][j], eg_fwd[i][j], bn_W, fwd_avg[i], fwd_var[i],
                    bwd_avg[i], bwd_var[i], mb.nr_eg, nn.widths[i+1], nn.widths[i])
                # TODO: We have to multiply bn_bwd[i] with gamma
                Rectifier.backward(eg_bwd[i][j], # Output: error of this layer, len=width
                    bn_bwd[i+1][j],    # Input: error from layer above, len=nr_out
                    bn_fwd[i][j],      # Input: signal from layer below, len=nr_wide
                    W,              # Weights of this layer
                    nn.widths[i+1],    # Width of next layer 
                    nn.widths[i]       # Width of this layer 
                )
        # The delta at bwd_state[0] can be used to 'fine tune' e.g. word vectors
        W -= nn.widths[1] * nn.widths[0] + nn.widths[1]
        for i in range(mb.nr_eg):
            MatVec.T_dot(bn_bwd[0][i],
                W, bn_bwd[1][i], nn.widths[1], nn.widths[0])
        # Get the averaged gradient for the minibatch
        # We compute this over the batch norms
        for i in range(mb.nr_eg):
            NeuralNet.set_gradient(mb.gradient,
                bn_fwd[i], bn_bwd[i], nn.widths, nn.nr_layer)
        nn.opt.update(nn.opt, nn.weights, mb.gradient,
            1.0, nn.nr_weight)
        # Fine-tune the embeddings
        # This is sort of wrong --- we're supposed to average over the minibatch.
        # However, most words are rare --- so most words will only have non-zero
        # gradient for 1 or 2 examples anyway.
        cdef ExampleC* eg
        if nn.embeds is not NULL:
            for i in range(mb.nr_eg):
                eg = &mb.egs[i]
                if eg.features is not NULL:
                    Embedding.fine_tune(nn.opt, nn.embeds, eg.fine_tune,
                        <const weight_t*>bn_bwd[0], nn.widths[0], eg.features, eg.nr_feat)

    @staticmethod
    cdef inline void insert_embeddingsC(NeuralNetC* nn, Pool mem,
            const ExampleC* egs, int nr_eg) except *:
        for i in range(nr_eg):
            eg = &egs[i]
            for j in range(eg.nr_feat):
                feat = eg.features[j]
                emb = <weight_t*>Map_get(nn.embeds.tables[feat.i], feat.key)
                if emb is NULL:
                    emb = <weight_t*>mem.alloc(nn.embeds.lengths[feat.i], sizeof(weight_t))
                    Initializer.normal(emb,
                        0.0, 1.0, nn.embeds.lengths[feat.i])
                    Map_set(mem, nn.embeds.tables[feat.i], feat.key, emb)
  
    @staticmethod
    cdef inline void forward(weight_t** state,
                        const weight_t* W,
                        const int* widths, int n) nogil:
        cdef int i
        for i in range(n-2): # Save last layer for softmax
            Rectifier.forward(state[i+1],
                state[i], W, widths[i+1], widths[i])
            W += widths[i+1] * widths[i] + widths[i+1]
        Softmax.forward(state[n-1],
            state[n-2], W, widths[n-1], widths[n-2])

    @staticmethod
    cdef inline void backward(weight_t** bwd,
                        const weight_t* costs,
                        const weight_t* const* fwd, 
                        const weight_t* W,
                        const int* widths, int n) nogil:
        Softmax.delta_log_loss(bwd[n-1],
            costs, fwd[n-1], widths[n-1])
        cdef int i
        for i in range(n-2, 0, -1):
            W -= widths[i+1] * widths[i] + widths[i+1]
            Rectifier.backward(bwd[i], # Output: error of this layer, len=width
                bwd[i+1],    # Input: error from layer above, len=nr_out
                fwd[i],      # Input: signal from layer below, len=nr_wide
                W,           # Weights of this layer
                widths[i+1], # Width of next layer 
                widths[i]    # Width of this layer 
            )
        # The delta at bwd_state[0] can be used to 'fine tune' e.g. word vectors
        W -= widths[1] * widths[0] + widths[1]
        MatVec.T_dot(bwd[0],
            W, bwd[1], widths[1], widths[0])

    @staticmethod
    cdef inline void set_gradient(weight_t* gradient,
                        const weight_t* const* fwd,
                        const weight_t* const* bwd,
                        const int* widths, int n) nogil:
        cdef int i
        for i in range(n-1):
            MatMat.add_outer_i(gradient,
                bwd[i+1], fwd[i], widths[i+1], widths[i])
            VecVec.add_i(gradient + (widths[i+1] * widths[i]),
                bwd[i+1], 1.0, widths[i+1])
            gradient += (widths[i+1] * widths[i]) + widths[i+1]


cdef class Embedding:
    cdef Pool mem
    cdef EmbeddingC* c

    @staticmethod
    cdef inline void init(EmbeddingC* self, Pool mem, vector_widths, features) except *: 
        assert max(features) < len(vector_widths)
        # Create tables, which may be shared between different features
        # e.g., we might have a feature for this word, and a feature for next
        # word. These occupy different parts of the input vector, but draw
        # from the same embedding table.
        uniqs = <MapC*>mem.alloc(len(vector_widths), sizeof(MapC))
        uniq_defaults = <weight_t**>mem.alloc(len(vector_widths), sizeof(void*))
        for i, width in enumerate(vector_widths):
            Map_init(mem, &uniqs[i], 8)
            uniq_defaults[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            Initializer.normal(uniq_defaults[i],
                0.0, 1.0, width)
        self.offsets = <int*>mem.alloc(len(features), sizeof(int))
        self.lengths = <int*>mem.alloc(len(features), sizeof(int))
        self.tables = <MapC**>mem.alloc(len(features), sizeof(void*))
        self.defaults = <weight_t**>mem.alloc(len(features), sizeof(void*))
        offset = 0
        for i, table_id in enumerate(features):
            self.tables[i] = &uniqs[table_id]
            self.lengths[i] = vector_widths[table_id]
            self.defaults[i] = uniq_defaults[table_id]
            self.offsets[i] = offset
            offset += vector_widths[table_id]

    @staticmethod
    cdef inline void set_input(weight_t* out, const FeatureC* features, int nr_feat,
            const EmbeddingC* layer) nogil:
        for i in range(nr_feat):
            feat = features[i]
            emb = <weight_t*>Map_get(layer.tables[feat.i], feat.key)
            if emb == NULL:
                emb = layer.defaults[feat.i]
            VecVec.add_i(&out[layer.offsets[feat.i]], 
                emb, feat.val, layer.lengths[feat.i])

    @staticmethod
    cdef inline void fine_tune(OptimizerC* opt, EmbeddingC* layer, weight_t* fine_tune,
                               const weight_t* delta, int nr_delta,
                               const FeatureC* features, int nr_feat) nogil:
        for i in range(nr_feat):
            # Reset fine_tune, because we need to modify the gradient
            memcpy(fine_tune, delta, sizeof(weight_t) * nr_delta)
            feat = features[i]
            weights = <weight_t*>Map_get(layer.tables[feat.i], feat.key)
            gradient = &fine_tune[layer.offsets[feat.i]]
            # TODO: Currently we can't store supporting parameters for the word
            # vectors in opt, so we can only do vanilla SGD. In practice this
            # seems to work very well!
            VanillaSGD.update(opt, weights, gradient,
                feat.val, layer.lengths[feat.i])


cdef class BN:
    @staticmethod
    cdef inline void mean(weight_t* mean,
                          const weight_t* const* acts,
                          int nr_eg, int nr_weight) nogil:
        for i in range(nr_eg):
            VecVec.add_i(mean,
                acts[i], 1.0, nr_weight)
        Vec.div_i(mean,
            nr_eg, nr_weight)

    @staticmethod
    cdef inline void variance(weight_t* var, const weight_t* const* acts,
                              const weight_t* mean, int nr_eg, int nr_weight) nogil:
        for i in range(nr_eg):
            for j in range(nr_weight):
                var[j] += (acts[i][j] - mean[j]) ** 2
        Vec.div_i(var,
            nr_eg, nr_weight)

    @staticmethod
    cdef inline void delta_variance(weight_t* d_var,
            const weight_t* mean, const weight_t* var, const weight_t*const* d_x,
            const weight_t* const* acts, int nr_eg, int nr_weight) nogil:
        cdef weight_t eps = 0.000001
        for i in range(nr_eg):
            for j in range(nr_weight):
                d_var[j] += d_x[i][j] * (acts[i][j] - mean[j])
        for i in range(nr_weight):
            d_var[i] *= 0.5 * (var[i] + eps) ** -1.5

    @staticmethod
    cdef inline void delta_mean(weight_t* d_mean, weight_t* tmp,
            const weight_t* d_var, const weight_t* const* d_x,
            const weight_t* const* x,
            const weight_t* mean, const weight_t* var, 
            int nr_eg, int nr_weight) nogil:
        cdef weight_t eps = 0.000001 
        for i in range(nr_weight):
            tmp[i] = -1 / c_sqrt(d_var[i] + eps)
        for i in range(nr_eg):
            for j in range(nr_weight):
                d_mean[j] += d_x[i][j] * tmp[j]
        Initializer.constant(tmp,
            0, nr_weight)
        for i in range(nr_eg):
            for j in range(nr_weight):
                tmp[j] += -2 * (x[i][j] - mean[j])
        Vec.div_i(tmp,
            nr_eg, nr_weight)
        for i in range(nr_weight):
            d_mean[i] += d_var[i] * tmp[i]

    @staticmethod
    cdef inline void forward(weight_t* out,
                        const weight_t* in_,
                        const weight_t* W,
                        const weight_t* avg, const weight_t* var,
                        int nr_wide) nogil:
        cdef weight_t eps = 0.000001 
        for i in range(nr_wide):
            out[i] = (in_[i] - avg[i]) / c_sqrt(var[i] + eps)
        # Scale
        VecVec.mul_i(out,
            W, nr_wide)
        # Shift
        VecVec.add_i(out,
            &W[nr_wide], 1.0, nr_wide)

    @staticmethod
    cdef inline void backward(weight_t* delta_x,   # Len == nr_wide
                              weight_t* delta_W,
                        const weight_t* delta_y,    # Len == nr_out
                        const weight_t* X,   # Len == nr_wide
                        const weight_t* W,
                        const weight_t* mean,
                        const weight_t* variance,
                        const weight_t* d_means,
                        const weight_t* d_vars,
                        int32_t nr_eg,
                        int32_t nr_out,
                        int32_t nr_wide) nogil:
        # In fwd pass, we received x and output Y.
        # We computed Y  = norm(x) * g + b
        # Call norm(x) x'. We computed x' = x - avg(X) / sqrt(var(X) + eps)
        #
        # Where avg(X) and var(X) were minibatch statistics.
        #
        # Here we receive:
        #
        # delta_out: This needs to be delta(x), just as though we did no batch norm
        # delta_in: This is the delta(y), i.e. the error from the layer above
        # X: This is the *input* to our layer, from which we computed Y.
        # means: These are the averages of the other Xs in our minibatch
        # variances: These are the variances of the other Xs in our minibatch
        # d_means: The delta of the means in the batch 
        # d_vars: The delta of the variances in the batch
        cdef weight_t eps = 0.000001 
        for i in range(nr_wide):
            delta_x[i] = delta_y[i] * W[i]
            delta_x[i] *= 1 / c_sqrt(variance[i] + eps)
            delta_x[i] += d_vars[i] * (2 * (X[i] - mean[i]) / nr_eg)
            delta_x[i] += d_means[i] * (1 / nr_eg)

 
cdef class Rectifier:
    @staticmethod
    cdef inline void forward(weight_t* out,
                        const weight_t* in_, const weight_t* W,
                        int nr_out, int nr_wide) nogil:
        # We're a layer of nr_wide cells, which we can think of as features.
        # We write to an array of nr_out activations, one for each conenction
        # to the next layer. We can think of the cells in the next layer like classes:
        # we want to know whether we make that cell activate.
        # 
        # It's tempting to think at first as though we output nr_wide activations.
        # We *receive* nr_wide activations. What we're determining now is,
        # given those activations, our weights and our biases, what's the
        # state of the next layer?
        MatVec.dot(out,
            W, in_, nr_out, nr_wide)
        # Bias
        VecVec.add_i(out,
            W + (nr_out * nr_wide), 1.0, nr_out)
        cdef int32_t i
        for i in range(nr_out):
            # Writing this way handles NaN
            if not (out[i] > 0):
                out[i] = 0

    @staticmethod
    cdef inline void backward(weight_t* delta_out,       # Len == nr_wide
                        const weight_t* delta_in,  # Len == nr_out
                        const weight_t* signal_in, # Len == nr_wide
                        const weight_t* W,
                        int32_t nr_out,
                        int32_t nr_wide) nogil:
        # delta = W.T.dot(prev_delta) * d_relu(signal_in)
        # d_relu(signal_in) is a binary vector, 0 when signal_in < 0
        # So, we do our dot product, and then clip to 0 on the dimensions where
        # signal_in is 0
        # Note that prev_delta is a column vector (the error of our output),
        # while delta is a row vector (the error of our neurons, which must match
        # the input layer's width)
        MatVec.T_dot(delta_out,
            W, delta_in, nr_out, nr_wide)
        cdef int32_t i
        for i in range(nr_wide):
            if signal_in[i] < 0:
                delta_out[i] = 0


cdef class Softmax:
    @staticmethod
    cdef inline void forward(weight_t* out,
                             const weight_t* in_,
                             const weight_t* W,
                             int32_t nr_out,
                             int32_t nr_wide) nogil:
        #w = W.dot(actvn) + b
        MatVec.dot(out,
            W, in_, nr_out, nr_wide)
        # Bias
        VecVec.add_i(out,
            W + (nr_out * nr_wide), 1.0, nr_out)
        #w = numpy.exp(w - max(w))
        Vec.add_i(out,
            -Vec.max(out, nr_out), nr_out)
        Vec.exp_i(out,
            nr_out)
        #w = w / sum(w)
        Vec.div_i(out,
            Vec.sum(out, nr_out), nr_out)

    @staticmethod
    cdef inline void delta_log_loss(weight_t* loss,
                        const weight_t* costs,
                        const weight_t* scores,
                        int32_t nr_out) nogil:
        # This assumes only one true class
        cdef int i
        for i in range(nr_out):
            loss[i] = scores[i] - (costs[i] == 0)


cdef class Initializer:
    @staticmethod
    cdef inline void normal(weight_t* weights, weight_t loc, weight_t scale, int n) except *:
        # See equation 10 here:
        # http://arxiv.org/pdf/1502.01852v1.pdf
        values = numpy.random.normal(loc=0.0, scale=scale, size=n)
        for i, value in enumerate(values):
            weights[i] = value

    @staticmethod
    cdef inline void constant(weight_t* weights, weight_t value, int n) nogil:
        for i in range(n):
            weights[i] = value


cdef class VanillaSGD:
    @staticmethod
    cdef inline void init(OptimizerC* self, Pool mem, int nr_weight, int* widths,
            int nr_layer, weight_t eta, weight_t eps, weight_t rho) except *:
        self.update = VanillaSGD.update
        self.eta = eta
        self.eps = eps
        self.rho = rho
        self.params = NULL
        self.ext = NULL
        self.nr = 0

    @staticmethod
    cdef inline void update(OptimizerC* opt, weight_t* weights, weight_t* gradient,
            weight_t scale, int nr_weight) nogil:
        '''
        Update weights with vanilla SGD
        '''
        Vec.mul_i(gradient, scale, nr_weight)
        # Add the derivative of the L2-loss to the gradient
        VecVec.add_i(gradient,
            weights, opt.rho, nr_weight)

        VecVec.add_i(weights,
            gradient, -opt.eta, nr_weight)


cdef class Adagrad:
    @staticmethod
    cdef inline void init(OptimizerC* self, Pool mem, int nr_weight, int* widths,
            int nr_layer, weight_t eta, weight_t eps, weight_t rho) except *:
        self.update = Adagrad.update
        self.eta = eta
        self.eps = eps
        self.rho = rho
        self.params = <weight_t*>mem.alloc(nr_weight, sizeof(weight_t))
        self.ext = NULL
        self.nr = 0

    @staticmethod
    cdef inline void update(OptimizerC* opt, weight_t* weights, weight_t* gradient,
            weight_t scale, int nr_weight) nogil:
        '''
        Update weights with vanilla SGD
        '''
        # Add the derivative of the L2-loss to the gradient
        cdef int i
        VecVec.add_i(gradient,
            weights, opt.rho, nr_weight)
        VecVec.add_pow_i(opt.params,
            gradient, 2.0, nr_weight)
        for i in range(nr_weight):
            gradient[i] *= opt.eta / (c_sqrt(opt.params[i]) + opt.eps)
        Vec.mul_i(gradient,
            scale, nr_weight)
        # Make the (already scaled) update
        VecVec.add_i(weights,
            gradient, -1.0, nr_weight)
