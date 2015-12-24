cimport cython
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt
from libc.stdint cimport int32_t

from cymem.cymem cimport Pool

from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_get as Map_get

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
    cdef inline void predictC(ExampleC* eg, const NeuralNetC* nn) nogil:
        Embedding.set_input(eg.fwd_state[0],
            eg.features, eg.nr_feat, nn.embeds)
        NeuralNet.forward(eg.fwd_state,
            nn.weights, nn.widths, nn.nr_layer)
        Example.set_scores(eg,
            eg.fwd_state[nn.nr_layer-1])
    
    @staticmethod
    cdef inline void trainC(NeuralNetC* nn, BatchC* mb) nogil:
        cdef int i
        # Compute forward and backward passes
        for i in range(mb.nr_eg):
            eg = &mb.egs[i]
            if nn.embeds is not NULL and eg.features is not NULL:
                Embedding.set_input(eg.fwd_state[0],
                    eg.features, eg.nr_feat, nn.embeds)
            NeuralNet.forward(eg.fwd_state,
                nn.weights, nn.widths, nn.nr_layer)
            NeuralNet.backward(eg.bwd_state,
                eg.costs, eg.fwd_state, nn.weights + nn.nr_weight, nn.widths, nn.nr_layer)
        # Get the averaged gradient for the minibatch
        for i in range(mb.nr_eg):
            NeuralNet.set_gradient(mb.gradient,
                mb.egs[i].fwd_state, mb.egs[i].bwd_state, nn.widths, nn.nr_layer)
        Vec.div_i(mb.gradient, mb.nr_eg, nn.nr_weight)

        nn.opt.update(nn.opt, nn.weights, mb.gradient,
            1.0, nn.nr_weight)

        # Fine-tune the embeddings
        # This is sort of wrong --- we're supposed to average over the minibatch.
        # But doing that is annoying.
        if nn.embeds is not NULL:
            for i in range(mb.nr_eg):
                eg = &mb.egs[i]
                if eg.features is not NULL:
                    Embedding.fine_tune(nn.opt, nn.embeds, eg.fine_tune,
                        eg.bwd_state[0], nn.widths[0], eg.features, eg.nr_feat)
    
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
        MatVec.T_dot(bwd[0], W, bwd[1], widths[1], widths[0])

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
            uniq_defaults[i] = <weight_t*>mem.alloc(width, sizeof(weight_t))
            Map_init(mem, &uniqs[i],
                8)
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
            opt.update(opt, weights, gradient,
                feat.val, layer.lengths[feat.i])


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


cdef class VanillaSGD:
    @staticmethod
    cdef inline void init(OptimizerC* self, Pool mem, int nr_weight, int* widths,
                    int nr_layer, weight_t eta, weight_t eps, weight_t rho) nogil:
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
