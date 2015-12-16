from __future__ import print_function
cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap
import numpy

from .api cimport arg_max_if_true, arg_max_if_zero
from .layer cimport Embedding, Rectifier, Softmax
from .structs cimport ExampleC, FeatureC, LayerC, HyperParamsC
from .typedefs cimport weight_t, atom_t, feat_t
from .api cimport Example, Learner
from .blas cimport VecVec


cdef class NeuralNet(Learner):
    def __init__(self, nr_class, nr_embed, hidden_layers,
                 weight_t eta=0.005, weight_t epsilon=1e-3, weight_t rho=1e-4):
        self.c.hyper_params.eta = eta
        self.c.hyper_params.epsilon = epsilon
        self.c.hyper_params.rho = rho
        self.c.nr_class = nr_class
        self.c.nr_in = nr_embed
        self.c.nr_layer = len(hidden_layers) + 1
        self.mem = Pool()
        self.c.nr_dense = 0
        self.c.layers = <LayerC*>self.mem.alloc(self.c.nr_layer, sizeof(LayerC))
        nr_wide = nr_embed
        for i, nr_out in enumerate(hidden_layers):
            self.c.layers[i] = Rectifier.init(nr_out, nr_wide, self.c.nr_dense)
            nr_wide = nr_out
            self.c.nr_dense += nr_wide * nr_out + nr_out
        self.c.layers[self.c.nr_layer-1] = Softmax.init(self.c.nr_class, nr_wide,
                                                        self.c.nr_dense)
        self.c.nr_dense += nr_wide * self.c.nr_class + self.c.nr_class
        self.c.weights = <weight_t*>self.mem.alloc(self.c.nr_dense, 
                                                   sizeof(self.c.weights[0]))
        self.c.support = <weight_t*>self.mem.alloc(self.c.nr_dense, 
                                                   sizeof(self.c.weights[0]))
        self.weights = PreshMap()
        self.train_weights = PreshMap()
        numpy.random.seed(0)
        # He initialization
        # Note that we don't initialize 'support'! This is left 0
        # http://arxiv.org/abs/1502.01852
        for i in range(self.c.nr_layer-1):
            n_weights = self.c.layers[i].nr_wide * self.c.layers[i].nr_out
            std = numpy.sqrt(2.0) * numpy.sqrt(1.0 / self.c.layers[i].nr_wide)
            init_weights = numpy.random.normal(loc=0.0, scale=std, size=n_weights)
            offset = self.c.layers[i].W
            for j in range(n_weights):
                self.c.weights[offset + j] = init_weights[j]

    property layers:
        def __get__(self):
            return [(self.c.layers[i].nr_out, self.c.layers[i].nr_wide) for i
                    in range(self.c.nr_layer)]
    property nr_class:
        def __get__(self): return self.c.nr_class
    property nr_embed:
        def __get__(self): return self.c.nr_in
    property nr_layer:
        def __get__(self): return self.c.nr_layer
    property nr_dense:
        def __get__(self): return self.c.nr_dense

    def __call__(self, Example eg):
        self.set_prediction(&eg.c)
        return eg.c.guess

    def train(self, Example eg):
        self.set_prediction(&eg.c)
        self.update(&eg.c)
        return eg.loss

    def Example(self, features, gold=None):
        cdef Example eg = Example()
        Example.init_classes(&eg.c, eg.mem, self.c.nr_class) 
        Example.init_nn_state(&eg.c, eg.mem, self.c.layers,
                              self.c.nr_layer, self.c.nr_dense)
        Example.init_features(&eg.c, eg.mem, self.c.nr_in, len(features))

        cdef feat_t key
        cdef weight_t value
        cdef int offset
        cdef int length
        cdef int i
        for i, (key, value, offset, length) in enumerate(features):
            eg.c.features[i].key = key
            eg.c.features[i].val = value
            eg.c.features[i].i = offset
            eg.c.features[i].length = length
            embed = <weight_t*>self.weights.get(key)
            if embed is NULL:
                embed = <weight_t*>self.mem.alloc(length, sizeof(weight_t))
                std = numpy.sqrt(2.0) * numpy.sqrt(1.0 / length)
                init_weights = numpy.random.normal(loc=0.0, scale=std, size=length)
                for j, weight in enumerate(init_weights):
                    embed[j] = weight
                self.weights.set(key, <void*>embed)
                self.train_weights.set(key, self.mem.alloc(length, sizeof(weight_t)))
 
        cdef int clas
        if gold is not None:
            self.set_costs(&eg.c, gold)
            for clas in range(self.c.nr_class):
                eg.c.costs[clas] = 1
            eg.c.costs[gold] = 0
            eg.c.best = gold
        return eg

    # from Learner cdef void set_costs(self, ExampleC* eg, int gold) except *:
    # cdef void set_features(self, ExampleC* eg, something) except *:

    cdef void set_prediction(self, ExampleC* eg) except *:
        cdef int32_t i, i_lyr
        Embedding.set_layer(eg.fwd_state[0], self.weights.c_map,
                            eg.features, eg.nr_feat)

        NeuralNet.forward(
            eg.fwd_state,
            self.c.weights,
            self.c.layers, 
            self.c.nr_layer
        )
        memcpy(eg.scores, eg.fwd_state[self.c.nr_layer],
               sizeof(eg.scores[0]) * eg.nr_class)

        eg.guess = arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        eg.best = arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)

    cdef void update(self, ExampleC* eg) except *:
        # Copy cost to bwd_state
        memcpy(eg.bwd_state[self.c.nr_layer], eg.costs,
               self.c.nr_class * sizeof(eg.costs[0]))

        NeuralNet.backward(
            eg.bwd_state,
            <const weight_t**>eg.fwd_state,
            <const weight_t*>self.c.weights,
            self.c.layers,
            self.c.nr_layer
        )

        NeuralNet.set_gradients(
            eg.gradient,
            <const weight_t**>eg.bwd_state,
            <const weight_t**>eg.fwd_state,
            self.c.layers,
            self.c.nr_layer
        )
        
        # L2 regularization
        VecVec.add_i(eg.gradient, self.c.weights, self.c.hyper_params.rho,
                     self.c.nr_dense)

        # Dense update
        adagrad(
            self.c.weights,
            eg.gradient,
            self.c.support,
            self.c.nr_dense,
            <void*>&self.c.hyper_params
        )

        for i in range(eg.nr_feat):
            feat = eg.features[i]
            embed = <weight_t*>self.weights.get(feat.key)
            support = <weight_t*>self.train_weights.get(feat.key)
            if embed is not NULL and support is not NULL:
                adagrad(
                    embed,
                    eg.gradient, # TODO fix this hack
                    support,
                    feat.length,
                    <void*>&self.c.hyper_params
                )


    # from Learner def end_training(self):
    # from Learner def dump(self, loc):
    # from Learner def load(self, loc):


@cython.cdivision(True)
cdef void adagrad(weight_t* weights, weight_t* gradient, void* _support, int32_t n,
                  const void* _hyper_params) nogil:
    '''
    Update weights with Adagrad
    '''
    support = <weight_t*>_support
    hp = <const HyperParamsC*>_hyper_params

    VecVec.add_pow_i(support, gradient, 2.0, n)

    cdef int i
    for i in range(n):
        gradient[i] *= hp.eta / (c_sqrt(support[i]) + hp.epsilon)
    VecVec.add_i(weights, gradient, 1.0, n)
