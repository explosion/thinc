from __future__ import print_function
cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt
import random

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .api cimport arg_max_if_true, arg_max_if_zero
from .layer cimport Embedding, Rectifier, Softmax
from .structs cimport ExampleC, FeatureC, LayerC, HyperParamsC
from .typedefs cimport weight_t, atom_t
from .api cimport Example, Learner
from .blas cimport VecVec


cdef class NeuralNet(Learner):
    def __init__(self, nr_class, nr_embed, hidden_layers):
        self.c.hyper_params.eta = 0.005
        self.c.hyper_params.epsilon = 1e-6
        self.c.hyper_params.rho = 1e-4
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
        support = <weight_t*>self.mem.alloc(self.c.nr_dense, 
                                                   sizeof(self.c.weights[0]))
        self.weights = PreshMap()
        self.train_weights = PreshMap()
        random.seed(0)
        for i in range(self.c.nr_dense):
            self.c.weights[i] = random.random()
            support[i] = random.random()
        self.c.support = support

    property layer_dims:
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

    def Example(self, feats, gold=None):
        cdef Example eg = Example(self.c.nr_class, len(feats), len(feats), len(feats))
        eg.c = self.allocate(eg.mem)
        eg.c.features = <FeatureC*>eg.mem.alloc(len(feats), sizeof(FeatureC))
        eg.c.nr_feat = len(feats)
        for i, (key, value) in enumerate(feats):
            eg.c.features[i].i = i
            eg.c.features[i].key = key
            eg.c.features[i].val = value
            eg.c.features[i].length = 1
        cdef int clas
        for clas in range(eg.c.nr_class):
            eg.c.is_valid[clas] = 1
        if gold is not None:
            for clas in range(self.c.nr_class):
                eg.c.costs[clas] = 1
            eg.c.costs[gold] = 0
            eg.c.best = gold
        return eg

    cdef ExampleC allocate(self, Pool mem) except *:
        cdef ExampleC eg
        eg.is_valid = <int*>mem.alloc(self.c.nr_class, sizeof(eg.is_valid[0]))
        eg.costs = <weight_t*>mem.alloc(self.c.nr_class, sizeof(eg.costs[0]))
        eg.atoms = <atom_t*>mem.alloc(self.c.nr_class, sizeof(eg.atoms[0]))
        eg.scores = <weight_t*>mem.alloc(self.c.nr_class, sizeof(eg.scores[0]))
        
        eg.gradient = <weight_t*>mem.alloc(self.c.nr_dense, sizeof(eg.gradient[0]))

        eg.fwd_state = <weight_t**>mem.alloc(self.c.nr_layer+1, sizeof(eg.fwd_state[0]))
        eg.bwd_state = <weight_t**>mem.alloc(self.c.nr_layer+1, sizeof(eg.bwd_state[0]))
        cdef int i
        for i in range(self.c.nr_layer):
            # Fwd state[i] is the incoming signal, so equal to layer size
            eg.fwd_state[i] = <weight_t*>mem.alloc(self.c.layers[i].nr_wide, sizeof(weight_t))
            # Bwd state[i] is the incoming error, so equal to output size
            eg.bwd_state[i] = <weight_t*>mem.alloc(self.c.layers[i].nr_out, sizeof(weight_t))
        eg.fwd_state[self.c.nr_layer] = <weight_t*>mem.alloc(self.c.nr_class, sizeof(weight_t))
        eg.bwd_state[self.c.nr_layer] = <weight_t*>mem.alloc(self.c.nr_class, sizeof(weight_t))

        eg.nr_class = self.c.nr_class
        eg.nr_atom = -1
        eg.nr_feat = -1
        eg.guess = 0
        eg.best = 0
        eg.cost = 0
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

        ## Dense update
        adagrad(
            self.c.weights,
            eg.gradient,
            self.c.support,
            self.c.nr_dense,
            <void*>&self.c.hyper_params
        )

        ## TODO: For simplicity for now, only support binary values.
        ## To support values, need to calculate total for each dimension, and then
        ## distribute to each feature by how much it contributed to that dimension
        ##
        ## What do we do about regularization for these updates? Nothing?
        #for i in range(eg.nr_feat):
        #    feat = eg.features[i]
        #    adagrad(
        #        <weight_t*>self.weights.get(feat.key),
        #        &eg.gradient[feat.i],
        #        <void*>self.train_weights.get(feat.key),
        #        feat.length,
        #        <void*>&self.c.hyper_params
        #    )

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
