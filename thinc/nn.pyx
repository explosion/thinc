cimport cython
from libc.stdint cimport int32_t
from libc.string cimport memset, memcpy
from libc.math cimport sqrt as c_sqrt

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from .api cimport arg_max_if_true, arg_max_if_zero
from .layer cimport Embedding, Rectifier, Softmax
from .structs cimport ExampleC, FeatureC, LayerC, HyperParamsC
from .typedefs cimport weight_t, atom_t
from .api cimport Example, Learner
from .blas cimport VecVec


cdef class NeuralNetwork(Learner):
    def __init__(self, nr_class, nr_embed, hidden_layers):
        self.nr_layer = len(hidden_layers) + 1
        self.nr_embed = nr_embed
        self.nr_dense = 0
        self.layers = <LayerC*>self.mem.alloc(self.nr_layer, sizeof(LayerC))
        nr_wide = self.nr_embed
        for i, (nr_out, activation) in hidden_layers:
            self.layers[i] = Rectifier.init(nr_wide, nr_out, self.nr_dense)
            nr_wide = nr_out
            self.nr_dense += nr_wide * nr_out + nr_out
        self.layers[i+1] = Softmax.init(nr_wide, self.nr_class, self.nr_dense)
        self.nr_dense += nr_wide * self.nr_class + self.nr_class

    def __call__(self, Example eg):
        self.set_prediction(&eg.c)
    
    cdef ExampleC allocate(self, Pool mem) except *:
        cdef ExampleC eg
        eg.is_valid = <int*>mem.alloc(self.nr_class, sizeof(eg.is_valid[0]))
        eg.costs = <int*>mem.alloc(self.nr_class, sizeof(eg.costs[0]))
        eg.atoms = <atom_t*>mem.alloc(self.nr_class, sizeof(eg.atoms[0]))
        eg.scores = <weight_t*>mem.alloc(self.nr_class, sizeof(eg.scores[0]))
        
        eg.gradient = <weight_t*>mem.alloc(self.nr_dense, sizeof(eg.gradient[0]))

        eg.fwd_state = <weight_t**>mem.alloc(self.nr_layer+1, sizeof(eg.fwd_state[0]))
        eg.bwd_state = <weight_t**>mem.alloc(self.nr_layer+1, sizeof(eg.bwd_state[0]))
        eg.fwd_state[self.nr_layer] = <weight_t*>mem.alloc(self.nr_class, sizeof(weight_t))
        eg.bwd_state[self.nr_layer] = <weight_t*>mem.alloc(self.nr_class, sizeof(weight_t))
        cdef int i
        for i in range(self.nr_layer):
            # Fwd state[i] is the incoming signal, so equal to layer size
            eg.fwd_state[i] = <weight_t*>mem.alloc(self.layers[i].nr_wide, sizeof(weight_t))
            # Bwd state[i] is the incoming error, so equal to output size
            eg.bwd_state[i] = <weight_t*>mem.alloc(self.layers[i].nr_out, sizeof(weight_t))
        
        eg.nr_class = self.nr_class
        eg.nr_atom = self.nr_atom
        eg.nr_feat = self.nr_feat
        eg.guess = 0
        eg.best = 0
        eg.cost = 0
        return eg
    
    # from Learner cdef void set_costs(self, ExampleC* eg, int gold) except *:
    # cdef void set_features(self, ExampleC* eg, something) except *:

    cdef void set_prediction(self, ExampleC* eg) except *:
        Embedding.set_layer(eg.fwd_state[0], self.weights.c_map,
                            eg.features, eg.nr_feat)
        weights = <const weight_t*>self.weights.get(1)

        NeuralNetwork.forward(
            eg.fwd_state,
            <const weight_t*>self.weights.get(1),
            self.layers, 
            self.nr_layer
        )

        memcpy(eg.scores, eg.fwd_state[self.nr_layer],
               sizeof(eg.scores[0]) * eg.nr_class)
        eg.guess = arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        eg.best = arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)
    

    cdef void update(self, ExampleC* eg) except *:
        NeuralNetwork.set_loss(
            eg.bwd_state[self.nr_layer],
            eg.scores,
            eg.best,
            self.nr_class
        )

        NeuralNetwork.backward(
            eg.bwd_state,
            <const weight_t**>eg.fwd_state,
            <const weight_t*>self.weights.get(1),
            self.layers,
            self.nr_layer
        )

        NeuralNetwork.set_gradients(
            eg.gradient,
            <const weight_t**>eg.bwd_state,
            <const weight_t**>eg.fwd_state,
            self.layers,
            self.nr_layer
        )
        
        # L2 regularization
        VecVec.add_i(eg.gradient, <const weight_t*>self.weights.get(1),
                     self.hyper_params.rho, self.nr_dense)

        # Dense update
        adagrad(
            <weight_t*>self.weights.get(1),
            eg.gradient,
            self.train_weights.get(1),
            self.nr_dense,
            <void*>&self.hyper_params
        )

        # TODO: For simplicity for now, only support binary values.
        # To support values, need to calculate total for each dimension, and then
        # distribute to each feature by how much it contributed to that dimension
        #
        # What do we do about regularization for these updates? Nothing?
        for i in range(eg.nr_feat):
            feat = eg.features[i]
            adagrad(
                <weight_t*>self.weights.get(feat.key),
                &eg.gradient[feat.i],
                <void*>self.train_weights.get(feat.key),
                feat.length,
                <void*>&self.hyper_params)

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
