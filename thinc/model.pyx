from os import path
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdlib cimport qsort
from libc.stdint cimport int32_t

from preshed.maps cimport PreshMap, MapStruct, map_get
from .sparse cimport SparseArray

from .api cimport Example, arg_max, arg_max_if_zero, arg_max_if_true
from .structs cimport SparseArrayC
from .typedefs cimport class_t, count_t


cdef class Model:
    def __init__(self):
        raise NotImplementedError

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.weights is not None:
            for feat_addr in self.weights.values():
                if feat_addr != 0:
                    PyMem_Free(<void*>feat_addr)

    def __call__(self, Example eg):
        self.set_scores(eg.c.scores, eg.c.features, eg.c.nr_feat)
        eg.c.guess = arg_max_if_true(eg.c.scores, eg.c.is_valid, eg.c.nr_class)

    cdef void set_scores(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil:
        pass


cdef class LinearModel(Model):
    '''A linear model for online supervised classification.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.
    '''
    def __init__(self):
        self.weights = PreshMap()
        self.mem = Pool()
    
    cdef void set_scores(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil:
        # This is the main bottle-neck of spaCy --- where we spend all our time.
        # Typical sizes for the dependency parser model:
        # * weights_table: ~9 million entries
        # * n_feats: ~200
        # * scores: ~80 classes
        # 
        # I think the bottle-neck is actually reading the weights from main memory.

        cdef const MapStruct* weights_table = self.weights.c_map
 
        cdef int i, j
        cdef FeatureC feat
        for i in range(nr_feat):
            feat = feats[i]
            class_weights = <const SparseArrayC*>map_get(weights_table, feat.key)
            if class_weights != NULL:
                j = 0
                while class_weights[j].key >= 0:
                    scores[class_weights[j].key] += class_weights[j].val * feat.val
                    j += 1
    
    def dump(self, nr_class, loc):
        cdef:
            feat_t key
            size_t i
            size_t feat_addr

        cdef _Writer writer = _Writer(loc, nr_class)
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
        writer.close()

    def load(self, loc):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef _Reader reader = _Reader(loc)
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)
        return reader._nr_class


cdef class NeuralNet(Model):
    cdef readonly Pool mem
    cdef NeuralNetC c

    def __init__(self, widths, embed=None, float eta=0.005, float eps=1e-6,
                 float mu=0.2, float rho=1e-4, float bias=0.0, float alpha=0.0):
        self.mem = Pool()
        NN.init(&self.c, self.mem, widths, eta, eps, mu, rho, bias, alpha)

    def __call__(self, features):
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        NN.predict_example(&eg.c,
            &self.c)
        return eg
   
    def train(self, features, y):
        memset(self.c.gradient,
            0, sizeof(self.c.gradient[0]) * self.c.nr_weight)
        cdef Example eg = self.eg
        eg.wipe(self.widths)
        eg.set_features(features)
        eg.set_label(y)

        NN.predict_example(&eg.c, &self.c)
        insert_sparse(self.c.sparse_weights, self.mem,
            self.c.embed_lengths, self.c.embed_offsets, self.c.embed_defaults,
            eg.c.features, eg.nr_feat)
        insert_sparse(self.c.sparse_momentum, self.mem,
            self.c.embed_lengths, self.c.embed_offsets, self.c.embed_defaults,
            eg.c.features, eg.c.nr_feat)
        NN.update(&self.c, &eg.c)
        return eg
 
    def Example(self, input_, label=None):
        if isinstance(input_, Example):
            return input_
        return Example(self.widths, input_, label)

    property weights:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.c.nr_weight
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT, self.c.weights)
            
        def __set__(self, weights):
            assert len(weights) == self.c.nr_weight
            for i, weight in enumerate(weights):
                self.c.weights[i] = weight

    property layers:
        def __get__(self):
            weights = self.weights
            cdef IteratorC it
            it.i = 0
            while NN.iter(&it, self.c.widths, self.c.nr_layer-1, 1):
                yield (weights[it.W:it.bias], weights[it.bias:it.gamma])

    property widths:
        def __get__(self):
            return tuple(self.c.widths[i] for i in range(self.c.nr_layer))

    property layer_l1s:
        def __get__(self):
            for W, bias in self.layers:
                w_l1 = sum(abs(w) for w in W) / len(W)
                bias_l1 = sum(abs(w) for w in W) / len(bias)
                yield w_l1, bias_l1

    property gradient:
        def __get__(self):
            return [self.c.gradient[i] for i in range(self.c.nr_weight)]

    property l1_gradient:
        def __get__(self):
            cdef int i
            cdef float total = 0.0
            for i in range(self.c.nr_weight):
                if self.c.gradient[i] < 0:
                    total -= self.c.gradient[i]
                else:
                    total += self.c.gradient[i]
            return total / self.c.nr_weight

    property embeddings:
        def __get__(self):
            cdef int i = 0
            cdef int j = 0
            cdef int k = 0
            cdef key_t key
            cdef void* value
            for i in range(self.c.nr_embed):
                j = 0
                while Map_iter(self.c.sparse_weights[i], &j, &key, &value):
                    emb = <float*>value
                    yield key, [emb[k] for k in range(self.c.embed_lengths[i])]

    property nr_layer:
        def __get__(self):
            return self.c.nr_layer
    property nr_weight:
        def __get__(self):
            return self.c.nr_weight
    property nr_out:
        def __get__(self):
            return self.c.widths[self.c.nr_layer-1]
    property nr_in:
        def __get__(self):
            return self.c.widths[0]

