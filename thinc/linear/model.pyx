cimport cython
from os import path
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.exc cimport PyErr_CheckSignals
from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdlib cimport qsort
from libc.stdint cimport int32_t

from preshed.maps cimport PreshMap, MapStruct, map_get
from .sparse cimport SparseArray

from .eg cimport Example
from .structs cimport SparseArrayC
from .typedefs cimport class_t, count_t
from .serialize cimport Writer
from .serialize cimport Reader


cdef class Model:
    def __init__(self):
        raise NotImplementedError

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

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.weights is not None:
            for feat_addr in self.weights.values():
                if feat_addr != 0:
                    PyMem_Free(<SparseArrayC*>feat_addr)

    def __call__(self, Example eg):
        self.set_scores(eg.c.scores, eg.c.features, eg.c.nr_feat)
        #eg.c.guess = arg_max_if_true(eg.c.scores, eg.c.is_valid, eg.c.nr_class)
        PyErr_CheckSignals()

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
                    scores[class_weights[j].key] += class_weights[j].val * feat.value
                    j += 1
    
    @cython.cdivision(True)
    def dump(self, nr_class, loc):
        cdef:
            feat_t key
            size_t i
            size_t feat_addr

        cdef Writer writer = Writer(loc, nr_class)
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
            if i % 1000 == 0:
                PyErr_CheckSignals()
        writer.close()

    @cython.cdivision(True)
    def load(self, loc):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef Reader reader = Reader(loc)
        cdef int i = 0
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)
            if i % 1000 == 0:
                PyErr_CheckSignals()
            i += 1
        return reader._nr_class
