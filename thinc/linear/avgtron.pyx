# cython: infer_types=True
# cython: cdivision=True
cimport cython
from os import path
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.exc cimport PyErr_CheckSignals
from libc.stdio cimport FILE, fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy

from libc.stdlib cimport qsort
from libc.stdint cimport int32_t

from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap, MapStruct, map_get

from .sparse cimport SparseArray

from ..extra.eg cimport Example
from ..structs cimport SparseArrayC, SparseAverageC
from ..typedefs cimport class_t, count_t, time_t, feat_t
from ..linalg cimport Vec, VecVec
from .serialize cimport Writer
from .serialize cimport Reader


cdef class AveragedPerceptron:
    '''A linear model for online supervised classification.
    Expected use is via Cython --- the Python API is impoverished and inefficient.

    Emphasis is on efficiency for multi-class classification, where the number
    of classes is in the dozens or low hundreds.
    '''
    def __init__(self, templates, *args, **kwargs):
        self.weights = PreshMap()
        self.time = 0
        self.averages = PreshMap()
        self.mem = Pool()
        self.extracter = ConjunctionExtracter(templates)

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.weights is not None:
            for feat_addr in self.weights.values():
                if feat_addr != 0:
                    PyMem_Free(<SparseArrayC*>feat_addr)
        if self.averages is not None:
            for feat_addr in self.averages.values():
                if feat_addr != 0:
                    feat = <SparseAverageC*>feat_addr
                    PyMem_Free(feat.avgs)
                    PyMem_Free(feat.times)

    def __call__(self, Example eg):
        self.set_scoresC(eg.c.scores,
            eg.c.features, eg.c.nr_feat)
        PyErr_CheckSignals()
        return eg.guess

    def train_example(self, Example eg):
        self(eg)
        self.update(eg)
        return eg

    def predict_example(self, Example eg):
        self(eg)
        return eg

    def update(self, Example eg):
        self.updateC(&eg.c)

    def dump(self, loc):
        cdef Writer writer = Writer(loc, self.weights.length)
        cdef feat_t key
        cdef size_t feat_addr
        for i, (key, feat_addr) in enumerate(self.weights.items()):
            if feat_addr != 0:
                writer.write(key, <SparseArrayC*>feat_addr)
            if i % 1000 == 0:
                PyErr_CheckSignals()
        writer.close()

    def load(self, loc):
        cdef feat_t feat_id
        cdef SparseArrayC* feature
        cdef Reader reader = Reader(loc)
        self.weights = PreshMap(reader.nr_feat)
        cdef int i = 0
        while reader.read(self.mem, &feat_id, &feature):
            self.weights.set(feat_id, feature)
            if i % 1000 == 0:
                PyErr_CheckSignals()
            i += 1

    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        cdef int i = 0
        for feat_id, feat_addr in self.averages.items():
            if feat_addr != 0:
                feat = <SparseAverageC*>feat_addr
                i = 0
                while feat.curr[i].key >= 0:
                    unchanged = (self.time + 1) - <time_t>feat.times[i].val
                    feat.avgs[i].val += unchanged * feat.curr[i].val
                    feat.curr[i].val = feat.avgs[i].val / self.time
                    i += 1

    property nr_feat:
        def __get__(self):
            return self.extracter.nr_templ

    cdef void set_scoresC(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil:
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
    cdef int updateC(self, const ExampleC* eg) except -1:
        self.time += 1
        guess = VecVec.arg_max_if_true(eg.scores, eg.is_valid, eg.nr_class)
        if eg.costs[guess] > 0:
            best = VecVec.arg_max_if_zero(eg.scores, eg.costs, eg.nr_class)
            for feat in eg.features[:eg.nr_feat]:
                self.update_weight(feat.key, best,   feat.value * eg.costs[guess])
                self.update_weight(feat.key, guess, -feat.value * eg.costs[guess])

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        if upd == 0:
            return 0
        feat = <SparseAverageC*>self.averages.get(feat_id)
        if feat == NULL:
            feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
            if feat is NULL:
                msg = (feat_id, clas, upd)
                raise MemoryError("Error allocating memory for feature: %s" % msg)
            feat.curr  = SparseArray.init(clas, upd)
            feat.avgs  = SparseArray.init(clas, 0)
            feat.times = SparseArray.init(clas, <weight_t>self.time)
            self.averages.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
        else:  
            i = SparseArray.find_key(feat.curr, clas)
            if i < 0:
                feat.curr = SparseArray.resize(feat.curr)
                feat.avgs = SparseArray.resize(feat.avgs)
                feat.times = SparseArray.resize(feat.times)
                self.weights.set(feat_id, feat.curr)
                i = SparseArray.find_key(feat.curr, clas)
            feat.curr[i].key = clas
            feat.avgs[i].key = clas
            # Apply the last round of updates, multiplied by the time unchanged
            feat.avgs[i].val += (self.time - feat.times[i].val) * feat.curr[i].val
            feat.curr[i].val += upd
            feat.times[i].val = self.time
