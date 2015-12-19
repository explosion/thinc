cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from libc.math cimport sqrt as c_sqrt

from .api cimport Example
from .typedefs cimport time_t, feat_t, weight_t, class_t
from .structs cimport SparseAverageC
from .sparse cimport SparseArray
from .blas cimport Vec, VecVec


cdef SparseAverageC* init_feat(int clas, weight_t weight, int time) except NULL:
    feat = <SparseAverageC*>PyMem_Malloc(sizeof(SparseAverageC))
    feat.curr  = SparseArray.init(clas, weight)
    feat.avgs  = SparseArray.init(clas, 0)
    feat.times = SparseArray.init(clas, <weight_t>time)
    return feat


cdef int update_feature(SparseAverageC* feat, int key, weight_t upd, int time) except -1:
    cdef int i = SparseArray.find_key(feat.curr, key)
    if i >= 0:
        is_resized = False
    else:
        is_resized = True
        feat.curr = SparseArray.resize(feat.curr)
        feat.avgs = SparseArray.resize(feat.avgs)
        feat.times = SparseArray.resize(feat.times)
        i = SparseArray.find_key(feat.curr, key)
   
    feat.curr[i].key = key
    feat.avgs[i].key = key
    feat.avgs[i].key = key
    # Apply the last round of updates, multiplied by the time unchanged
    feat.avgs[i].val += (time - feat.times[i].val) * feat.curr[i].val
    feat.curr[i].val += upd
    feat.times[i].val = time
    return is_resized


@cython.cdivision(True)
cdef int average_weights(SparseAverageC* feat, time_t time) except -1:
    cdef time_t unchanged

    cdef int i = 0
    while feat.curr[i].key >= 0:
        unchanged = (time + 1) - <time_t>feat.times[i].val
        feat.avgs[i].val += unchanged * feat.curr[i].val
        feat.curr[i].val = feat.avgs[i].val / time
        i += 1


cdef class Updater:
    def __init__(self, PreshMap weights):
        self.time = 0
        self.train_weights = PreshMap()
        self.weights = weights
        self.mem = Pool()

    def __dealloc__(self):
        cdef size_t feat_addr
        # Use 'raw' memory management, instead of cymem.Pool, for weights.
        # The memory overhead of cymem becomes significant here.
        if self.train_weights is not None:
            for feat_addr in self.train_weights.values():
                if feat_addr != 0:
                    feat = <SparseAverageC*>feat_addr
                    PyMem_Free(feat.avgs)
                    PyMem_Free(feat.times)

    def __call__(self, Example eg):
        raise NotImplementedError

    cdef void update(self, ExampleC* eg) except *:
        raise NotImplementedError

    cdef void update_embedding(self, feat_t feat_id, weight_t feat_value,
                               const weight_t* gradient, int32_t length) except *:
        weights = <weight_t*>self.weights.get(feat_id)
        support = <weight_t*>self.train_weights.get(feat_id)
        if weights is NULL:
            weights = <weight_t*>self.mem.alloc(length, sizeof(weights[0]))
            support = <weight_t*>self.mem.alloc(length, sizeof(weights[0]))
            self.weights.set(feat_id, <void*>weights)
            self.train_weights.set(feat_id, <void*>support)

        adagrad(
            weights,
            support,
            feat_value,
            gradient,
            length,
            <void*>&self.c.hyper_params
        )

    cpdef int update_sparse(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        feat = <SparseAverageC*>self.train_weights.get(feat_id)
        if feat == NULL:
            feat = init_feat(clas, upd, self.time)
            self.train_weights.set(feat_id, feat)
            self.weights.set(feat_id, feat.curr)
        else:  
            is_resized = update_feature(feat, clas, upd, self.time)
            if is_resized:
                self.weights.set(feat_id, feat.curr)


cdef class AveragedPerceptronUpdater(Updater):
    def __call__(self, Example eg):
        self.update(eg.c.gradient, eg.c.nr_class, eg.c.features, eg.c.nr_feat)

    cdef void update(self, const weight_t* upd, int32_t nr_class,
                     const FeatureC* feats, int32_t nr_feat) except *:
        cdef weight_t weight, upd
        cdef feat_t feat_id
        cdef int i
        
        self.time += 1
        
        for clas in range(nr_class):
            if gradient[clas] != 0:
                for i in range(nr_feat):
                    if feats[i].val != 0:
                        self.update_weight(feats[i].key, feats[i].val * upd[clas])
    
    def end_training(self):
        cdef feat_id
        cdef size_t feat_addr
        for feat_id, feat_addr in self.train_weights.items():
            if feat_addr != 0:
                average_weights(<SparseAverageC*>feat_addr, self.time)



