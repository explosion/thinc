cdef class AveragedPerceptron:
    cdef Pool mem
    cdef PreshMap weights
    cdef PreshMap averages
    cdef ConjunctionExtractor extractor
    cdef int time
    
    cdef void set_scores(self, weight_t* scores, const FeatureC* feats, int nr_feat) nogil
    cdef void update(self, ExampleC* eg) except *
    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1
