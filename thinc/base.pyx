from .extra.eg cimport Example


cdef class Model:
    def __call__(self, Example eg):
        raise NotImplementedError

    def train_example(self, Example eg):
        raise NotImplementedError

    def predict_example(self, Example eg):
        raise NotImplementedError

    def dump(self, loc):
        raise NotImplementedError

    def load(self, loc):
        raise NotImplementedError

    def end_training(self):
        pass

    @property
    def nr_feat(self):
        raise NotImplementedError

    cpdef int update_weight(self, feat_t feat_id, class_t clas, weight_t upd) except -1:
        pass
    cdef void set_scoresC(self, weight_t* scores,
            const void* feats, int nr_feat, int is_sparse) nogil:
        pass
