# cython: profile=True
from libc.stdlib cimport malloc, free, calloc
from libc.stdint cimport uint64_t

from thinc.ext.murmurhash cimport *

DEF MAX_FEAT_LEN = 10


cdef class Extractor:
    def __cinit__(self, templates, match_templates, bag_of_words=None):
        assert not bag_of_words
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        templates = tuple(sorted(set([tuple(sorted(f)) for f in templates])))
        self.nr_template = len(templates)
        self.templates = <Template*>calloc(self.nr_template, sizeof(Template))
        # Sort each feature, and sort and unique the set of them
        cdef Template* pred
        for id_, args in enumerate(templates):
            assert len(args) < MAX_FEAT_LEN
            pred = &self.templates[id_]
            pred.id = id_
            pred.n = len(args)
            for i, element in enumerate(sorted(args)):
                pred.args[i] = element
        self.nr_match = len(match_templates)
        self.match_preds = <MatchPred*>calloc(self.nr_match, sizeof(MatchPred))
        cdef MatchPred* match_pred
        for id_, (idx1, idx2) in enumerate(match_templates):
            match_pred = &self.match_preds[id_]
            match_pred.id = id_ + self.nr_template
            match_pred.idx1 = idx1
            match_pred.idx2 = idx2
        self.nr_feat = self.nr_template + (self.nr_match * 2) + 2

    def __dealloc__(self):
        free(self.templates)
        free(self.match_preds)

    cdef int count(self, dict counts, uint64_t* features, double inc) except -1:
        cdef size_t f = 0
        cdef uint64_t value
        while features[f] != 0:
            value = features[f]
            if value not in counts:
                counts[value] = 0
            counts[value] += inc
            f += 1

    cdef int extract(self, uint64_t* features, size_t* context) except -1:
        cdef:
            size_t i, j, size
            uint64_t value
            bint seen_non_zero
            Template* pred
        cdef size_t f = 0
        # Extra trick:
        # Always include this feature to give classifier priors over the classes
        features[0] = 1
        f += 1
        for i in range(self.nr_template):
            pred = &self.templates[i]
            seen_non_zero = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                if value != 0:
                    seen_non_zero = True
                pred.raws[j] = value
            if seen_non_zero:
                pred.raws[pred.n] = pred.id
                features[f] = MurmurHash64A(pred.raws, sizeof(pred.raws), i)
                f += 1
        cdef MatchPred* match_pred
        cdef size_t match_id
        for match_id in range(self.nr_match):
            match_pred = &self.match_preds[match_id]
            value = context[match_pred.idx1]
            if value != 0 and value == context[match_pred.idx2]:
                match_pred.raws[0] = value
                match_pred.raws[1] = match_pred.id
                features[f] = MurmurHash64A(match_pred.raws, sizeof(match_pred.raws),
                                            match_pred.id)
                f += 1
                match_pred.raws[0] = 0
                features[f] = MurmurHash64A(match_pred.raws, sizeof(match_pred.raws),
                                            match_pred.id)
                f += 1
        features[f] = 0
