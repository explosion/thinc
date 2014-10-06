# cython: profile=True
from libc.stdint cimport uint64_t
from cymem.cymem cimport Pool

from murmurhash.mrmr cimport hash64


DEF MAX_FEAT_LEN = 10


cdef class Extractor:
    def __cinit__(self, templates, match_templates, bag_of_words=None):
        assert not bag_of_words
        self.mem = Pool()
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        templates = tuple(sorted(set([tuple(sorted(f)) for f in templates])))
        self.nr_template = len(templates)
        self.templates = <Template*>self.mem.alloc(self.nr_template, sizeof(Template))
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
        self.match_preds = <MatchPred*>self.mem.alloc(self.nr_match, sizeof(MatchPred))
        cdef MatchPred* match_pred
        for id_, (idx1, idx2) in enumerate(match_templates):
            match_pred = &self.match_preds[id_]
            match_pred.id = id_ + self.nr_template
            match_pred.idx1 = idx1
            match_pred.idx2 = idx2
        self.nr_feat = self.nr_template + (self.nr_match * 2) + 1
        self.features = <size_t*>self.mem.alloc(self.nr_feat, sizeof(Feature*))
        for i in range(self.nr_feat):
            self.features[i] = <size_t>self.mem.alloc(1, sizeof(Feature))

    cdef int count(self, dict counts, feat_t* features, double inc) except -1:
        cdef size_t template_id
        cdef size_t key
        cdef Feature* feature
        for template_id in range(self.nr_feat):
            feature = <Feature*>features[template_id] 
            if not feature.is_active:
                continue
            key = <size_t>feature
            if key not in counts:
                counts[key] = 0
            counts[key] += inc

    cdef size_t* extract(self, size_t* context) except NULL:
        cdef:
            size_t i, j, size
            size_t value
            bint seen_non_zero
            Template* pred
        cdef size_t* features = self.features
        return features
        cdef size_t f = 0
        # Extra trick:
        # Always include this feature to give classifier priors over the classes
        cdef Feature* feat = <Feature*>features[0]
        feat.vals[0] = 1
        feat.n = 1
        feat.is_active = True
        for i in range(self.nr_template):
            f += 1
            feat = <Feature*>features[f]
            pred = &self.templates[i]
            feat.is_active = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                feat.vals[j] = value
                if value != 0:
                    feat.is_active = True
        return features
        """
        cdef MatchPred* match_pred
        cdef size_t match_id
        for match_id in range(self.nr_match):
            match_pred = &self.match_preds[match_id]
            value = context[match_pred.idx1]
            if value != 0 and value == context[match_pred.idx2]:
                match_pred.raws[0] = value
                match_pred.raws[1] = match_pred.id
                features[f] = hash64(match_pred.raws, sizeof(match_pred.raws),
                                     match_pred.id)
                f += 1
                match_pred.raws[0] = 0
                features[f] = hash64(match_pred.raws, sizeof(match_pred.raws),
                                     match_pred.id)
                f += 1
            else:
                features[f] = 0
                f += 1
                features[f] = 0
                f += 1
        """
