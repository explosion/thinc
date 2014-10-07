# cython: profile=True
from libc.stdint cimport uint64_t
from cymem.cymem cimport Pool

from murmurhash.mrmr cimport hash64


DEF MAX_FEAT_LEN = 10


cdef class Extractor:
    def __init__(self, templates, match_templates, bag_of_words=None):
        assert not bag_of_words
        self.mem = Pool()
        self.trie = SequenceIndex(offset=1)
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
            pred.n = len(args) + 1
            pred.raws[0] = pred.id
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
        self._features = <feat_t*>self.mem.alloc(self.nr_feat + 1, sizeof(feat_t))

    def __call__(self, list py_context):
        cdef size_t i
        cdef size_t c
        context = <size_t*>self.mem.alloc(len(py_context), sizeof(size_t))
        for i, c in enumerate(py_context):
            context[i] = c
        feats = self.extract(context)
        self.mem.free(context)
        output = []
        i = 0
        while feats[i] != 0:
            output.append(feats[i])
            i += 1
        return output

    cdef int count(self, dict counts, feat_t* feats, double inc) except -1:
        cdef size_t i = 0
        while feats[i] != 0:
            counts[feats[i]] += inc
            i += 1

    cdef feat_t* extract(self, size_t* context) except NULL:
        cdef:
            size_t i, j, size
            size_t value
            bint seen_non_zero
            Template* pred
        cdef feat_t* features = self._features
        # Extra trick:
        # Always include this feature to give classifier priors over the classes
        features[0] = 1
        for i in range(self.nr_template):
            pred = &self.templates[i]
            pred.raws[0] = i
            seen_non_zero = False
            # Offset by 1, as the template ID is at position 0
            for j in range(1, pred.n):
                value = context[pred.args[j-1]]
                pred.raws[j] = value
                if value != 0:
                    seen_non_zero = True
            if seen_non_zero:
                features[i+1] = self.trie.index(pred.raws, pred.n)
        features[i+2] = 0
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
