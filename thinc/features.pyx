from libc.stdint cimport uint64_t
from cymem.cymem cimport Pool

from murmurhash.mrmr cimport hash64


DEF MAX_TEMPLATE_LEN = 10


cdef class Extractor:
    """Extract composite features from a sequence of atomic values, according to
    the schema specified by a list of templates.
    """
    def __init__(self, templates):
        self.mem = Pool()
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        templates = tuple(sorted(set([tuple(sorted(f)) for f in templates])))
        self.n_templ = len(templates) + 1
        self.templates = <Template*>self.mem.alloc(len(templates), sizeof(Template))
        self.feats = <Feature*>self.mem.alloc(self.n_templ, sizeof(Feature))
        # Sort each feature, and sort and unique the set of them
        cdef int i, j, idx
        for i, indices in enumerate(templates):
            assert len(indices) < MAX_TEMPLATE_LEN
            for j, idx in enumerate(sorted(indices)):
                self.templates[i].indices[j] = idx
            self.templates[i].length = len(indices)

    cdef Feature* get_feats(self, atom_t* atoms, int* length) nogil:
        length[0] = self.set_feats(self.feats, atoms)
        return self.feats

    cdef int set_feats(self, Feature* feats, atom_t* atoms) nogil:
        cdef Template* templ
        cdef Feature* feat
        feats[0].i = 0
        feats[0].key = 1
        feats[0].value = 1
        cdef bint seen_non_zero
        cdef int templ_id
        cdef int n_feats = 1
        cdef int i
        for templ_id in range(self.n_templ-1):
            templ = &self.templates[templ_id]
            seen_non_zero = False
            for i in range(templ.length):
                templ.atoms[i] = atoms[templ.indices[i]]
                seen_non_zero = seen_non_zero or templ.atoms[i]
            if seen_non_zero:
                feat = &feats[n_feats]
                feat.i = templ_id
                feat.key = hash64(templ.atoms, templ.length * sizeof(atom_t), templ_id)
                feat.value = 1
                n_feats += 1
        return n_feats


cdef int count_feats(dict counts, const Feature* feats, int n_feats, weight_t inc) except -1:
    cdef int i
    cdef feat_t f
    for i in range(n_feats):
        f = feats[i].key
        key = (feats[i].i, f)
        counts.setdefault(key, 0)
        counts[key] += inc
