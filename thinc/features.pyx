# cython: profile=True
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

    cdef Feature* get_feats(self, atom_t* atoms, int* length) except NULL:
        length[0] = self.set_feats(self.feats, atoms)
        return self.feats

    cdef int set_feats(self, Feature* feats, atom_t* atoms) except -1:
        cdef Template* templ
        cdef Feature* feat
        feats[0].i = 0
        feats[0].key = 1
        feats[0].value = 1
        cdef int i, j
        for i in range(1, self.n_templ):
            templ = &self.templates[i-1]
            feat = &feats[i]
            feat.i = i
            if templ.length == 1:
                feat.key = atoms[templ.indices[0]]
                feat.value = 1
            else:
                for j in range(templ.length):
                    templ.atoms[j] = atoms[templ.indices[j]]
                feat.i = i
                feat.key = hash64(templ.atoms, templ.length * sizeof(atom_t), 0)
                feat.value = 1
        return self.n_templ


cdef int count_feats(dict counts, Feature* feats, int n_feats, weight_t inc) except -1:
    cdef int i
    cdef feat_t f
    for i in range(n_feats):
        assert feats[i].i == i
        f = feats[i].key
        key = (feats[i].i, f)
        counts.setdefault(key, 0)
        counts[key] += inc
