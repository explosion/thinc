from libc.stdint cimport uint64_t
from cymem.cymem cimport Pool

from murmurhash.mrmr cimport hash64

from ..extra.eg cimport Example

include "../compile_time_constants.pxi"


cdef class ConjunctionExtracter:
    """Extract composite features from a sequence of atomic values, according to
    the schema specified by a list of templates.
    """
    def __init__(self, templates, linear_mode=True):
        self.mem = Pool()
        nr_atom = 0
        for templ in templates:
            nr_atom = max(nr_atom, max(templ))
        self.linear_mode = linear_mode
        self.nr_atom = nr_atom
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        templates = tuple(sorted(set([tuple(sorted(f)) for f in templates])))
        self._py_templates = templates
        self.nr_embed = 1
        self.nr_templ = len(templates) + 1
        self.templates = <TemplateC*>self.mem.alloc(len(templates), sizeof(TemplateC))
        # Sort each feature, and sort and unique the set of them
        cdef int i, j, idx
        for i, indices in enumerate(templates):
            assert len(indices) < MAX_TEMPLATE_LEN
            for j, idx in enumerate(sorted(indices)):
                self.templates[i].indices[j] = idx
            self.templates[i].length = len(indices)

    def __call__(self, Example eg):
        eg.c.nr_feat = self.set_features(eg.c.features, eg.c.atoms)

    cdef int set_features(self, FeatureC* feats, const atom_t* atoms) nogil:
        cdef int n_feats = 0
        if self.linear_mode:
            feats[0].key = 1
            feats[0].value = 1
            n_feats += 1
        cdef bint seen_non_zero
        cdef int templ_id
        cdef int i
        cdef atom_t[MAX_TEMPLATE_LEN] extracted
        for templ_id in range(self.nr_templ-1):
            templ = self.templates[templ_id]
            if not self.linear_mode and templ.length == 1:
                feats[n_feats].key = atoms[templ.indices[0]]
                feats[n_feats].value = 1
                feats[n_feats].i = templ_id
                n_feats += 1
                continue
            seen_non_zero = False
            for i in range(templ.length):
                extracted[i] = atoms[templ.indices[i]]
                seen_non_zero = seen_non_zero or extracted[i]
            if seen_non_zero:
                feats[n_feats].key = hash64(extracted, templ.length * sizeof(extracted[0]),
                                            templ_id if self.linear_mode else 0)
                feats[n_feats].value = 1
                feats[n_feats].i = templ_id
                n_feats += 1
        return n_feats

    def __reduce__(self):
        return (self.__class__, (self.nr_atom, self._py_templates))
