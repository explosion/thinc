# cython: profile=True
from libc.stdint cimport uint64_t
from cymem.cymem cimport Pool

from murmurhash.mrmr cimport hash64


DEF MAX_TEMPLATE_LEN = 10
DEF MAX_FEATS = 200


cdef int conj_feat(feat_t* feats, int f, int templ_id, atom_t* atoms, int n,
                   void* extra_args) nogil:
    feats[f] = hash64(atoms, n * sizeof(atom_t), templ_id)
    f += 1
    feats[f] = 0
    return f


cdef int backoff_feat(feat_t* feats, int f, int templ_id, atom_t* atoms, int n,
                   void* extra_args) nogil:
    cdef int i
    for i in range(n):
        feats[f] = hash64(atoms, (n-1) * sizeof(atom_t), templ_id)
        f += 1
    feats[f] = 0
    return f


cdef int match_feat(feat_t* feats, int f, int templ_id, atom_t* atoms, int n,
                     void* extra_args) nogil:
    cdef int i
    for i in range(1, n):
        if atoms[i] != atoms[0]:
            break
    else:
        feats[f] = templ_id
        f += 1
        feats[f] = 0
    return f



FEATURE_FUNCS[<int>ConjFeat] = conj_feat
FEATURE_FUNCS[<int>BackoffFeat] = backoff_feat
FEATURE_FUNCS[<int>MatchFeat] = match_feat


cdef class Extractor:
    """Extract composite features from a sequence of atomic values, according to
    the schema specified by a list of templates.  Each template specifies how
    to construct a function that's roughly f(atom_t*) --> feat_t*.
    
    Let's say we're doing POS tagging. Atoms is a 5-tuple, with the values for:
    (prev prev tag, prev tag, prev word, curr word, next word)

    E.g. we're tagging "gray" in "The old gray mare", then
    ignoring implementation specifics we might have:

    atoms = (DT, JJ, old, gray, mare)

    If we have two feature templates:

    ((prev tag, prev word), Conj)
    ((prev word, curr word), Match)

    This will result in:

    Conj(JJ, old) --> hash((JJ, old))
    Match(old, gray) --> bool(old == old)

    Care is taken to make sure this is all suitably efficient, e.g. we take
    C function pointers as arguments, not Python functions.

    There's actually no Python interface at all --- to use via Python, use the
    instance.Instance.extract method, which takes an Extractor as an argument,
    and fills the Instance's feats array.
    """
    def __init__(self, templates):
        self.mem = Pool()
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        templates = tuple(sorted(set([tuple(sorted(f)) for f in templates])))
        self.n = len(templates)
        self.templates = <Template*>self.mem.alloc(self.n, sizeof(Template))
        # Sort each feature, and sort and unique the set of them
        cdef int i, j
        cdef FeatureFuncName func_name
        for i, (indices, func_name) in enumerate(templates):
            assert len(indices) < MAX_TEMPLATE_LEN
            for j, idx in enumerate(sorted(indices)):
                self.templates[i].indices[j] = idx
            self.templates[i].n = len(indices)
            self.templates[i].func = FEATURE_FUNCS[<int>func_name]

    cdef int count(self, dict counts, feat_t* feats, weight_t inc) except -1:
        cdef int i
        while feats[i] != 0:
            counts[feats[i]] += inc
            i += 1

    cdef int extract(self, feat_t* feats, weight_t* values, atom_t* atoms,
                     void* extra_args) except -1:
        cdef:
            int i, j, size
            feat_t value
            bint seen_non_zero
            Template* templ
        cdef int f = 0
        # Extra trick:
        # Always include this feature to give classifier priors over the classes
        feats[0] = 1
        f += 1
        for i in range(self.n):
            templ = &self.templates[i]
            for j in range(templ.n):
                templ.atoms[j] = atoms[templ.indices[j]]
            f += templ.func(feats, f, templ.id, templ.atoms, templ.n, extra_args)
        return f
