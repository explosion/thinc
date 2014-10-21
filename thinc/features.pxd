from cymem.cymem cimport Pool

from .typedefs cimport *

DEF MAX_TEMPLATE_LEN = 10
DEF MAX_FEATS = 200


ctypedef int (*eval_func)(feat_t*, int, atom_t*, int, void*) nogil


cpdef enum FeatureFuncName:
    NonZeroConjFeat
    ConjFeat
    BackoffFeat
    MatchFeat
    SumFeat
    N_FEATURE_FUNCS


cdef eval_func[<int>N_FEATURE_FUNCS] FEATURE_FUNCS


cdef struct Template:
    int n
    int[MAX_TEMPLATE_LEN] indices
    atom_t[MAX_TEMPLATE_LEN] atoms
    eval_func func


cdef class Extractor:
    cdef Pool mem
    cdef Template* templates
    cdef readonly int n
    cdef int extract(self, feat_t* feats, weight_t* values, atom_t* atoms, void* extra_args) except -1
    cdef int count(self, dict counts, feat_t* feats, weight_t inc) except -1
