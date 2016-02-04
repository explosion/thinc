# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_set as Map_set

from ..structs cimport FeatureC
from ..structs cimport ConstantsC

from ..typedefs cimport len_t
from ..typedefs cimport idx_t

from ..linalg cimport MatMat, MatVec, VecVec, Vec

from ..structs cimport do_feed_fwd_t
from ..structs cimport do_feed_bwd_t
from ..structs cimport do_update_t


DEF EPS = 0.00000001 
DEF ALPHA = 1.0


