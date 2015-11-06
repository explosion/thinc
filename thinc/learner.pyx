from libc.stdio cimport fopen, fclose, fread, fwrite, feof, fseek
from libc.errno cimport errno
from libc.string cimport memcpy
from libc.string cimport memset

from libc.stdlib cimport qsort
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc

import random
import cython
from os import path

from murmurhash.mrmr cimport hash64
from cymem.cymem cimport Address

from preshed.maps cimport MapStruct
from preshed.maps cimport map_get

from .typedefs cimport feat_t

from cython.parallel import prange
cimport numpy as np
import numpy as np

from contextlib import contextmanager


cimport sparse

from sparse cimport SparseArrayC
from .api cimport Example

include "compile_time_constants.pxi"


