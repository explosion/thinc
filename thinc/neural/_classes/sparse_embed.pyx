import contextlib
from libc.stdint cimport uint64_t
from libc.string cimport memcpy, memset

from collections import Counter

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t
from numpy cimport ndarray
cimport numpy as np
import numpy

from ...typedefs cimport weight_t, atom_t, feat_t
from ...typedefs cimport len_t, idx_t
from ...linalg cimport MatMat, MatVec, VecVec, Vec
from ...structs cimport EmbedC, do_update_t
from ...api import layerize
from ..ops import NumpyOps
from .._classes.embed import Embed


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef void initC(EmbedC* embedC, Pool mem, int width, int nr_support) except *: 
    embedC.nr_support = nr_support
    embedC.width = width
    embedC.default = <float*>mem.alloc(width * nr_support, sizeof(float))
    embedC.d_default = <float*>mem.alloc(width * nr_support, sizeof(float))
    embedC.vectors = <MapC*>mem.alloc(1, sizeof(MapC))
    embedC.d_vectors = <MapC*>mem.alloc(1, sizeof(MapC))
    Map_init(mem, embedC.vectors, 8)
    Map_init(mem, embedC.d_vectors, 8)
    for i in range(width):
        embedC.default[i] = numpy.random.uniform(-0.05, 0.05)


cdef void embedC(float* out,
        const uint64_t* ids, int nr_id, const MapC* vectors,
        const float* default, int width) nogil:
    cdef uint64_t id_
    for id_ in ids[:nr_id]:
        emb = <const float*>Map_get(vectors, id_)
        if emb is not NULL:
            # Use (trainable) default values if feature is missing.
            memcpy(out, emb, width * sizeof(out[0]))
        out += width


cdef void insert_missingC(Pool mem, MapC* vectors, MapC* d_vectors,
        const float* default, int width, int nr_support,
        const uint64_t* ids, int nr_id) except *:
    cdef uint64_t id_
    for id_ in ids[:nr_id]:
        emb = <float*>Map_get(vectors, id_)
        if emb is NULL:
            emb = <float*>mem.alloc(width * nr_support, sizeof(emb[0]))
            # Inherit default, including averages
            memcpy(emb, default, sizeof(emb[0]) * width * nr_support)
            Map_set(mem, vectors, id_, emb)
            Map_set(mem, d_vectors, id_,
                mem.alloc(width, sizeof(float)))


cdef void inc_gradientsC(MapC* d_vectors, float* d_default,
        const float* delta, int width, const uint64_t* ids, int nr_id) nogil:
    cdef uint64_t id_
    for id_ in ids[:nr_id]:
        d_emb = <weight_t*>Map_get(d_vectors, id_)
        # If the feature was missing, update the default
        VecVec.add_i(d_emb or d_default, delta, 1., width)
        delta += width


cdef void averageC(EmbedC* layer) nogil:
    cdef key_t key
    cdef void* value
    cdef int i = 0
    while Map_iter(layer.vectors, &i, &key, &value):
        emb = <float*>value
        memcpy(emb, &emb[layer.width], layer.width * sizeof(emb[0]))
    # Additionally, average defaults
    emb = layer.default
    memcpy(emb, &emb[layer.width], layer.width * sizeof(emb[0]))


def SparseEmbed(nO, **kwargs):
    impl = _SparseEmbed(nO, **kwargs)
    model = layerize(impl.begin_update)
    model._impl = impl
    model.on_data_hooks = [impl.lsuv_init]
    return model


cdef class _SparseEmbed:
    cdef Pool mem
    cdef EmbedC* c
    cdef readonly int nO
    cdef public object ops
    cdef public object on_data_hooks
    cdef public object on_init_hooks
    
    def __init__(self, int nO, **kwargs):
        self.nO = nO
        self.mem = Pool()
        self.ops = NumpyOps()
        self.c = <EmbedC*>self.mem.alloc(sizeof(EmbedC), 1)
        initC(self.c, self.mem, self.nO, 1)

    def begin_update(self, uint64_t[::1] ids, float drop=0.):
        outer_ids = ids
        def finish_update(float[:, ::1] delta, sgd=None):
            cdef uint64_t[::1] ids = outer_ids
            inc_gradientsC(self.c.d_vectors, self.c.d_default,
                &delta[0, 0], self.c.width, &ids[0], ids.shape[0])
            cdef uint64_t id_
            if sgd is not None:
                for id_ in ids[:ids.shape[0]]:
                    emb = <float*>Map_get(self.c.vectors, id_)
                    if emb is NULL:
                        continue
                    d_emb = <float*>Map_get(self.c.d_vectors, id_)
                    if d_emb is NULL:
                        continue
                    for d in d_emb[:self.c.width]:
                        if d != 0.0:
                            py_emb = ptr2ndarray(emb, self.c.width)
                            py_grad = ptr2ndarray(d_emb, self.c.width)
                            sgd(py_emb, py_grad, key=id_)
                            break
                py_emb = ptr2ndarray(self.c.default, self.c.width)
                py_grad = ptr2ndarray(self.c.d_default, self.c.width)
                sgd(py_emb, py_grad, key=id(self.mem))
            insert_missingC(self.mem, self.c.vectors, self.c.d_vectors,
                self.c.default, self.c.width, self.c.nr_support,
                &ids[0], ids.shape[0])
            return None
        cdef int nr_id = ids.shape[0]
        cdef ndarray[float, ndim=2] output = numpy.zeros((nr_id, self.c.width), dtype='float32')
        embedC(<float*>output.data,
            &ids[0], nr_id, self.c.vectors, self.c.default, self.c.width)
        return output, finish_update

    def lsuv_init(self, _, X, y=None):
        dense_embed = Embed(self.nO, self.nO)
        for hook in dense_embed.on_data_hooks:
            hook(dense_embed, self.ops.asarray(X, dtype='i'), y)
        cdef float[:, ::1] vectors = dense_embed._embed(X)
        cdef uint64_t id_
        cdef int i, j
        for i, id_ in enumerate(X):
            emb = <float*>Map_get(self.c.vectors, id_)
            if emb is NULL:
                emb = <float*>self.mem.alloc(self.c.width * self.c.nr_support, sizeof(emb[0]))
                Map_set(self.mem, self.c.vectors, id_, emb)
                d_emb = <float*>self.mem.alloc(self.c.width, sizeof(emb[0]))
                Map_set(self.mem, self.c.d_vectors, id_, d_emb)
                for j in range(vectors.shape[1]):
                    emb[j] = vectors[i, j]


cdef ndarray ptr2ndarray(float* ptr, int n):
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp>n
    # Create a 1D array, of length 'size'
    ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT, ptr)
    return ndarray
