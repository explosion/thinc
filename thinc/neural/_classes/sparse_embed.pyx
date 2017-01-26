from libc.stdint cimport uint64_t
from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool
from preshed.maps cimport MapStruct as MapC
from preshed.maps cimport map_init as Map_init
from preshed.maps cimport map_set as Map_set
from preshed.maps cimport map_get as Map_get
from preshed.maps cimport map_iter as Map_iter
from preshed.maps cimport key_t
from numpy cimport ndarray

from ...typedefs cimport weight_t, atom_t, feat_t
from ...typedefs cimport len_t, idx_t
from ...linalg cimport MatMat, MatVec, VecVec, Vec
from ...structs cimport EmbedC, do_update_t
from ..ops import NumpyOps


cdef void initC(EmbedC* embedC, Pool mem, int width, int nr_support) except *: 
    embedC.nr_support = nr_support
    embedC.width = width
    embedC.default = <float*>mem.alloc(width * nr_support, sizeof(float))
    embedC.d_default = <float*>mem.alloc(width * nr_support, sizeof(float))
    embedC.vectors = <MapC*>mem.alloc(1, sizeof(MapC))
    embedC.d_vectors = <MapC*>mem.alloc(1, sizeof(MapC))
    Map_init(mem, embedC.vectors, 8)
    Map_init(mem, embedC.d_vectors, 8)


cdef void embedC(float* out,
        const uint64_t* ids, int nr_id, const MapC* vectors,
        const float* default, int width) nogil:
    for id_ in ids[:nr_id]:
        emb = <const float*>Map_get(vectors, id_)
        # Use (trainable) default values if feature is missing.
        memcpy(out, emb or default, width * sizeof(out[0]))
        out += width


cdef void updateC(MapC* vectors, MapC* d_vectors, float* default, float* d_default,
        int width, const uint64_t* ids, int nr_id) nogil:
    cdef float* d_emb
    cdef float* emb
    cdef float d
    cdef uint64_t id_
    for id_ in ids[:nr_id]:
        d_emb = <float*>Map_get(d_vectors, id_)
        emb = <float*>Map_get(vectors, id_)
        if d_emb is NULL or emb is NULL:
            d_emb = d_default
            emb = default
        for d in d_emb[:width]:
            if d != 0.0:
                # TODO: Plug in solvers here
                VecVec.add_i(emb, d_emb, -0.001, width)
                memset(d_emb, 0, width * sizeof(d_emb[0]))
                #do_update(emb, d_emb, width, NULL)
                break


cdef void insert_missingC(Pool mem, MapC* vectors, MapC* d_vectors,
        const float* default, int width, int nr_support,
        const uint64_t* ids, int nr_id) except *:
    for id_ in ids[:nr_id]:
        emb = <float*>Map_get(vectors, id_)
        if emb is NULL:
            emb = <float*>mem.alloc(width * nr_support, sizeof(emb[0]))
            # Inherit default, including averages
            memcpy(emb, default, sizeof(emb[0]) * width * nr_support)
            Map_set(mem, vectors, id_, emb)
            Map_set(mem, d_vectors, id_,
                mem.alloc(width, sizeof(float)))


cdef void fine_tuneC(MapC* d_vectors, float* d_default,
        const float* delta, int width, const uint64_t* ids, int nr_id) nogil:
    for id_ in ids[:nr_id]:
        d_emb = <weight_t*>Map_get(d_vectors, id_)
        # If the feature was missing, update the default
        VecVec.add_i(d_emb or d_default, delta, 1., width)


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


cdef class SparseEmbed:
    cdef Pool mem
    cdef readonly int nO
    cdef EmbedC* c

    def __init__(self, nO, **kwargs):
        self.mem = Pool()
        self.c = <EmbedC*>self.mem.alloc(sizeof(EmbedC), 1)
        self.nO = nO

        initC(self.c, self.mem, nO, 1)

    def predict(self, uint64_t[::1] ids):
        ops = NumpyOps()
        cdef int nr_id = ids.shape[0]
        cdef ndarray[float, ndim=2] output = ops.allocate((nr_id, self.c.width), dtype='float32')
        embedC(<float*>output.data,
            &ids[0], nr_id, self.c.vectors, self.c.default, self.c.width)
        return output

    def begin_update(self, uint64_t[::1] ids, float drop=0.):
        def finish_update(float[:, ::1] delta, sgd=None):
            fine_tuneC(self.c.d_vectors, self.c.d_default,
                &delta[0, 0], self.c.width, &ids[0], ids.shape[0])
            insert_missingC(self.mem, self.c.vectors, self.c.d_vectors,
                self.c.default, self.c.width, self.c.nr_support,
                &ids[0], ids.shape[0])
            updateC(self.c.vectors, self.c.d_vectors, self.c.default, self.c.d_default,
                self.c.width, &ids[0], ids.shape[0])
            return None
        return self.predict(ids), finish_update
