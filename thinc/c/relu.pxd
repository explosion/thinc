from flags cimport task_s, flag_t

cdef task_s make_task(flag_t* status, int layer_id,
        float* X, float* dX, int nr_dim, int batch_size, int N) nogil
