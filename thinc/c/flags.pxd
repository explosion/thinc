cimport libc.stdint
# TODO: Mark this volatile
ctypedef libc.stdint.int32_t flag_t

cdef extern from "unistd.h":
    cdef void usleep(unsigned int microseconds) nogil

ctypedef void* (*run_task_f)(void*) nogil

cdef struct task_s:
    void* run(void*) nogil
    void* args

cdef int count_tasks_remaining(int* fwd, int* bwd, const flag_t* status,
        int layer_id, int N) nogil

cdef void yield_output(flag_t* status, int size, int layer_id) nogil

cdef void yield_gradient(flag_t* status, int size, int layer_id) nogil

cdef void get_input(int* index, int* size, flag_t* status,
        int layer_id, int max_batch, int N) nogil
 
cdef void get_gradient(int* index, int* size, flag_t* status,
        int layer_id, int max_batch, int N) nogil
