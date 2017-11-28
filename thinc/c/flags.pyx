'''
Flag status

* 0: Unset
* 1: Op 1 in progress
* 2: Op 1 complete
* 3: Op 2 in progress
* 4: Op 2 complete
*-5: loss set
*-4: d_op 2 in progress
*-3: d_op 2 complete
*-2: d_op 1 in progress
*-1: d_op 1 complete
'''
# cython: infer_types=True
cimport libc.stdint
# TODO: Mark this volatile
ctypedef libc.stdint.int32_t flag_t

cdef extern from "unistd.h":
    cdef void usleep(unsigned int microseconds) nogil


cdef int count_tasks_remaining(int* fwd, int* bwd, const flag_t* status, int layer_id, int N) nogil:
    '''Count how many instances haven't been processed by the layer.'''
    cdef int n_todo_f = 0
    cdef int n_todo_b = 0
    for flag in status[:N]:
        if 0 <= flag < layer_id:
            n_todo_f += 1
            n_todo_b += 1
        elif flag < -layer_id:
            n_todo_b += 1
    fwd[0] = n_todo_f
    bwd[0] = n_todo_b


cdef void get_input(int* index, int* size, flag_t* status,
        int layer_id, int max_batch, int N) nogil:
    '''Find a sequence of examples to work on for the forward pass.
    The examples need to be contiguous, and they all need to be ready.
    The status of the examples is updated, marking them as 'in progress'.'''
    _get_examples(index, size, status,
        layer_id-1, layer_id, max_batch, N)


cdef void get_gradient(int* index, int* size, flag_t* status,
        int layer_id, int max_batch, int N) nogil:
    '''Find a sequence of examples to work on for the forward pass.
    The examples need to be contiguous, and they all need to be ready.
    The status of the examples is updated, marking them as 'in progress'.'''
    _get_examples(index, size, status,
        -(layer_id+1), -layer_id, max_batch, N)


cdef inline void _get_examples(int* index, int* size, flag_t* status,
        int target_status, int new_status, int max_batch, int N) nogil:
    '''Find a sequence of examples to work on.
    The examples need to be contiguous, and they all need to be ready.
    The status of the examples is updated, marking them as 'in progress'.'''
    cdef int i, start, end
    for i in range(N):
        if status[i] == target_status:
            start = i
            break
    else:
        index[0] = 0
        size[0] = 0
        return
    for i in range(start+1, min(start+max_batch, N)):
        if status[i] != target_status:
            end = i
    for i in range(start, end):
        status[i] = new_status
    index[0] = start
    size[0] = end-start




cdef void yield_output(flag_t* status, int size, int layer_id) nogil:
    for i in range(size):
        status[i] = layer_id+1


cdef void yield_gradient(flag_t* status, int size, int layer_id) nogil:
    for i in range(size):
        status[i] = -layer_id
