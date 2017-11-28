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


cdef int tasks_remaining(const flag_t* status, int layer_id, int N) nogil:
    '''Count how many instances haven't been processed by the layer.'''
    cdef int count = 0
    for flag in status[:N]:
        if 0 <= flag < layer_id:
            count += 1
    return count


cdef void await_input(int* index, int* size, flag_t* status,
        int layer_id, int max_batch, int N) nogil:
    size[0] = 0
    index[0] = 0
    target = layer_id-1
    # Try to get the gradient, sleeping for 100 microseconds at first, and
    # then longer -- until giving up after 1 second.
    cdef int sleep_factor, i, j
    for sleep_factor in range(2, 6):
        for i in range(N):
            if status[i] == target:
                status[i] = layer_id
                index[0] = i
                size[0] = 1
                for j in range(j, min(i+max_batch, N)):
                    if status[j] == target:
                        size[0] += 1
                        status[j] = layer_id
                    else:
                        break
                break
        else:
            continue
        break


cdef void await_gradient(flag_t* status, int size, int layer_id) nogil:
    target = -(layer_id+1)
    # Try to get the gradient, sleeping for 100 microseconds at first, and
    # then longer -- until giving up after 1 second.
    for sleep_factor in range(2, 6):
        for i in range(size):
            if status[i] < target:
                #usleep(10**sleep_factor)
                break
        else:
            for i in range(size):
                status[i] = -layer_id
            return
        return


cdef void yield_output(flag_t* status, int size, int layer_id) nogil:
    for i in range(size):
        status[i] = layer_id+1


cdef void yield_gradient(flag_t* status, int size, int layer_id) nogil:
    for i in range(size):
        status[i] = -layer_id
