from flags cimport flag_t, task_s
from params cimport params_s


cdef task_s make_task(params_s* params, float learn_rate) nogil
