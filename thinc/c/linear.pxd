from .flags cimport flag_t, task_s
from .params cimport params_s


cdef task_s make_task(flag_t* status, int layer_id, params_s* params,
        float* inputs, float* outputs, float* d_inputs, float* d_outputs,
        int nr_out, int nr_in, int batch_size, int N) nogil
