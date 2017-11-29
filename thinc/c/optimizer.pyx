# cython: infer_types=True
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, memset

from .flags cimport run_task_f


cdef struct args_s:
    params_s* params
    float learn_rate
    int nr_state


cdef task_s make_task(params_s* params, float learn_rate) nogil:
    args = <args_s*>calloc(1, sizeof(args_s))
    args.params = params
    args.learn_rate = learn_rate
    cdef task_s task
    task.run = <run_task_f>run_task
    task.args = args
    return task


cdef void* run_task(args_s* args) nogil:
    params = args.params
    learn_rate = args.learn_rate
    if params.nr_grad_upd > params.nr_weight_upd:
        new_params = <params_s*>calloc(1, sizeof(params_s))
        new_params.weights = <float*>calloc(params.N, sizeof(float))
        new_params.gradient = <float*>calloc(params.N, sizeof(float))
        memcpy(new_params.weights, params.weights, params.N * sizeof(float))
        for i in range(params.N):
            new_params.weights[i] -= learn_rate * params.gradient[i]
        new_params.nr_grad_upd = params.nr_grad_upd
        new_params.nr_weight_upd = params.nr_grad_upd
        params.next = new_params
        new_params.prev = params
        new_params.next = NULL
