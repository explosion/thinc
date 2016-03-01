from ...extra.eg cimport Example
from ...structs cimport ConstantsC
from ...typedefs cimport len_t
from ..forward cimport dot_plus
from ..forward cimport ELU

import numpy as np


def call_dot_plus__ELU(float[:] top_acts, float[:] btm_acts, float[:] W, float[:] bias):
    eg = Example([btm_acts.shape[0], top_acts.shape[0]])
    eg.set_input(btm_acts)
    #dot_plus__ELU(eg.c.fwd_state, eg.c.bwd_state[0], &W[0], <const len_t*>eg.c.widths,
    #              0, 3, &hp)
    dot_plus(&top_acts[0], &bias[0], bias.shape[0], &btm_acts[0], btm_acts.shape[0], &W[0])
    ELU(&top_acts[0], bias.shape[0])


#def call_d_ELU__(float[:] top_acts, float[:] btm_acts, float[:] W, float[:] bias):
#    cdef const ConstantsC hp
#    d_ELU__dot(gradient,
#        bwd, averages, W, fwd, shape, 3, 3, &hp)
