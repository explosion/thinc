from ..solve cimport *
from ..forward cimport *
from ..backward cimport *


def call_dot_plus(
    float[:] out,
        float[:] in_,
        float[:] W,
        float[:] bias,
        int nr_top,
        int nr_btm):
    dot_plus(&out[0],
        &bias[0], nr_top, &in_[0], nr_btm, &W[0])
    return out


def call_d_dot(
    float[:] btm_diff,
        float[:] top_diff,
        float[:] W,
        int nr_top,
        int nr_btm):
    d_dot(&btm_diff[0],
        nr_btm, &top_diff[0], nr_top, &W[0])
    return btm_diff


def call_ELU(float[:] out, int nr_out):
    ELU(&out[0], nr_out)
    return out


def call_d_ELU(float[:] delta, float[:] signal_out, int nr_out):
    d_ELU(&delta[0], &signal_out[0], nr_out)
    return delta
