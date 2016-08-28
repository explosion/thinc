from ..solve cimport *
from ..forward cimport *
from ..backward cimport *
import numpy


def call_dot_plus(
    weight_t[:] out,
        weight_t[:] in_,
        weight_t[:] W,
        weight_t[:] bias,
        int nr_top,
        int nr_btm):
    affine(&out[0],
        &in_[0], &W[0], &bias[0], nr_top, nr_btm, 1)
    return out


def call_d_dot(
    weight_t[:] d_x,
    weight_t[:] d_w,
    weight_t[:] d_b,
        weight_t[:] d_out,
        weight_t[:] x,
        weight_t[:] w,
        int nr_top,
        int nr_btm,
        int nr_batch):
    d_affine(&d_x[0], &d_w[0], &d_b[0],
        &d_out[0], &x[0], &w[0], nr_top, nr_btm, 1)
    return d_x


def call_ELU(weight_t[:] out, int nr_out):
    ELU(&out[0], nr_out, 1)
    return out


def call_d_ELU(weight_t[:] delta, weight_t[:] signal_out, int nr_out):
    d_ELU(&delta[0], &signal_out[0], nr_out)
    return delta

#
#def call_normalize(weight_t[:, :] data, int nr_batch, int n):
#    assert nr_batch != 1 # Fix NULL calls to normalize to use with minibatch 1
#    cdef weight_t[:] flattened = numpy.ascontiguousarray(data).flatten()
#    normalize(&flattened[0], NULL, NULL, nr_batch, n)
#    return numpy.ascontiguousarray(flattened).reshape((nr_batch, n))
