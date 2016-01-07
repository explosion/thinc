from .. cimport funcs


def call_dot_plus(
    float[:] out,
        float[:] in_,
        float[:] W,
        float[:] bias,
        int nr_out,
        int nr_wide):
    funcs.dot_plus(&out[0],
        &in_[0], &W[0], &bias[0], nr_out, nr_wide)
    return out


def call_d_dot(
    float[:] btm_diff,
        float[:] top_diff,
        float[:] W,
        int nr_out,
        int nr_wide):
    funcs.d_dot(&btm_diff[0],
        &top_diff[0], &W[0], nr_out, nr_wide)
    return btm_diff



def call_ELU(float[:] out, int nr_out):
    funcs.ELU(&out[0], nr_out)
    return out


def call_d_ELU(float[:] delta, float[:] signal_out, int nr_out):
    funcs.d_ELU(&delta[0], &signal_out[0], nr_out)
    return delta
