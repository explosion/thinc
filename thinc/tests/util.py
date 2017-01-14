from ..neural.ops import NumpyOps
from ..neural._classes.affine import Affine

from .strategies import arrays_OI_O_BI

def get_model(W_b_input, cls=Affine):
    ops = NumpyOps()
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = cls(nr_out, nr_in)
    model.W[:] = W
    model.b[:] = b
    return model

def get_shape(W_b_input):
    W, b, input_ = W_b_input
    return input_.shape[0], W.shape[0], W.shape[1]
