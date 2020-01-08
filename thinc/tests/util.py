import contextlib
from pathlib import Path
import tempfile
import shutil
from thinc.layers import Linear


@contextlib.contextmanager
def make_tempdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))


def get_model(W_b_input, cls=Linear):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = cls(nr_out, nr_in)
    model.set_param("W", W)
    model.set_param("b", b)
    return model


def get_shape(W_b_input):
    W, b, input_ = W_b_input
    return input_.shape[0], W.shape[0], W.shape[1]
