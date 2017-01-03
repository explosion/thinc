from hypothesis import given, assume
from hypothesis.strategies import tuples, lists, integers, floats
from hypothesis.extra.numpy import arrays


def get_ops():
    return NumpyOps()


def get_model(W_values, b_values):
    model = Affine(W_values.shape[0], W_values.shape[1], ops=NumpyOps())
    model.initialize_params()
    model.W[:] = W_values
    model.b[:] = b_values
    return model


def get_output(input_, W_values, b_values):
    return numpy.einsum('oi,bi->bo', W_values, input_) + b_values


def get_input(nr_batch, nr_in):
    ops = NumpyOps()
    return ops.allocate((nr_batch, nr_in))


def lengths(lo=1, hi=10):
    return integers(min_value=lo, max_value=hi)


def shapes(min_rows=1, max_rows=100, min_cols=1, max_cols=100):
    return tuples(
        lengths(lo=min_rows, hi=max_rows),
        lengths(lo=min_cols, hi=max_cols))


def ndarrays_of_shape(shape, lo=-100.0, hi=100.0, dtype='float32'):
    return arrays(
        'float32',
        shape=shape,
        elements=floats(min_value=lo, max_value=hi))
    

def ndarrays(min_len=0, max_len=10, min_val=-10.0, max_val=10.0):
    return lengths(lo=1, hi=2).flatmap(
        lambda n: ndarrays_of_shape(n, lo=min_val, hi=max_val))


def affine_params_and_input(
    min_batch=1,
    max_batch=16,
    min_out=1,
    max_out=16,
    min_in=1,
    max_in=16
):
    shapes = tuples(
        lengths(lo=min_batch, hi=max_batch),
        lengths(lo=min_in, hi=max_out),
        lengths(lo=min_in, hi=max_in))

    def W_b_inputs(shape):
        batch_size, nr_out, nr_in = shape
        W = ndarrays_of_shape((nr_out, nr_in))
        b = ndarrays_of_shape((nr_out,))
        input_ = ndarrays_of_shape((batch_size, nr_in))
        return tuples(W, b, input_)

    return shapes.flatmap(W_b_inputs)
