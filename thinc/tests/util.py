from thinc.layers.affine import Affine


def get_model(W_b_input, cls=Affine):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = cls(nr_out, nr_in)
    model.set_param("W", W)
    model.set_param("b", b)
    return model


def get_shape(W_b_input):
    W, b, input_ = W_b_input
    return input_.shape[0], W.shape[0], W.shape[1]
