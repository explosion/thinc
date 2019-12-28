from .base import Model
from ..neural import util


def forward(model, X, is_train):
    Y = model.ops.relu(X)

    def relu_backward(dY):
        return model.ops.backprop_relu(dY, Y)

    return Y, relu_backward


def init(model, X=None, Y=None):
    if X is not None:
        X_width = util.get_width(X)
        model.set_dim("nI", X_width)
        model.set_dim("nO", X_width)
    elif Y is not None:
        Y_width = util.get_width(Y)
        model.set_dim("nI", Y_width)
        model.set_dim("nO", Y_width)


def make_ReLu():
    return Model(
        "relu",
        forward,
        init=init,
        dims={"nO": None, "nI": None},
        params={},
        attrs={},
        layers=[],
    )
