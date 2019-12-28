from .base import Model


def forward(model, X):
    nW = model.get_attr("window_size")
    Y = model.ops.seq2col(X, nW)

    def backprop_convolution(dY):
        return model.ops.backprop_seq2col(dY, nW)

    return Y, backprop_convolution


def ExtractWindow(window_size=1):
    return Model(
        "extract_window",
        forward,
        attrs={"window_size": window_size},
        init=None,
        dims={},
        params={},
        layers=[],
    )
