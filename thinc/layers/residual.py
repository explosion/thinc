from .base import Model


def forward(model, X, is_train):
    if not is_train:
        predict(model, X)

    y, bp_y = model._layers[0].begin_update(X)
    if isinstance(X, list):
        output = [X[i] + y[i] for i in range(len(X))]
    elif isinstance(X, tuple) and isinstance(y, tuple) and len(X) == 2:
        # Handle case where we have (data, lengths) tuple
        output = (X[0] + y[0], y[1])
    else:
        output = X + y

    def residual_bwd(d_output):
        dX = bp_y(d_output)
        if isinstance(d_output, list) or isinstance(d_output, tuple):
            return [d_output[i] + dX[i] for i in range(len(d_output))]
        else:
            return d_output + dX

    return output, residual_bwd


def init(model, X=None, Y=None):
    model._layers[0].initialize(X=X, Y=Y)
    model.set_dim("nO", model._layers[0].get_dim("nO"))
    model.set_dim("nI", model._layers[0].get_dim("nI"))


def make_Residual(layer):
    return Model(
        forward,
        init=init,
        layers=[layer]
        params={},
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
        attrs={}
    )
