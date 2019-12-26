from .model import Model


def _run_child_hooks(model, X=None, Y=None):
    for layer in model._layers[:-1]:
        for hook in layer.on_data_hooks:
            hook(layer, X=X)
        if X is not None:
            X = layer.predict(X)
    for hook in model._layers[-1].on_data_hooks:
        hook(model._layers[-1], X=X, Y=Y)


class FeedForward(Model):
    """A feed-forward network, that chains multiple Model instances together."""

    name = "feed-forward"

    def __init__(self, layers, **kwargs):
        Model.__init__(self, **kwargs)
        self.on_data_hooks = [_run_child_hooks]
        for layer in layers:
            if isinstance(layer, FeedForward):
                self._layers.extend(layer._layers)
            else:
                self._layers.append(layer)

    def infer_dimensions(self, X=None, Y=None):
        if Y is not None:
            self._layers[-1].infer_dimensions(X=None, Y=Y)
        for layer in self._layers:
            layer.infer_dimensions(X=X)
            X = layer.predict(X)

    def has_dim(self, name):
        return self._layers[-1].has_dim(name)

    def get_dim(self, name):
        if name == "nI":
            return self._layers[0].get_dim(name)
        else:
            return self._layers[-1].get_dim(name)

    def set_dim(self, name, value):
        if name == "nI":
            self._layers[0].set_dim(name, value)
        else:
            self._layers[-1].set_dim(name, value)

    def has_param(self, name):
        return name in self._layers[-1]._params

    def get_param(self, name):
        return self._layers[-1].get_param(name)

    def set_param(self, name, value):
        return self._layers[-1].set_param(name, value)

    def has_grad(self, name):
        return self._layers[-1].has_grad(name)

    def get_grad(self, name):
        return self._layers[-1].get_grad(name)

    def set_grad(self, name, value):
        self._layers[-1].set_grad(name, value)

    def predict(self, X):
        for layer in self._layers:
            X = layer.predict(X)
        return X

    def begin_update(self, X, drop=0.0):
        callbacks = []
        for layer in self._layers:
            X, inc_layer_grad = layer.begin_update(X, drop=drop)
            callbacks.append(inc_layer_grad)

        def continue_update(gradient):
            for callback in reversed(callbacks):
                if gradient is None or callback is None:
                    break
                gradient = callback(gradient)
            return gradient

        return X, continue_update
