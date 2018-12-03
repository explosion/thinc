# coding: utf8
from __future__ import unicode_literals

from .model import Model
from ... import describe


def _run_child_hooks(model, X, y):
    for layer in model._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, y)
        X = layer(X)


@describe.on_data(_run_child_hooks)
class FeedForward(Model):
    """A feed-forward network, that chains multiple Model instances together."""

    name = "feed-forward"

    def __init__(self, layers, **kwargs):
        self._layers = []
        for layer in layers:
            if isinstance(layer, FeedForward):
                self._layers.extend(layer._layers)
            else:
                self._layers.append(layer)
        Model.__init__(self, **kwargs)

    @property
    def input_shape(self):
        return self._layers[0].input_shape

    @property
    def output_shape(self):
        return self._layers[-1].output_shape

    def predict(self, X):
        for layer in self._layers:
            X = layer(X)
        return X

    def begin_update(self, X, drop=0.0):
        callbacks = []
        for layer in self._layers:
            X, inc_layer_grad = layer.begin_update(X, drop=drop)
            callbacks.append(inc_layer_grad)

        def continue_update(gradient, sgd=None):
            for callback in reversed(callbacks):
                if gradient is None or callback is None:
                    break
                gradient = callback(gradient, sgd)
            return gradient

        return X, continue_update
