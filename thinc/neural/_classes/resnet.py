# coding: utf8
from __future__ import unicode_literals

from .model import Model


class Residual(Model):
    def __init__(self, layer):
        Model.__init__(self)
        self._layers.append(layer)
        self.on_data_hooks.append(on_data)

    def __call__(self, X):
        Y = self._layers[0](X)
        if isinstance(X, list) or isinstance(X, tuple):
            return [X[i] + Y[i] for i in range(len(X))]
        else:
            return X + Y

    def begin_update(self, X, drop=0.0):
        y, bp_y = self._layers[0].begin_update(X, drop=drop)
        if isinstance(X, list) or isinstance(X, tuple):
            output = [X[i] + y[i] for i in range(len(X))]
        else:
            output = X + y

        def residual_bwd(d_output, sgd=None):
            dX = bp_y(d_output, sgd)
            if isinstance(d_output, list) or isinstance(d_output, tuple):
                return [d_output[i] + dX[i] for i in range(len(d_output))]
            else:
                return d_output + dX

        return output, residual_bwd


def on_data(self, X, y=None):
    for layer in self._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, y)
        if hasattr(layer, "W"):
            layer.W.fill(0)
