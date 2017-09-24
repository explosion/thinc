from .model import Model
from ...api import layerize
from .affine import Affine

import cytoolz as toolz


class Residual(Model):
    def __init__(self, layer):
        Model.__init__(self)
        self._layers.append(layer)
        self.on_data_hooks.append(on_data)

    def __call__(self, X):
        return X + self._layers[0](X)

    def begin_update(self, X, drop=0.):
        y, bp_y = self._layers[0].begin_update(X, drop=drop)
        output = X+y
        def residual_bwd(d_output, sgd=None):
            return d_output + bp_y(d_output, sgd)
        return output, residual_bwd

def on_data(self, X, y=None):
    for layer in self._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, y)
        if hasattr(layer, 'W'):
            layer.W.fill(0)
