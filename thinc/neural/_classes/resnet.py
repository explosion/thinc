from .model import Model
from ...api import layerize
from .affine import Affine

import cytoolz as toolz


def Residual(layer):
    def residual_fwd(X, drop=0.):
        y, bp_y = layer.begin_update(X, drop=drop)
        output = X+y
        def residual_bwd(d_output, sgd=None):
            return d_output + bp_y(d_output, sgd)
        return output, residual_bwd
    model = layerize(residual_fwd)
    model._layers.append(layer)
    def on_data(self, X, y=None):
        for layer in self._layers:
            for hook in layer.on_data_hooks:
                hook(layer, X, y)
            if hasattr(layer, 'W'):
                layer.W.fill(0)
    model.on_data_hooks.append(on_data)
    return model
