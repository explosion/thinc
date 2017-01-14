from .model import Model
from ... import describe

def _run_child_hooks(model, X, y):
    for layer in model._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, y)

@describe.on_data(_run_child_hooks)
class FeedForward(Model):
    '''A feed-forward network, that chains multiple Model instances together.'''
    def __init__(self, layers, **kwargs):
        Model.__init__(self, **kwargs)
        self._layers.extend(layers)

    @property
    def input_shape(self):
        return self._layers[0].input_shape

    @property
    def output_shape(self):
        return self._layers[-1].output_shape

    def begin_update(self, X, drop=0.):
        callbacks = []
        for layer in self.layers:
            X = self.ops.xp.ascontiguousarray(X, dtype='float32')
            X, inc_layer_grad = layer.begin_update(X, drop=drop)
            callbacks.append(inc_layer_grad)
        def continue_update(gradient, sgd=None):
            for callback in reversed(callbacks):
                gradient = self.ops.xp.ascontiguousarray(gradient, dtype='float32')
                gradient = callback(gradient, sgd)
            return gradient
        return X, continue_update
