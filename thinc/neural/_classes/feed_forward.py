from .model import Model


class FeedForward(Model):
    '''A feed-forward network, that chains multiple Model instances together.'''
    def __init__(self, layers, **kwargs):
        Model.__init__(self, **kwargs)
        self.layers.extend(layers)
        if self.layers:
            nO = self.layers[0].output_shape[1]
            for layer in self.layers[1:]:
                if nO is not None and layer.nI is None:
                    layer.nI = nO
                nO = layer.nO 

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def begin_update(self, X, drop=0.):
        callbacks = []
        for layer in self.layers:
            assert layer.W is not None
            assert layer.b is not None
            X = self.ops.xp.ascontiguousarray(X, dtype='float32')
            X, inc_layer_grad = layer.begin_update(X, drop=drop)
            callbacks.append(inc_layer_grad)
        def continue_update(gradient, sgd=None):
            for callback in reversed(callbacks):
                gradient = self.ops.xp.ascontiguousarray(gradient, dtype='float32')
                gradient = callback(gradient, sgd)
            return gradient
        return X, continue_update
