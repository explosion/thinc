from .model import Model


class FeedForward(Model):
    '''A feed-forward network, that chains multiple Model instances together.'''
    def __init__(self, *layers, **kwargs):
        Model.__init__(self, **kwargs)
        self.layers.extend(layers)

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def begin_update(self, X):
        callbacks = []
        for layer in self.layers:
            X = self.ops.xp.ascontiguousarray(X, dtype='float64')
            X, inc_layer_grad = layer.begin_update(X)
            callbacks.append(inc_layer_grad)
        def continue_update(gradient):
            for callback in reversed(callbacks):
                gradient = self.ops.xp.ascontiguousarray(gradient, dtype='float64')
                gradient = callback(gradient)
            return gradient
        return X, continue_update
