from .model import Model
from ... import describe


class FunctionLayer(Model):
    """Wrap functions into weightless Model instances, for use as network
    components."""

    def __init__(
        self,
        begin_update,
        predict=None,
        predict_one=None,
        nI=None,
        nO=None,
        *args,
        layers=tuple(),
        **kwargs
    ):
        self.begin_update = begin_update
        if predict is not None:
            self.predict = predict
        if predict_one is not None:
            self.predict_one = predict_one
        self.nI = nI
        self.nO = nO
        Model.__init__(self)
        self._layers.extend(layers)
        

def run_child_hooks(self, X, y):
    for child in self._layers:
        for hook in child.on_data_hooks:
            hook(child, X, y)


@describe.on_data(run_child_hooks)
class wrap(Model):
    def __init__(self, begin_update, layer, *, predict=None, predict_one=None):
        self.begin_update = begin_update
        if predict is not None:
            self.predict = predict
        if predict_one is not None:
            self.predict_one = predict_one
        Model.__init__(self)
        self._layers = [layer]

    def has_dim(self, name):
        return self._layers[-1].has_dim(name)

    def get_dim(self, name):
        return self._layers[-1].get_dim(name)

    def set_dim(self, name, value):
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


class ConcatenationLayer(Model):
    name = "concatenate"
    
    def begin_update(self, X, drop=0.):
        Ys, callbacks = zip(*[lyr.begin_update(X, drop=drop) for lyr in layers])
        lengths = [Y.shape[1] for Y in Ys]
        output = self.ops.xp.hstack(Ys)

        def finish_update_concatenate(d_output, sgd=None):
            layer_grad = None
            start = 0
            for bwd, length in zip(callbacks, lengths):
                if bwd is not None:
                    d = bwd(d_output[:, start:start+length], sgd=sgd)
                    if d is not None and hasattr(X, "shape"):
                        if layer_grad is None:
                            layer_grad = d
                        else:
                            layer_grad += d
                start += length
            return layer_grad

        return output, finish_update_concatenate


class AdditionLayer(Model):
    def begin_update(self, X, drop=0.0):
        outs, callbacks = zip(*[lyr.begin_update(X, drop=drop) for lyr in layers])
        out = outs[0]
        for o in outs:
            out += o

        def backward(d_out, sgd=None):
            grads = [bp(d_out, sgd=sgd) for bp in callbacks if bp is not None]
            grads = [g for g in grads if g is not None]
            if grads:
                total = grads[0]
                for g in grads:
                    total += g
                return total
            else:
                return None

        return out, backward
