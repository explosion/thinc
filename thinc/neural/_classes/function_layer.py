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
