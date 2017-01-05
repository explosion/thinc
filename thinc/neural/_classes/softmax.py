from .affine import Affine


class Softmax(Affine):
    name = 'softmax'
    
    @property
    def describe_params(self):
        yield 'W-%s' % self.name, (self.nr_out, self.nr_in), None
        yield 'b-%s' % self.name, (self.nr_out,), None
    
    def predict_batch(self, X):
        output = Affine.predict_batch(self, X)
        act = self.activate(output)
        return act

    def begin_update(self, X, dropout=0.0, **kwargs):
        return Affine.begin_update(self, X, dropout=0.0)

    def activate(self, X):
        return self.ops.softmax(X, axis=-1)


