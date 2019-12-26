from .model import Model


class Dropout(Model):
    name = "dropout"

    def __init__(self, rate, factor=1.0):
        Model.__init__(self)
        self.rate = rate
        self.factor = factor
        self.is_enabled = True

    def predict(self, X):
        return X

    def begin_update(self, X):
        rate = self.drop * self.factor
        if not self.is_enabled:
            return X, lambda dY: dY
        elif isinstance(X, tuple) and len(X) == 2:
            Y, wrap_backprop = self.ops.dropout(X[0], rate, inplace=False)
            return (Y, X[1]), wrap_backprop(lambda dY: dY)
        elif isinstance(X, list):
            Y, wrap_backprop = self.ops.dropout_sequences(X, rate, inplace=False)
            return Y, wrap_backprop(lambda dY: dY)
        else:
            Y, wrap_backprop = self.ops.dropout(X, rate, inplace=False)
            return Y, wrap_backprop(lambda dY: dY)
