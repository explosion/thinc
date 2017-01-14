from .model import Model
from .batchnorm import BatchNorm
from ...api import layerize
from .affine import Affine

import cytoolz as toolz


class ResBlock(Model): # pragma: no cover
    name = 'resblock'

    @property
    def input_shape(self):
        return (self.nr_out,)

    @property
    def output_shape(self):
        return (self.nr_out,)

    def __init__(self, width, nr_in=None, **kwargs):
        self.nr_out = width
        self.nr_in = width if nr_in is None else nr_in
        self.name = kwargs.get('name')
        Model.__init__(self, **kwargs)
        self.layers = [
            BatchNorm(name='%s-bn1' % self.name),
            layerize(_relu(self.ops)),
            Affine(width, width, name='%s-weight1' % self.name),
            #BatchNormalization(name='%s-bn2' % self.name),
            #ScaleShift(width, name='%s-scaleshift2' % self.name),
            #layerize(_relu(self.ops)),
            #Affine(width, width, name='%s-weight2' % self.name),
        ]


@toolz.curry
def _relu(ops, X, dropout=0.0): # pragma: no cover
    x_copy = ops.xp.ascontiguousarray(X, dtype='f')
    ops.relu(x_copy, inplace=True)
    mask = X > 0
    def finish_update(gradient, *args, **kwargs):
        gradient = ops.xp.ascontiguousarray(gradient, dtype='f')
        ops.backprop_relu(gradient, x_copy)
        return gradient
    X[:] = x_copy
    return X, finish_update


class Residual(Model): # pragma: no cover
    name = 'residual'

    Block = ResBlock

    @property
    def input_shape(self):
        return (self.nr_out,)

    @property
    def output_shape(self):
        return (self.nr_out,)

    def __init__(self, width, **kwargs):
        self.nr_out = width
        Model.__init__(self, **kwargs)
        if 'name' in kwargs:
            kwargs.pop('name')
        self.layers = [
            self.Block(width, width, name='rb1-%s' % self.name, **kwargs),
            self.Block(width, width, name='rb2-%s' % self.name, **kwargs)
        ]

    def begin_update(self, X, dropout=0.0):
        out1, upd1 = self.layers[0].begin_update(X, dropout=0.)
        out2, upd2 = self.layers[1].begin_update(out1, dropout=0.)
        output, bp_dropout = self.ops.dropout(out1 + out2, dropout)
        def finish_update(gradient):
            grad2 = upd2(gradient)
            grad1 = upd1(grad2)
            return grad1 + grad2
        return output, bp_dropout(finish_update)
