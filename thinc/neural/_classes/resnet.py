from .model import Model
from .batchnorm import BatchNormalization, ScaleShift
from ...api import layerize
from .affine import Affine


class ResBlock(Model):
    name = 'resblock'
    @property
    def input_shape(self):
        return (self.nr_out,)

    @property
    def output_shape(self):
        return (self.nr_out,)

    def __init__(self, width, **kwargs):
        self.nr_out = width
        self.nr_in = width
        layers = [
            BatchNormalization(name='%s-bn1' % self.name),
            ScaleShift(width, name='%s-scaleshift1' % self.name),
            layerize(_relu, name='%s-relu1-func' % self.name),
            Affine(width, width, name='%s-weight1' % self.name),
            BatchNormalization(name='%s-bn2' % self.name),
            ScaleShift(width, name='%s-scaleshift2' % self.name),
            layerize(_relu, name='%s-relu2-func' % self.name),
            Affine(width, width, name='%s-weight2' % self.name),
        ]
        Model.__init__(self, *layers, **kwargs)


def _relu(X, dropout=0.0):
    mask = X > 0
    def finish_update(gradient, *args, **kwargs):
        return gradient * mask
    return X * mask, finish_update


class ReLuResBN(Model):
    name = 'resblock'
    @property
    def input_shape(self):
        return (self.nr_out,)

    @property
    def output_shape(self):
        return (self.nr_out,)

    def __init__(self, width, **kwargs):
        self.nr_out = width
        subkw = dict(kwargs)
        if 'name' in subkw:
            subkw.pop('name')
        layers = [ResBlock(width, name='rb1-%s' % self.name, **subkw),
                  ResBlock(width, name='rb2-%s' % self.name, **subkw)]
        Model.__init__(self, *layers, **kwargs)

    def begin_update(self, X, dropout=0.0):
        out1, upd1 = self.layers[0].begin_update(X, dropout=0.)
        out2, upd2 = self.layers[1].begin_update(out1, dropout=0.)
        output, bp_dropout = self.ops.dropout(out1 + out2, dropout)
        def finish_update(gradient, optimizer=None, **kwargs):
            grad2 = upd2(gradient, optimizer=optimizer, **kwargs)
            grad1 = upd1(grad2, optimizer=optimizer, **kwargs)
            return grad1 + grad2
        return output, bp_dropout(finish_update)
