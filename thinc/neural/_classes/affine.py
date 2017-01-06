from .model import Model
from ..exceptions import ShapeError


class Affine(Model):
    '''Computes the linear transform Y = (W @ X) + b.

    See also: ReLu, Softmax, Maxout
    '''
    name = 'affine'
    
    @property
    def describe_params(self):
        '''
        Yields (name, shape, initializer) triples describing the weights directly
        owned by the layer.
        '''
        yield 'W-%s' % self.name, (self.nr_out, self.nr_in), self.ops.xavier_uniform_init
        yield 'b-%s' % self.name, (self.nr_out,), None

    @property
    def shape(self):
        if self.output_shape is None or self.input_shape is None:
            return None
        else:
            return (self.nr_out, self.nr_in)

    @property
    def output_shape(self):
        return (self.nr_out,) if self.nr_out is not None else None
    @output_shape.setter
    def output_shape(self, value):
        self.nr_out = value[0]

    @property
    def input_shape(self):
        return (self.nr_in,) if self.nr_in is not None else None
    @input_shape.setter
    def input_shape(self, value):
        self.nr_in = value[0]

    @property
    def W(self):
        return self.params.get('W-%s' % self.name, require=True)

    @property
    def b(self):
        return self.params.get('b-%s' % self.name, require=True)

    @property
    def d_W(self):
        return self.params.get('d_W-%s' % self.name, require=True)

    @property
    def d_b(self):
        return self.params.get('d_b-%s' % self.name, require=True)

    def __init__(self, nr_out=None, nr_in=None, *args, **kwargs):
        '''Construct an Affine object.

        Arguments:
            nr_out (int): Width of output vector.
            nr_in (int): Width of input vector.
        Keyword arguments:
            ops (Ops): Handler for mathematical operations. 
            device (str):
                Specify the device to execute the layer on,
                e.g. 'cpu', 'gpu0', 'gpu1'.
            Trainer (type): Class used by begin_training().
        '''
        self.nr_out = nr_out
        self.nr_in = nr_in
        # This sets attributes from kwargs.
        # args is passed for potential subclasses.
        Model.__init__(self, *args, **kwargs)

    def predict_batch(self, input_BI):
        return self.ops.affine(self.W, self.b, input_BI)

    def begin_update(self, input_BI, dropout=0.0):
        self.check_input(input_BI)
        output_BO = self.predict_batch(input_BI)
        if dropout != 0.0:
            output_BO, bp_dropout = self.ops.dropout(output_BO, dropout)
            return output_BO, bp_dropout(self._get_finish_update(input_BI))
        else:
            return output_BO, self._get_finish_update(input_BI)

    def _get_finish_update(self, acts_BI):
        def finish_update(d_acts_BO, optimizer=None, **kwargs):
            d_b = self.d_b
            d_W = self.d_W
            d_b += d_acts_BO.sum(axis=0)
            d_W += self.ops.batch_outer(d_acts_BO, acts_BI)
            if optimizer is not None and not kwargs.get('is_child'):
                optimizer(self.params.weights, self.params.gradient,
                    key=('', self.name), **kwargs)
            return self.ops.batch_dot(d_acts_BO, self.W.T)
        return finish_update
