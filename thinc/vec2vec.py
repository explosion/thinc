from .base import Model
from .exceptions import ShapeError


class Affine(Model):
    name = 'affine'
    nr_out = None
    nr_in = None
    data = None

    @property
    def is_initialized(self):
        return self.W is not None

    @property
    def input_shape(self):
        return (self.nr_in,)

    @property
    def output_shape(self):
        return (self.nr_out,)

    @property
    def nr_weight(self):
        return (self.nr_out * self.nr_in) + self.nr_out

    def setup(self, *args, **kwargs):
        self.W = None
        if 'W' in kwargs:
            self.nr_out = kwargs.get('W').shape[0]
            self.nr_in = kwargs.get('W').shape[1]
        if self.nr_out is not None and self.nr_in is not None:
            self.set_weights(initialize=True)
            self.set_gradient()
        if 'W' in kwargs:
            self.W[:] = kwargs.get('W')
        if 'b' in kwargs:
            self.b[:] = kwargs.get('b')

    def set_weights(self, data=None, initialize=True, example=None):
        if example is not None:
            self.nr_in = example.shape[-1]
        if data is None:
            if self.data is None:
                self.data = self.ops.allocate_pool(self.nr_weight,
                                name=(self.name, 'pool'))
            data = self.data
        self.W = data.allocate_shape((self.nr_out, self.nr_in))
        self.b = data.allocate_shape((self.nr_out,))
        if initialize:
            self.ops.xavier_uniform_init(self.W, inplace=True)

    def set_gradient(self, data=None, initialize=False):
        if data is None:
            self.d_data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'pool'))
        else:
            self.d_data = data
        self.d_W = self.d_data.allocate_shape((self.nr_out, self.nr_in))
        self.d_b = self.d_data.allocate_shape((self.nr_out,))

    def predict_batch(self, input_BI):
        return self.ops.affine(self.W, self.b, input_BI)

    def begin_update(self, input_BI, dropout=0.0):
        self.check_shape(input_BI, True)
        output_BO = self.predict_batch(input_BI)
        output_BO, bp_dropout = self.ops.dropout(output_BO, dropout)
        return output_BO, bp_dropout(self._get_finish_update(input_BI))
   
    def _get_finish_update(self, acts_BI):
        def finish_update(d_acts_BO, optimizer=None, **kwargs):
            self.d_b += d_acts_BO.sum(axis=0)
            self.d_W += self.ops.batch_outer(d_acts_BO, acts_BI)
            if optimizer is not None and self.data is not None:
                optimizer(self.W, self.d_W, key=('W', self.name), **kwargs)
                optimizer(self.b, self.d_b, key=('b', self.name), **kwargs)
            return self.ops.batch_dot(d_acts_BO, self.W.T)
        return finish_update


class ReLu(Affine):
    name = 'relu'
    def predict_batch(self, input_BI):
        output_BO = Affine.predict_batch(self, input_BI)
        return self.ops.xp.maximum(output_BO, 0., out=output_BO)

    def begin_update(self, input_BI, dropout=0.0):
        output_BO, finish_affine = Affine.begin_update(self, input_BI)
        def finish_update(gradient, *args, **kwargs):
            return finish_affine(gradient * (output_BO > 0), *args, **kwargs)
        return output_BO, finish_update


class Softmax(Affine):
    name = 'softmax'

    def set_weights(self, example=None, data=None, initialize=True):
        Affine.set_weights(self, example=example, data=data, initialize=False)

    def predict_batch(self, input_bi):
        output_bo = Affine.predict_batch(self, input_bi)
        return self.ops.softmax(output_bo, axis=-1)

    def begin_update(self, input_BI, dropout=0.0):
        return Affine.begin_update(self, input_BI, dropout=0.0)


class Maxout(Affine):
    name = 'maxout'
    nr_piece = 3

    @property
    def nr_weight(self):
        return self.nr_piece * ((self.nr_out * self.nr_in) + self.nr_out)

    def set_weights(self, data=None, initialize=True, example=None):
        if example is not None:
            self.nr_in = example.shape[-1]
        if data is None:
            if self.data is None:
                self.data = self.ops.allocate_pool(self.nr_weight,
                                name=(self.name, 'pool'))
            data = self.data
        self.W = data.allocate_shape((self.nr_out, self.nr_piece, self.nr_in))
        self.b = data.allocate_shape((self.nr_out, self.nr_piece))
        if initialize:
            for i in range(self.nr_piece):
                self.ops.xavier_uniform_init(self.W[:, i], inplace=True)

    def set_gradient(self, data=None, initialize=False):
        if data is None:
            self.d_data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'pool'))
        else:
            self.d_data = data
        self.d_W = self.d_data.allocate_shape((self.nr_out, self.nr_piece, self.nr_in))
        self.d_b = self.d_data.allocate_shape((self.nr_out, self.nr_piece))


    def predict_batch(self, input_bi):
        acts_bop = self.ops.xp.tensordot(input_bi, self.W, axes=[[1], [-1]])
        acts_bop += self.b
        which_bo = self.ops.argmax(acts_bop, axis=-1)
        return _take_which(self.ops, acts_bop, which_bo)

    def begin_update(self, input_BI, dropout=0.0):
        W_OCI = self.W
        b_OC = self.b
        output_BOC = self.ops.xp.tensordot(input_BI, W_OCI, axes=[[1], [-1]])
        output_BOC += b_OC
        which_BO = self.ops.argmax(output_BOC, axis=-1)
        best_BO = _take_which(self.ops, output_BOC, which_BO)
        best_BO, bp_dropout = self.ops.dropout(best_BO, dropout, inplace=True)
        finish_update = self._get_finish_update(input_BI, which_BO)
        return best_BO, bp_dropout(finish_update)

    def _get_finish_update(self, acts_BI, which_BO):
        def finish_update(d_acts_BO, optimizer=None, **kwargs):
            d_acts_BOP = self.ops.allocate((d_acts_BO.shape[0], self.nr_out,
                                           self.nr_piece))
            for i in range(self.nr_piece):
                d_acts_BOP[:, :, i] += d_acts_BO * (which_BO == i)
            self.d_b += d_acts_BOP.sum(axis=0)
            self.d_W += self.ops.xp.tensordot(d_acts_BOP, acts_BI, axes=[[0], [0]])
            # Bop,opi->Bi
            d_acts_BI = self.ops.xp.tensordot(d_acts_BOP, self.W, axes=[[1,2], [0, 1]])
            return d_acts_BI
        return finish_update


def _take_which(ops, x, which, axis=-1):
    output = ops.allocate(which.shape)
    for i in range(x.shape[axis]):
        output += x[:, :, i] * (which == i)
    return output
 
