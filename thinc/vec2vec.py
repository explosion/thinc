from .base import Model
from .exceptions import ShapeError


class Affine(Model):
    name = 'affine'
    nr_out = None
    nr_in = None

    @property
    def nr_weight(self):
        return (self.nr_out * self.nr_in) + self.nr_out

    def set_weights(self, data=None, initialize=True):
        if data is None:
            self.data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'pool'))
        else:
            self.data = data
        self.W = self.data.allocate_shape((self.nr_out, self.nr_in))
        self.b = self.data.allocate_shape((self.nr_out,))
        if initialize:
            self.ops.xavier_uniform_init(self.W, inplace=True)
            assert self.W.flatten().sum() != 0.0

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
            if optimizer is not None:
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

    def set_weights(self, data=None, initialize=True):
        Affine.set_weights(self, data=data, initialize=False)

    def predict_batch(self, input_bi):
        output_bo = Affine.predict_batch(self, input_bi)
        return self.ops.softmax(output_bo, axis=-1)

    def begin_update(self, input_BI, dropout=0.0):
        return Affine.begin_update(self, input_BI, dropout=0.0)


class Maxout(Affine):
    name = 'maxout'
    def predict_batch(self, input_bi):
        take_which = self.ops.take_which
        argmax = self.ops.argmax
        affine = self.ops.affine
        return take_which(argmax(affine(input_bi, self.W, self.b)))

    def begin_update(self, input_BI, dropout=0.0):
        W_OCI = self.W
        b_OC = self.b
        output_BOC = self.ops.affine(W_OCI, b_OC, input_BI)
        which_BO = self.ops.argmax(output_BOC, axis=-1)
        best_BO = self.ops.take_which(output_BOC, which_BO)
        mask_BO = self.ops.get_dropout(best_BO.shape, dropout)
        finish_update = self._get_finish_update(input_BI, which_BO, mask_BO)
        if mask_BO is not None:
            best_BO *= mask_BO
        return best_BO, finish_update

    def _get_finish_update(self, acts_BI, which_BO, mask_BO):
        def finish_update(d_acts_BO):
            raise NotImplementedError
            #d_acts_BO *= mask_BO
            ## TODO
            #self.d_b += d_acts_BO.sum(axis=0)
            #self.d_W += d_W_OCI
            #return d_acts_BI
        return backward
