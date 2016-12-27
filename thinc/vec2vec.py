from .base import Model
from .exceptions import ShapeError


class Affine(Model):
    name = 'affine'
    W = None
    b = None
    nr_out = None
    nr_in = None
    params_data = None

    @property
    def nr_weight(self):
        if self.W is not None:
            return self.W.size + (0 if self.b is None else self.b.size)
        else:
            return (self.nr_out * self.nr_in) + self.nr_out

    def initialize_weights(self, x=None, data=None, is_batch=True):
        if data is None:
            if self.params_data is None:
                self.params_data = self.ops.allocate_pool(self.nr_weight,
                                        name=(self.name, 'pool'))
            data = self.params_data
        if self.W is None:
            self.W = self.ops.allocate_param(data, (self.nr_out, self.nr_in),
                        name=(self.name, 'W'))
            self.ops.xavier_uniform_init(self.W, inplace=True)
        if self.b is None:
            self.b = self.ops.allocate_param(data, (self.nr_out,),
                        name=(self.name, 'b'))

    def predict_batch(self, input_BI):
        self.check_shape(input_BI, True)
        return self.ops.affine(self.W, self.b, input_BI)

    def begin_update(self, input_BI, dropout=0.0):
        self.check_shape(input_BI, True)
        output_BO = self.ops.affine(self.W, self.b, input_BI)
        mask = self.ops.get_dropout(output_BO.shape, dropout)
        if mask is not None:
            output_BO *= mask
        return output_BO, self._get_finish_update(input_BI, mask)
    
    def _get_finish_update(self, acts_BI, mask):
        def finish_update(d_acts_BO, sgd, **kwargs):
            d_b = self.ops.allocate(self.b.shape)
            d_W = self.ops.allocate(self.W.shape)
            if mask is not None:
                d_acts_BO *= mask
            d_b += d_acts_BO.sum(axis=0)
            outer = self.ops.batch_outer(d_acts_BO, acts_BI)
            d_W += outer
            d_acts_BI = self.ops.batch_dot(d_acts_BO, self.W.T)
            assert d_acts_BI.shape == acts_BI.shape
            
            #sgd(self.W, d_W, key=('W', self.name), **kwargs)
            #sgd(self.b, d_b, key=('b', self.name), **kwargs)
            return d_acts_BI
        return finish_update


class ReLu(Affine):
    name = 'relu'
    def predict_batch(self, input_BI):
        dotted = self.ops.xp.tensordot(input_BI, self.W, axes=[[1], [1]])
        dotted += self.b
        return self.ops.clip_low(dotted, 0, inplace=True)

    def begin_update(self, input_BI, dropout=0.0):
        output_BO = self.ops.affine(self.W, self.b, input_BI)
        mask = self.ops.get_dropout(output_BO.shape, dropout)
        if mask is not None:
            mask *= output_BO > 0
            output_BO *= mask
        return output_BO, self._get_finish_update(input_BI, mask)


class Softmax(Affine):
    name = 'softmax'
    def predict_batch(self, input_bi):
        output_bo = self.ops.affine(self.W, self.b, input_bi)
        return self.ops.softmax(output_bo, axis=-1, inplace=True)


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
