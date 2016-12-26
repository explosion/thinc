# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True


def take_param(shape, pool):
    nr_weight = reduce(lambda a, b: a*b, shape)
    return pool[:nr_weight].reshape(shape), pool[nr_weight:]


def chain_forward(layers, X):
    for layer in layers:
        X = layer.predict_batch(X)
    return X


def chain_backward(layers, X):
    X = self.do.asarray(X)
    backwards = []
    for layer in layers:
        X, callback = layer.begin_update(X, drop=drop)
        backwards.append(callback)
    def finish_update(gradient):
        for backward in reversed(backwards):
            gradient = backward(gradient)
        return gradient
    return X, finish_update


cdef class Model:
    def pipe(self, stream, batch_size=1000):
        for batch in minibatch(stream, batch_size):
            ys = self(batch)
            for y in ys:
                yield y

    def __call__(self, X):
        raise NotImplementedError

    def update(self, X, drop=0.0):
        raise NotImplementedError


class ReLu(Model):
    def __init__(self, W=None, b=None, ops=None):
        self.ops = ops
        self.W = W
        self.b = b
        self.d_W = self.ops.allocate(self.W.shape)
        self.d_b = self.ops.allocate(self.b.shape)

    def predict_batch(self, input_BI):
        output_BO = self.ops.affine(input_, self.W, self.b)
        return self.ops.clip_low(output_BO, 0, inplace=True)

    def forward(self, input_, drop=0.0):
        output_BO = self.ops.affine(input_, self.W, self.b)
        mask = self.ops.get_dropout(output_BO.shape, drop)
        mask *= output_BO > 0
        output_BO *= mask
        return output, self._get_backward(input_, mask)
    
    def _get_backward(self, acts_BI, mask):
        def backward(d_acts_BO):
            d_acts_BO *= mask
            outer = self.ops.d_affine(d_acts_BO, acts_BI)
            self.d_b += d_acts_BO.sum(axis=0)
            outer = self.ops.tensordot(d_acts_BO, acts_BI, axes=[[0], [0]])
            self.d_W += outer
            d_acts_BI = self.ops.batch_dot(d_acts_BO, self.W.T)
            return d_acts_BI
        return finish_update


class Maxout(Model):
    def __init__(self, W=None, b=None, ops=None):
        self.ops = ops
        self.W = W
        self.b = b
        self.d_W = self.ops.allocate(self.W.shape)
        self.d_b = self.ops.allocate(self.b.shape)

    def predict_batch(self, input_BI):
        W_OCI = self.W
        b_OC = self.b
        output_BOC = self.ops.affine(input_BI, W_OCI, b_OC)
        which_BO = self.ops.argmax(output_BOC, axis=-1)
        best_BO = self.ops.take_which(output_BOC, which_BO)
        return best

    def forward(self, input_BI, drop=0.0):
        W_OCI = self.W
        b_OC = self.b
        output_BOC = self.ops.affine(input_BI, W_OCI, b_OC)
        which_BO = self.ops.argmax(output_BOC, axis=-1)
        best_BO = self.ops.take_which(output_BOC, which_BO)
        mask_BO = self.ops.get_dropout(best_BO.shape, drop)
        finish_update = self._get_backward(input_BI, which_BO, mask_BO)
        best_BO *= mask
        return best_BO, finish_update
    
    def _get_backward(self, acts_BI, which_BO, mask_BO):
        def backward(d_acts_BO):
            d_acts_BO *= mask_BO
            # TODO
            self.d_b += d_acts_BOC.sum(axis=0)
            self.d_W += d_W_OCI
            return d_acts_BI
        return finish_update


class Embed(Model):
    def __init__(self, vectors=None, W=None, ops=None):
        self.ops = ops
        self.vectors = vectors
        self.W = W

    def predict_batch(self, ids):
        vectors = self.allocate((len(ids), self.W.shape[1]))
        for i, id_ in enumerate(batch):
            if id_ in self.vectors:
                vectors[i] = self.vectors[id_]
        return self.ops.batch_dot(vectors, self.W)

    def forward(self, ids):
        return self.predict_batch(ids)

    def _get_backward(self):
        def backward(gradients):
            for x, delta_in in zip(X, gradients):
                for i, id_ in enumerate(x):
                    embed_grad = self.gradients.get(id_)
                    if embed_grad is not None:
                        embed_grad += delta_in[i]
            return None
        return finish_update


class NumpyOps(object):
    def get_dropout(self, shape, drop=0.0):
        pass

    def allocate(self, shape):
        pass

    def asarray(self, data):
        pass

    def batch_dot(self, weights, signal):
        pass

    def affine(self, weights, bias, signal):
        pass

    def d_affine(self):
        pass

    def argmax(self, x):
        pass

    def take_which(self, x, which):
        pass


#    def _initialize_params(self, shapes):
#        self.data = self.do.allocate((self.nr_weight,))
#        self.gradient = self.do.allocate((self.nr_weight,))
#       
#        params_pool = self.data
#        gradients_pool = self.gradients
#        for name, shape in self.shapes:
#            param, params_pool = take_param(shape, pool)
#            gradient, gradients_pool = take_param(shape, gradients_pool)
#            self.params[name] = param
#            self.gradients[name] = gradient
#
#
