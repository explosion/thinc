from .model import Model


class Embed(Model):
    name = 'embed'
    
    @property
    def describe_params(self):
        init = self.ops.xavier_uniform_init
        yield 'W-%s' % self.name, (self.nr_out, self.vector_length), init

    @property
    def shape(self):
        if self.output_shape is None or self.input_shape is None:
            return None
        else:
            return (self.nr_out, self.vector_length)

    @property
    def output_shape(self):
        return (self.nr_out,) if self.nr_out is not None else None

    @property
    def input_shape(self):
        return (self.vector_length,) if self.vector_length is not None else None

    @property
    def W(self):
        return self.params.get('W-%s' % self.name, require=True)

    @property
    def d_W(self):
        return self.params.get('d_W-%s' % self.name, require=True)

    def __init__(self, nr_out, vector_length, vectors=None, **kwargs):
        self.nr_out = nr_out
        self.vector_length = vector_length
        self.vectors = vectors if vectors is not None else {}
        Model.__init__(self, **kwargs)

    def check_input(self, X):
        return True

    def add_vector(self, id_, vector_dim, add_gradient=True):
        if not hasattr(self, 'vectors') or self.vectors is None:
            self.vectors = {}
        param = self.ops.allocate(vector_dim)
        param[:] = self.ops.xp.random.uniform(-0.1, 0.1, vector_dim)
        self.vectors[id_] = param
        if add_gradient:
            if not hasattr(self, 'gradients'):
                self.gradients = {}
            self.gradients[id_] = self.ops.allocate(param.shape)

    def get_vector(self, id_):
        return self.vectors.get(id_)

    def get_gradient(self, id_):
        return self.gradients.get(id_)

    def predict_batch(self, ids):
        if len(ids) < 1000:
            vectors = self._embed(ids)
            return self.ops.batch_dot(vectors, self.W)
        id_map = {}
        for i, id_ in enumerate(ids):
            if id_ not in id_map:
                id_map[id_] = [i]
            else:
                id_map[id_].append(i)
        mapped = sorted(id_map.items())
        vectors = self._embed([id_ for id_, _ in mapped])
        result = self.ops.batch_dot(vectors, self.W)
        output = self.ops.allocate((len(ids), self.W.shape[0]))
        for i, (_, occurs) in enumerate(mapped):
            for j in occurs:
                output[j] = result[i]
        return output

    def begin_update(self, ids, dropout=0.0):
        for id_ in ids:
            vector = self.get_vector(id_)
            if vector is None:
                self.add_vector(id_, self.input_shape[0], add_gradient=True)
        return self.predict_batch(ids), self._get_finish_update(ids)

    def check_input(self, x, expect_batch=False):
        return True

    def _embed(self, ids):
        vectors = self.ops.allocate((len(ids), self.W.shape[1]))
        for i, id_ in enumerate(ids):
            vector = self.get_vector(id_)
            if vector is not None:
                vectors[i] = vector
        return vectors

    def _get_finish_update(self, ids):
        def finish_update(gradients, optimizer=None, **kwargs):
            d_W = self.d_W
            d_W += self.ops.batch_outer(gradients, self._embed(ids))
            gradients = self.ops.batch_dot(gradients, self.W.T)
            tuned = set()
            for id_, delta_in in zip(ids, gradients):
                embed_grad = self.gradients.get(id_)
                if embed_grad is not None:
                    embed_grad += delta_in
                    tuned.add(id_)
            if optimizer is not None:
                if not kwargs.get('is_child'):
                    optimizer(self.params.weights, self.params.gradient)
                for id_ in tuned:
                    vector = self.get_vector(id_)
                    grad = self.gradients.get(id_)
                    optimizer(vector, grad, key=(self.name, id_))
            return None
        return finish_update



#    nr_out = None
#    nr_in = None
#    data = None
#
#    @property
#    def is_initialized(self):
#        return self.W is not None
#
#    @property
#    def input_shape(self):
#        return (self.nr_in,)
#
#    @property
#    def output_shape(self):
#        return (self.nr_out,)
#
#    @property
#    def nr_weight(self):
#        return self.nr_out * self.nr_in
#
#    def setup(self, vectors, W, **kwargs):
#        self.vectors = vectors
#        self.W = W
#
#    def set_weights(self, data=None, initialize=True, example=None):
#        if example is not None:
#            self.nr_in = example.shape[-1]
#        if data is None:
#            if self.data is None:
#                self.data = self.ops.allocate_pool(self.nr_weight,
#                                name=(self.name, 'pool'))
#            data = self.data
#        self.W = data.allocate_shape((self.nr_out, self.nr_in))
#        if initialize:
#            self.ops.xavier_uniform_init(self.W, inplace=True)
#
#    def set_gradient(self, data=None, initialize=False):
#        if data is None:
#            self.d_data = self.ops.allocate_pool(self.nr_weight,
#                            name=(self.name, 'pool'))
#        else:
#            self.d_data = data
#        self.d_W = self.d_data.allocate_shape((self.nr_out, self.nr_in))
#        self.gradients = {}
#
#
