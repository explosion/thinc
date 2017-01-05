from .base import Model


class Embed(Model):
    name = 'embed'
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
        return self.nr_out * self.nr_in

    def setup(self, vectors, W, **kwargs):
        self.vectors = vectors
        self.W = W

    def set_weights(self, data=None, initialize=True, example=None):
        if example is not None:
            self.nr_in = example.shape[-1]
        if data is None:
            if self.data is None:
                self.data = self.ops.allocate_pool(self.nr_weight,
                                name=(self.name, 'pool'))
            data = self.data
        self.W = data.allocate_shape((self.nr_out, self.nr_in))
        if initialize:
            self.ops.xavier_uniform_init(self.W, inplace=True)

    def set_gradient(self, data=None, initialize=False):
        if data is None:
            self.d_data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'pool'))
        else:
            self.d_data = data
        self.d_W = self.d_data.allocate_shape((self.nr_out, self.nr_in))
        self.gradients = {}

    def add_vector(self, id_, shape, add_gradient=True):
        if not hasattr(self, 'vectors') or self.vectors is None:
            self.vectors = {}
        param = self.ops.allocate(shape)
        param[:] = self.ops.xp.random.uniform(-0.1, 0.1, shape)
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
                self.add_vector(id_, self.input_shape, add_gradient=True)
        return self.predict_batch(ids), self._get_finish_update(ids)

    def _embed(self, ids):
        vectors = self.ops.allocate((len(ids), self.W.shape[1]))
        for i, id_ in enumerate(ids):
            vector = self.get_vector(id_)
            if vector is not None:
                vectors[i] = vector
        return vectors

    def _get_finish_update(self, ids):
        def finish_update(gradients, optimizer=None, **kwargs):
            self.d_W  += self.ops.batch_outer(gradients, self._embed(ids))
            gradients = self.ops.batch_dot(gradients, self.W.T)
            tuned = set()
            for id_, delta_in in zip(ids, gradients):
                embed_grad = self.gradients.get(id_)
                if embed_grad is not None:
                    embed_grad += delta_in
                    tuned.add(id_)
            if optimizer is not None:
                for id_ in tuned:
                    vector = self.get_vector(id_)
                    grad = self.gradients.get(id_)
                    optimizer(vector, grad, key=(self.name, id_))
            return None
        return finish_update
