from .model import Model


@declare_dimensions(
    V=("Vector width"),
    N=("Number of vectors"),
    O=("Size of output"),
)
@declare_input(Ints())
@declare_output(shape=Floats("O"))
@declare_weights(
    W=Schema(
        "A projection matrix, to change vector dimensionality",
        shape=("O", "V"),
        initialize=xavier_init,
        static=False
    ),
    vectors=Schema(
        "Embedding table",
        shape=("N", "V"),
        initialize=uniform_init(-0.1, 0.1)
    )
)
class Embed(Model):
    def predict_batch(self, ids):
        if len(ids) < 1000:
            vectors = self._embed(ids)
            return self.ops.batch_dot(vectors, self.w.W)
        id_map = {}
        for i, id_ in enumerate(ids):
            if id_ not in id_map:
                id_map[id_] = [i]
            else:
                id_map[id_].append(i)
        mapped = sorted(id_map.items())
        vectors = self._embed([id_ for id_, _ in mapped])
        result = self.ops.batch_dot(vectors, self.w.W)
        output = self.ops.allocate((len(ids), self.n.O))
        for i, (_, occurs) in enumerate(mapped):
            for j in occurs:
                output[j] = result[i]
        return output

    def begin_update(self, ids):
        def finish_update(gradients):
            self.weights.d.W += self.ops.batch_outer(gradients, self._embed(ids))
            gradients = self.ops.batch_dot(gradients, self.w.W.T)
            for id_, delta_in in zip(ids, gradients):
                self.weights.d.vectors[id_] += delta_in
            return None
        return self.predict(ids), finish_update

    def _embed(self, ids):
        vectors = self.ops.allocate((len(ids), self.n.V))
        for i, id_ in enumerate(ids):
            vectors[i] = self.w.vectors[id_]
        return vectors
