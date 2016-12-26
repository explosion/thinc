from .base import Model
from .ops import NumpyOps


class Embed(Model):
    def __init__(self, vectors=None, W=None, ops=None):
        self.ops = ops
        self.vectors = vectors
        self.W = W

    def predict_batch(self, ids):
        if len(ids) < 1000:
            vectors = self.ops.allocate((len(ids), self.W.shape[1]))
            for i, id_ in enumerate(ids):
                vectors[i] = self.vectors[id_]
            return self.ops.batch_dot(vectors, self.W)
        id_map = {}
        for i, id_ in enumerate(ids):
            if id_ not in id_map:
                id_map[id_] = [i]
            else:
                id_map[id_].append(i)
        mapped = sorted(id_map.items())
        vectors = self.ops.allocate((len(mapped), self.W.shape[1]))
        for i, (id_, _) in enumerate(mapped):
            vectors[i] = self.vectors[id_]
        result = self.ops.batch_dot(vectors, self.W)
        output = self.ops.allocate((len(ids), self.W.shape[0]))
        for i, (_, occurs) in enumerate(mapped):
            for j in occurs:
                output[j] = result[i]
        return output

    def forward(self, ids):
        return self.predict_batch(ids)

    def _get_backward(self, ids):
        def backward(gradients):
            for id_, delta_in in zip(ids, gradients):
                embed_grad = self.gradients.get(id_)
                if embed_grad is not None:
                    embed_grad += delta_in
            return None
        return backward



