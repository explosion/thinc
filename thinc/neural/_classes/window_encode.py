from collections import defaultdict

from .base import Model


class SpacyWindowEncode(Model):
    nr_piece = 3
    nr_feat = 5
    nr_out = None
    nr_in = None

    @property
    def nr_weight(self):
        nr_W = self.nr_out * self.nr_in
        nr_b = self.nr_out
        return self.nr_feat * self.nr_piece * (nr_W + nr_b)

    def setup(self, *args, **kwargs):
        self.data = None
        self.W = None
        if kwargs.get('W') is not None:
            self.nr_out = kwargs.get('W').shape[0]
            self.nr_piece = kwargs.get('W').shape[1]
            self.nr_in = kwargs.get('W').shape[2]
        if self.nr_out is not None and self.nr_in is not None \
        and self.nr_piece is not None:
            self.set_weights(initialize=True)
            self.set_gradient()
        if kwargs.get('W') is not None:
            self.W[:] = kwargs.get('W')
        if kwargs.get('b') is not None:
            self.b[:] = kwargs.get('b')

    def set_weights(self, data=None, initialize=True, example=None):
        if data is None:
            if self.data is None:
                self.data = self.ops.allocate_pool(self.nr_weight,
                                name=(self.name, 'pool'))
            data = self.data
        self.W = data.allocate_shape((self.nr_out, self.nr_piece,
                                      self.nr_feat, self.nr_in))
        self.b = data.allocate_shape((self.nr_out, self.nr_piece))
        if initialize:
            scale = self.ops.xp.sqrt(2. / (self.W.shape[0] + self.W.shape[2] * self.W.shape[3]))
            shape = (self.W.shape[0], self.W.shape[2], self.W.shape[3])
            for i in range(self.nr_piece):
                self.W[:,i] = self.ops.xp.random.uniform(-scale, scale, shape)

    def set_gradient(self, data=None, initialize=False):
        if data is None:
            self.d_data = self.ops.allocate_pool(self.nr_weight,
                            name=(self.name, 'pool'))
        else:
            self.d_data = data
        self.d_W = self.d_data.allocate_shape(self.W.shape)
        self.d_b = self.d_data.allocate_shape(self.b.shape)

    def predict_batch(self, X):
        out, _ = self._forward(X)
        out = self.ops.flatten(out)
        return out

    def begin_update(self, docs, dropout=0.0):
        outputs, whiches = self._forward(docs)
        flat_out = self.ops.flatten(outputs)
        whiches = self.ops.flatten(whiches)
        flat_out, bp_dropout = self.ops.dropout(flat_out, dropout,
                                               inplace=True)
        finish_update = self._get_finish_update(docs, flat_out, whiches)
        return flat_out, bp_dropout(finish_update)

    def add_vector(self, id_, shape, add_gradient=True):
        pass

    def get_vector(self, id_):
        raise NotImplementedError

    def _get_vector(self, word):
        return word.orth, word.vector / (word.vector_norm or 1.)
    
    def get_gradient(self, id_):
        return None

    def average_params(self, optimizer):
        pass

    def _forward(self, batch):
        positions, vectors = self._get_positions(batch)
        dotted = self._dot_ids(positions, vectors, [len(seq) for seq in batch])
        out = [self.ops.allocate((len(x), self.nr_out)) for x in batch]
        whiches = []
        for i, cands in enumerate(dotted):
            cands += self.b
            which = self.ops.argmax(cands)
            best = self.ops.take_which(cands, which)
            out[i][:] = best
            whiches.append(which)
        return out, whiches

    def _get_positions(self, batch):
        ids = defaultdict(list)
        vectors = {}
        for i, doc in enumerate(batch):
            for j, word in enumerate(doc):
                key, vector = self._get_vector(word)
                ids[key].append((i, j))
                vectors[key] = vector
        return ids, vectors

    def _get_finish_update(self, docs, outputs, whiches):
        def finish_update(gradients, optimizer=None, **kwargs):
            all_inputs = self._get_all_inputs(docs)
            all_gradients = self._get_all_gradients(gradients,
                                                    whiches)
            if all_inputs.shape[0] == 0 or all_gradients.shape[0] == 0:
                return None
            self.d_b += all_gradients.sum(axis=0)
            # Bop,Bfi->opfi
            self.d_W += self.ops.batch_outer(all_gradients, all_inputs)
            return None
        return finish_update

    def _dot_ids(self, ids, vectors, lengths):
        out = [self.ops.allocate((length, self.nr_out, self.nr_piece))
                             for length in lengths]
        for id_, egs in ids.items():
            vector = vectors[id_]
            if vector is None:
                continue
            # opFi,i->Fop
            hidden = self.ops.xp.tensordot(self.W, vector, axes=[[3], [0]])
            hidden = hidden.transpose((2, 0, 1))
            for i, j in egs:
                out_i = out[i]
                if j >= 2 and (j+2) < lengths[i]:
                    out_i[j-2:j+3] += hidden
                else:
                    if j >= 2:
                        out_i[j-2] += hidden[0]
                    if j >= 1:
                        out_i[j-1] += hidden[1]
                    out_i[j] += hidden[2]
                    if (j+1) < lengths[i]:
                        out_i[j+1] += hidden[3]
                    if (j+2) < lengths[i]:
                        out_i[j+2] += hidden[4]
        return out

    def _get_all_inputs(self, X):
        total_length = sum(len(x) for x in X)
        all_inputs = self.ops.allocate((total_length, 5, self.nr_in))
        i = 0
        for doc in X:
            vectors = [self._get_vector(word)[1] for word in doc]
            for vector in vectors:
                if i >= 2:
                    all_inputs[i-2, 0] = vector
                if i >= 1:
                    all_inputs[i-1, 1] = vector
                all_inputs[i, 2] = vector
                if (i+1) < total_length:
                    all_inputs[i+1, 3] = vector
                if (i+2) < total_length:
                    all_inputs[i+2, 4] = vector
                i += 1
        return all_inputs

    def _get_all_gradients(self, gradients, whiches):
        all_gradients = self.ops.allocate((len(gradients), self.nr_out,
                                           self.nr_piece))
        for i in range(self.nr_piece):
            all_gradients[:, :, i] += gradients * (whiches == i)
        return all_gradients
