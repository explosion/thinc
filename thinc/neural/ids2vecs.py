from collections import defaultdict

from .base import Model


class WindowEncode(Model):
    nr_piece = 3
    nr_feat = 5
    vectors = None
    nr_out = None
    nr_in = None
    static = True

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
            self.ops.xavier_uniform_init(self.W, inplace=True)

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

    def begin_update(self, ids, dropout=0.0):
        outputs, whiches = self._forward(ids)
        flat_out = self.ops.flatten(outputs)
        flat_out, bp_dropout = self.ops.dropout(flat_out, dropout,
                                               inplace=True)
        finish_update = self._get_finish_update(ids, flat_out, whiches)
        return flat_out, bp_dropout(finish_update)

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

    def _forward(self, batch):
        positions = self._get_positions(batch)
        dotted = self._dot_ids(positions, [len(seq) for seq in batch])
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
        for i, seq in enumerate(batch):
            for j, id_ in enumerate(seq):
                ids[id_].append((i, j))
        for id_ in ids:
            vector = self.vectors.get(id_)
            if vector is None:
                self.add_param(id_, (self.nr_in,))
        return ids

    def _get_finish_update(self, ids, flat_out, whiches):
        def finish_update(flat_gradients, optimizer=None, **kwargs):
            lengths = [len(w) for w in whiches]
            
            all_inputs = self._get_all_inputs(ids)
            flat_whiches = self.ops.flatten(whiches)
            all_gradients = self._get_all_gradients(flat_gradients,
                                                    flat_whiches)
            if all_inputs.shape[0] == 0 or all_gradients.shape[0] == 0:
                return None
            self.d_b += all_gradients.sum(axis=0)
            # Bop,Bfi->opfi
            self.d_W += self.ops.batch_outer(all_gradients, all_inputs)
            tuned_ids = self._fine_tune(self.W, ids, all_gradients)
            if optimizer is not None:
                for id_ in tuned_ids:
                    optimizer(self.vectors.get(id_), self.get_gradient(id_),
                              key=(self.name, id_))
            return None
        return finish_update

    def _dot_ids(self, ids, lengths):
        out = [self.ops.allocate((length, self.nr_out, self.nr_piece))
                             for length in lengths]
        for id_, egs in ids.items():
            vector = self.vectors.get(id_)
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

    def _fine_tune(self, W, X, all_gradients):
        if self.static:
            return set()
        # Bop,opfi->Bfi
        tuning = self.ops.xp.tensordot(all_gradients, W, axes=[[1,2], [0, 1]])
        tuned_ids = set()
        i = 0
        for ids in X:
            for id_ in ids:
                d_vector = self.get_gradient(id_)
                if d_vector is None:
                    i += 1
                    continue
                if i >= 2:
                    d_vector += tuning[i-2, 0]
                if i >= 1:
                    d_vector += tuning[i-1, 1]
                d_vector += tuning[i, 2]
                if (i+1) < tuning.shape[0]:
                    d_vector += tuning[i+1, 3]
                if (i+2) < tuning.shape[0]:
                    d_vector += tuning[i+2, 4]
                i += 1
                tuned_ids.add(id_)
                assert not self.ops.xp.isnan(d_vector).any()
        return tuned_ids

    def _get_all_inputs(self, X):
        total_length = sum(len(x) for x in X)
        all_inputs = self.ops.allocate((total_length, 5, self.nr_in))
        i = 0
        for ids in X:
            vectors = [self.vectors.get(id_) for id_ in ids]
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

    def _get_all_gradients(self, gradients, flat_whiches):
        all_gradients = self.ops.allocate((len(gradients), self.nr_out,
                                           self.nr_piece))
        for i in range(self.nr_piece):
            all_gradients[:, :, i] += gradients * (flat_whiches == i)
        return all_gradients
