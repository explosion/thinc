from collections import defaultdict

from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ... import check
from ...check import is_sequence


def _set_dimensions_if_given(model, *args, **kwargs):
    if len(args) >= 1:
        model.nO = args[0]
    if 'pieces' in kwargs:
        model.nP = kwargs['pieces']
    if 'window' in kwargs:
        model.nF = kwargs['window'] * 2 + 1


def _set_dimensions_if_needed(model, X, y=None):
    if model.nO == None and y is not None:
        model.nO = y.max() + 1


@describe.input(("nB", "nI"))
@describe.output(("nB", "nO"))
@describe.on_data(_set_dimensions_if_needed)
@describe.on_init(_set_dimensions_if_given)
@describe.attributes(
    nP=Dimension("Number of pieces"),
    nF=Dimension("Number of features"),
    nO=Dimension("Size of output"),
    W=Synapses("Weights matrix", lambda obj: (obj.nO, obj.nP, obj.nF, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W)),
    b=Biases("Bias vector", lambda obj: (obj.nO, obj.nP)),
    d_W=Gradient("W"),
    d_b=Gradient("b")
)
class MaxoutWindowEncode(Model):
    @property
    def nI(self):
        return self.embed.nO

    def __init__(self, embed, *args, **kwargs):
        self.embed = embed
        Model.__init__(self, *args, **kwargs)

    def predict(self, positions):
        uniq_ids = self.ops.asarray(sorted(positions.keys()), dtype='i')
        uniq_vectors = self.embed.predict(uniq_ids)
        out, _ = self._forward(uniq_ids, positions, uniq_vectors)
        return out

    def begin_update(self, positions, drop=0.):
        uniq_ids = self.ops.asarray(sorted(positions.keys()))
        uniq_vectors, fine_tune = self.embed.begin_update(uniq_ids, drop=drop)
        assert uniq_ids.shape[0] == uniq_vectors.shape[0]
        
        best__bo, which__bo = self._forward(uniq_ids, positions, uniq_vectors)
        best__bo, bp_dropout = self.ops.dropout(best__bo, drop, inplace=True)

        def finish_update(gradient__bo, sgd=None):
            gradient__bop = self.ops.backprop_take(gradient__bo, which__bo, self.nP)
            self.d_b += gradient__bop.sum(axis=0)

            inputs__bfi = self.ops.allocate((which__bo.shape[0], self.nF, self.nI))
            _get_full_inputs(inputs__bfi, uniq_ids, positions, uniq_vectors)
            # Bop,Bfi->opfi
            self.d_W += self.ops.batch_outer(gradient__bop, inputs__bfi)
            if fine_tune is not None:
                gradient__bfi = self.ops.xp.einsum(
                    'bop,opfi->bfi', gradient__bop, self.W)
                gradient__ui = _get_vector_gradients(self.ops, uniq_ids, positions,
                                                     gradient__bfi)
                return fine_tune(gradient__ui)
            else:
                return None
        return best__bo, bp_dropout(finish_update)

    def _forward(self, uniq_ids, positions, vectors):
        hidden = _compute_hidden_layer(self.ops, self.W, vectors)
        cands = _get_output(self.ops, uniq_ids, positions, hidden)
        cands += self.b
        which = self.ops.argmax(cands)
        best = self.ops.take_which(cands, which)
        return best, which


def _get_output(ops, uniq_ids, positions, H__ufop):
    nU, nF, nO, nP = H__ufop.shape
    window_size = int((nF-1)/2)
    total_length = sum(len(val) for val in positions.values())
    # Shift the input, so that we don't have to special-case the starts and
    # ends. We'll shift back afterwards.
    out__bop = ops.allocate((total_length+(window_size*2), nO, nP))
    
    for vec_idx, id_ in enumerate(uniq_ids):
        tok_idxs = positions[id_]
        v__fop = H__ufop[vec_idx]
        for i in tok_idxs:
            slice__fop = out__bop[i : i+5]
            slice__fop += v__fop
            out__bop[i : i+5] = slice__fop
    # Shift the output, to correct for the 'padding' shift above.
    out__bop = out__bop[window_size : -window_size]
    return out__bop


def _compute_hidden_layer(ops, W__opfi, vectors__ui): # pragma: no cover
    H__uopf = ops.xp.tensordot(vectors__ui, W__opfi, axes=[[1], [3]])
    H__ufop = H__uopf.transpose((0, 3, 1, 2))
    return H__ufop


def _get_vector_gradients(ops, uniq_ids, positions, gradients__bfi):
    gradients__ui = ops.allocate((len(uniq_ids), gradients__bfi.shape[-1]))
    for u, id_ in enumerate(uniq_ids):
        grad = gradients__ui[u]
        for tok_idx in positions[id_]:
            grad += gradients__bfi[tok_idx].sum(axis=0)
    return gradients__ui



def _get_full_inputs(writeto, uniq_ids, positions, vectors):
    for vec_idx, id_ in enumerate(uniq_ids):
        tok_idxs = positions[id_]
        vector = vectors[vec_idx]
        for tok_idx in tok_idxs:
            writeto[tok_idx, 2] = vector
    # Col 2 has w
    vectors = writeto[:, 2]
    # Col 0 has LL
    writeto[2:, 0] = vectors[:-2]
    # Col 1 has L
    writeto[1:, 1] = vectors[:-1]
    # Col 3 has R
    writeto[:-1, 3] = vectors[1:]
    # Col 4 has RR
    writeto[:-2, 4] = vectors[2:]
    ## Now use the lengths to zero LL, L, R and RR features as appropriate.
    return writeto
