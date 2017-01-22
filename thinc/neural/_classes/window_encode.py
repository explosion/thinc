from collections import defaultdict

from .model import Model
from .._lsuv import do_lsuv
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
    model.nI = model.embed.nO


def _set_dimensions_if_needed(model, X, y=None):
    if model.nO == None and y is not None:
        model.nO = y.max() + 1
 

def LSUVinit(model, positions, y=None):
    ids = []
    for id_, occurs in positions.items():
        ids.extend(id_ for _ in occurs)
    ids = model.ops.asarray(ids, dtype='i')
    for hook in model.embed.on_data_hooks:
        hook(model.embed, ids, y)
    return do_lsuv(model.ops, model.W, model, positions)


@describe.input(("nB", "nI"))
@describe.output(("nB", "nO"))
@describe.on_data(_set_dimensions_if_needed, LSUVinit)
@describe.on_init(_set_dimensions_if_given)
@describe.attributes(
    nP=Dimension("Number of pieces"),
    nF=Dimension("Number of features"),
    nO=Dimension("Size of output"),
    nI=Dimension("Size of input"),
    W=Synapses("Weights matrix", lambda obj: (obj.nO, obj.nP, obj.nF, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W)),
    b=Biases("Bias vector", lambda obj: (obj.nO, obj.nP)),
    d_W=Gradient("W"),
    d_b=Gradient("b")
)
class MaxoutWindowEncode(Model):
    name = 'window-encode'
    @property
    def nW(self):
        return int((self.nF-1)/2)

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
            gradient__bop = self.ops.backprop_maxout(gradient__bo, which__bo, self.nP)
            self.d_b += gradient__bop.sum(axis=0)
            inputs__bfi = _get_full_inputs(
                self.ops, uniq_ids, positions, uniq_vectors, self.nW)
            if fine_tune is not None:
                gradient__bfi = self.ops.xp.tensordot(gradient__bop, self.W,
                    axes=[[1,2], [0,1]])
                gradient__ui = _get_vector_gradients(self.ops, uniq_ids, positions,
                                                     gradient__bfi)
                grad_out = fine_tune(gradient__ui)
            else:
                grad_out = None
            # Bop,Bfi->opfi
            self.d_W += self.ops.xp.tensordot(gradient__bop, inputs__bfi, axes=[[0], [0]])
            if sgd is not None: # pragma: no cover
                sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
            return grad_out
        return best__bo, bp_dropout(finish_update)

    def _forward(self, uniq_ids, positions, vectors):
        assert self.nP != 0
        hidden = _compute_hidden_layer(self.ops, self.W, vectors)
        cands = _get_output(self.ops, uniq_ids, positions, hidden)
        cands += self.b
        return self.ops.maxout(cands)


def _get_output(ops, uniq_ids, positions, H__ufop):
    nU, nF, nO, nP = H__ufop.shape
    nW = int((nF-1)/2)
    total_length = sum(len(val) for val in positions.values())
    # Shift the input, so that we don't have to special-case the starts and
    # ends. We'll shift back afterwards.
    out__bop = ops.allocate((total_length+(nW*2), nO, nP))
    # Let's say we have the word 'of'
    # Its vector is at vec_idx=1. It occurred at [1, 10, 12, 17] in our data.
    # Our data has 17 tokens.
    for vec_idx, id_ in enumerate(uniq_ids):
        tok_idxs = positions[id_] # [1, 10, 12, 17]
        # v__fop holds the feature weights for 'of' in 5 rows:
        # 0: _ _ _ _ of
        # 1: _ _ _ of _
        # 2: _ _ of _ _
        # 3: _ of _ _ _
        # 4: of _ _ _ _
        # The weights at row 0 are the ones that apply for a word two before 'of',
        # i.e. when 'of' is in the RR position. The weights at row 4 are the ones
        # that apply when 'of' is two *after* the focus word. We can therefore
        # add these weights to a slice of the output.
        v__fop = H__ufop[vec_idx]
        ops.increment_slices(out__bop, v__fop, tok_idxs)
    # Shift the output, to correct for the 'padding' shift above.
    out__bop = out__bop[nW : -nW]
    return out__bop


def _compute_hidden_layer(ops, W__opfi, vectors__ui):
    H__uopf = ops.xp.tensordot(vectors__ui, W__opfi, axes=[[1], [3]])
    H__ufop = H__uopf.transpose((0, 3, 1, 2))
    return H__ufop


def _get_vector_gradients(ops, uniq_ids, positions, tune__bfi):
    nW = int((tune__bfi.shape[1]-1) / 2)
    gradients__ui = ops.allocate((len(uniq_ids), tune__bfi.shape[-1]))
    for u, id_ in enumerate(uniq_ids):
        d_vector = gradients__ui[u]
        for i in positions[id_]:
            # Let's say the input was
            # "the heart of the matter"
            # What's the gradient for the vector of "of"?
            # It was used 5 times --- as a feature for each word
            for f in range(nW):
                if i-(nW-f) >= 0:
                    d_vector += tune__bfi[i-(nW-f), f]
            for f in range(nW+1):
                if (i+f) < tune__bfi.shape[0]:
                    d_vector += tune__bfi[i+f, nW+f]
    return gradients__ui


def _get_full_inputs(ops, uniq_ids, positions, vectors__ui, nW):
    total_length = sum(len(occurs) for occurs in positions.values())
    vectors__bi = ops.allocate((total_length, vectors__ui.shape[1]))
    # Get the (non-unique) vectors, as a contiguous array.
    for vec_idx, id_ in enumerate(uniq_ids):
        tok_idxs = positions[id_]
        vector = vectors__ui[vec_idx]
        for tok_idx in tok_idxs:
            vectors__bi[tok_idx] = vector
    # Let's say the input was
    # "the heart of the matter"
    # If "of" is RR, we read of[0]
    # If "of" is R, we read of[1]
    # If "of" is W, we read of[2]
    # If "of" is L, we read of[3]
    # If "of" is LL, we read of[4]
    X__bfi = ops.allocate((total_length, nW*2+1, vectors__ui.shape[1]))
    for f in range(1, nW+1):
        X__bfi[:-f, nW-f] = vectors__bi[f:] # Words at the start aren't R features
    X__bfi[:, nW] = vectors__bi
    for f in range(1, nW+1):
        X__bfi[f:, nW+f] = vectors__bi[:-f] # Words at the end aren't L features
    return X__bfi
