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
            gradient__bop = self.ops.backprop_take(gradient__bo, which__bo, self.nP)
            self.d_b += gradient__bop.sum(axis=0)

            inputs__bfi = self.ops.allocate((which__bo.shape[0], self.nF, self.nI))
            _get_full_inputs(inputs__bfi, uniq_ids, positions, uniq_vectors, self.nW)
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
        # Incrememnt each slice of the output, with the word's feature values
        v__fop = H__ufop[vec_idx]
        for i in tok_idxs:
            # Let's say 'of' occurred at position 1 (i==3 given shifting)
            # - output[2] (i.e. of-is-R): += of_weights[1]
            # - output[3] (i.e. of-is-W): += of_weights[2]
            # - output[4] (i.e. of-is-L): += of_weights[3]
            # - output[5] (i.e. of-is-LL): += of_weights[4]
            out__bop[i : i+nF] += v__fop
    # Shift the output, to correct for the 'padding' shift above.
    out__bop = out__bop[nW : -nW]
    return out__bop


def _compute_hidden_layer(ops, W__opfi, vectors__ui):
    H__uopf = ops.xp.tensordot(vectors__ui, W__opfi, axes=[[1], [3]])
    H__ufop = H__uopf.transpose((0, 3, 1, 2))
    return H__ufop


def _get_vector_gradients(ops, uniq_ids, positions, tune__bfi):
    gradients__ui = ops.allocate((len(uniq_ids), tune__bfi.shape[-1]))
    for u, id_ in enumerate(uniq_ids):
        d_vector = gradients__ui[u]
        for i in positions[id_]:
            # Let's say the input was
            # "the heart of the matter"
            # What's the gradient for the vector of "of"?
            # It was used 5 times --- as a feature for each word
            if i >= 2:
                # It was used as feature RR of "the"
                d_vector += tune__bfi[i-2, 0]
            if i >= 1:
                # It was used as feature R of "heart"
                d_vector += tune__bfi[i-1, 1]
            # It was used as feature W of "of"
            d_vector += tune__bfi[i, 2]
            if (i+1) < tune__bfi.shape[0]:
                # It was used as feature L of "the"
                d_vector += tune__bfi[i+1, 3]
            if (i+2) < tune__bfi.shape[0]:
                # It was used as feature LL of "matter"
                d_vector += tune__bfi[i+2, 4]
    return gradients__ui



def _get_full_inputs(writeto, uniq_ids, positions, vectors, nW):
    for vec_idx, id_ in enumerate(uniq_ids):
        tok_idxs = positions[id_]
        vector = vectors[vec_idx]
        for tok_idx in tok_idxs:
            writeto[tok_idx, nW] = vector
    vectors = writeto[:, nW]
    # Let's say the input was
    # "the heart of the matter"
 
    # If "of" is RR, we read of[0]
    # If "of" is R, we read of[1]
    # If "of" is W, we read of[2]
    # If "of" is L, we read of[3]
    # If "of" is LL, we read of[4]
    writeto[2:, 0] = vectors[2:] # Words at the start aren't R features
    writeto[1:, 1] = vectors[1:]
    writeto[:, 2] = vectors
    writeto[:-1, 3] = vectors[:-1] # Words at the end aren't L features
    writeto[:-2, 4] = vectors[:-2]
    # - output[4] (i.e. of-is-L): += of_weights[3]
    # - output[5] (i.e. of-is-LL): += of_weights[4]
    # - output[3] (i.e. of-is-W): += of_weights[2]
    return writeto
