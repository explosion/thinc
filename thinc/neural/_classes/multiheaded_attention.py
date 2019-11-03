from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape, layerize, wrap
from ..util import get_array_module
from ..ops import NumpyOps
import numpy
import copy
import math
import blis.py


class PaddedAttentionInputs(object):
    def __init__(self, QKV, lengths):
        self.QKV = QKV
        self.nB, self.nL, three, self.nH, self.nD = QKV.shape
        self.nN = sum(lengths)
        assert three == 3
        self.lengths = lengths
        self.ops = NumpyOps()

    def _get_feature(self, index, dims, contiguous):
        data = self.QKV[:, :, index]
        return self.transpose(data,
            current=("nB", "nL", "nH", "nD"), target=dims, contiguous=contiguous)

    def get_queries(self, dims, contiguous=False):
        return self._get_feature(0, dims, contiguous)

    def get_keys(self, dims, contiguous=False):
        return self._get_feature(1, dims, contiguous)
    
    def get_values(self, dims, contiguous=False):
        return self._get_feature(2, dims, contiguous)

    def transpose(self, array, target, current, contiguous=False):
        assert set(current) == set(target)
        if current == target:
            return array
        array = array.transpose([current.index(d) for d in target])
        if contiguous:
            array = self.ops.xp.ascontiguousarray(array)
        return array

    def get_attn(self, scale=1.):
        queries = self.get_queries(("nB", "nH", "nL", "nD"))
        keys = self.get_keys(("nB", "nH", "nD", "nL"))
        dots = self.ops.xp.matmul(queries, keys)
        dots /= scale
        for i, length in enumerate(self.lengths):
            dots[i, :, :, length:] = -self.ops.xp.inf
        attn = self.ops.softmax(dots, axis=-1)
        for i, length in enumerate(self.lengths):
            attn[i, :, length:] = 0

        def backprop_attn(d_attn):
            assert d_attn.shape == (self.nB, self.nH, self.nL, self.nL), d_attn.shape
            d_dots = d_attn * self.ops.backprop_softmax(attn, d_attn, axis=-1)
            d_dots /= scale
            # (nB, nH, nQ, nK) @ (nB, nH, nD, nK).T = (nB, nH, nQ, nD)
            d_queries = self.ops.xp.matmul(d_dots, keys.transpose((0, 1, 3, 2)))
            # (nB, nH, nQ, nK).T @ (nB, nH, nQ, nD) = (nB, nH, nK, nD)
            d_keys = self.ops.xp.matmul(d_dots.transpose((0, 1, 3, 2)), queries)
            return d_queries, d_keys
        return attn, backprop_attn

    def apply_attn(self, attn):
        values = self.get_values(("nB", "nH", "nL", "nD"), contiguous=True)
        attn = self.ops.xp.ascontiguousarray(attn)
        context = self.ops.xp.matmul(attn, values)
        assert context.shape == (self.nB, self.nH, self.nL, self.nD)
        context = self.transpose(context,
            current=("nB", "nH", "nL", "nD"), target=("nB", "nL", "nH", "nD"))

        def backprop_apply_attn_padded(d_context):
            nonlocal values, attn
            d_context = self.transpose(d_context,
                current=("nB", "nL", "nH", "nD"), target=("nB", "nH", "nL", "nD"))
            values = self.transpose(values,
                current=("nB", "nH", "nV", "nD"), target=("nB", "nH", "nD", "nV"))
            # (nB, nH, nQ, nD) @ (nB, nH, nV, nD).T = (nB, nH, nQ, nV)
            d_attn = self.ops.xp.matmul(d_context, values)
            # (nB, nH, nQ, nV).T @ (nB, nH, nQ, nD) = (nB, nH, nV, nD)
            attn = self.transpose(attn,
                current=("nB", "nH", "nQ", "nV"), target=("nB", "nH", "nV", "nQ"))
            d_values = self.ops.xp.matmul(attn, d_context)
            return d_values, d_attn
        return context, backprop_apply_attn_padded


class AttentionInputs(object):
    """Inputs for an attention model."""
    def __init__(self, QKV, lengths, dims=("nN", "qkv", "nH", "nD")): 
        self.ops = NumpyOps()
        self.nH = QKV.shape[dims.index("nH")]
        self.nN = QKV.shape[dims.index("nN")]
        self.nD = QKV.shape[dims.index("nD")]
        self.nM = self.nH*self.nD
        self.nB = len(lengths)
        self.nP = sum(length*length for length in lengths)
        self.nL = max(lengths)
        self.lengths = lengths
        self.qkv_dims = ("qkv", "nH", "nN", "nD")
        self.QKV = self.transpose(QKV, target=self.qkv_dims,
            current=dims, contiguous=True)

    @property
    def slices(self):
        start = 0
        attn_start = 0
        for length in self.lengths:
            q_slice = slice(start, start+length)
            k_slice = slice(start, start+length)
            attn_slice = slice(attn_start, attn_start+length*length)
            yield q_slice, k_slice, attn_slice, (length, length)
            start += length
            attn_start += length * length

    def _get_feature(self, index, dims, contiguous):
        assert self.qkv_dims[0] == "qkv"
        current = self.qkv_dims[1:]
        data = self.QKV[index]
        return self.transpose(data,
            current=current, target=dims, contiguous=contiguous)

    def get_queries(self, dims, contiguous=False):
        return self._get_feature(0, dims, contiguous)

    def get_keys(self, dims, contiguous=False):
        return self._get_feature(1, dims, contiguous)
    
    def get_values(self, dims, contiguous=False):
        return self._get_feature(2, dims, contiguous)

    def transpose(self, array, target, current, contiguous=False):
        for i, dim in enumerate(current):
            if hasattr(self, dim):
                assert array.shape[i] == getattr(self, dim)
        assert set(current) == set(target)
        if current == target:
            return array
        array = array.transpose([current.index(d) for d in target])
        if (contiguous and not array.flags["C_CONTIGUOUS"]):
            array = self.ops.xp.ascontiguousarray(array)
        return array

    def get_attn(self, scale=1.):
        return self._get_attn_cpu(self.ops.xp.array([scale], dtype="f"))

    def _get_attn_cpu(self, scale):
        # Transpose keys to (heads, lengths, dims)
        Q = self.get_queries(("nH", "nN", "nD"), contiguous=True)
        K = self.get_keys(("nH", "nN", "nD"), contiguous=True)
        attn = numpy.zeros((self.nH, self.nP), dtype="f")
        for h in range(self.nH):
            for q_slice, k_slice, attn_slice, attn_shape in self.slices:
                # Do the dot product and attention in-place, for efficiency.
                blis.py.gemm(Q[h, q_slice], K[h, k_slice], beta=scale, trans2=True,
                    out=attn[h, attn_slice].reshape(attn_shape))
                self.ops.softmax(attn[h, attn_slice].reshape(attn_shape),
                    axis=-1, inplace=True)
        
        def backprop_get_attn_cpu(d_attn):
            dQ = self.ops.allocate((self.nH, self.nN, self.nD))
            dK = self.ops.allocate((self.nH, self.nN, self.nD))
            for h in range(self.nH):
                for q_slice, k_slice, attn_slice, attn_shape in self.slices:
                    seq_attn = attn[h, attn_slice].reshape(attn_shape)
                    d_seq_attn = d_attn[h, attn_slice].reshape(attn_shape)
                    # d_dots has shape (qlength, klength)
                    d_dots = self.ops.backprop_softmax(seq_attn, d_seq_attn, axis=-1)
                    assert d_dots.shape == attn_shape
                    # Compute (qlength, klength) @ (klength, nD) = (qlength, nD)
                    blis.py.gemm(d_dots, K[h, k_slice], alpha=scale,
                        out=dQ[h, q_slice])
                    # Compute (qlength, klength).T @ (qlength, nD) = (klength, nD)
                    blis.py.gemm(d_dots, Q[h, q_slice], trans1=True, alpha=scale,
                        out=dK[h, k_slice])
            return dQ, dK
        return attn, backprop_get_attn_cpu

    def apply_attn(self, attn):
        assert attn.shape == (self.nH, self.nP)
        values = self.get_values(("nH", "nN", "nD"), contiguous=True)
        # Attn is (nH, nQ, nKV)
        # V    is (nH, nV, nD)
        # where nK == nV
        # Do: (nH, nQ, nK) @ (nH, nV, nD) = (nH, nQ, nD)
        output = self.ops.allocate((self.nH, self.nN, self.nD))
        for h in range(self.nH):
            for q_slice, v_slice, attn_slice, attn_shape in self.slices:
                seq_attn = attn[h, attn_slice].reshape(attn_shape)
                self.ops.gemm(seq_attn, values[h, v_slice], out=output[h, q_slice])

        def backprop_apply_attention(d_output):
            d_values = self.ops.allocate((self.nH, self.nN, self.nD))
            d_attn = self.ops.allocate((self.nH, self.nP))
            for h in range(self.nH):
                for q_slice, v_slice, attn_slice, attn_shape in self.slices:
                    seq_attn = attn[h, attn_slice].reshape(attn_shape)
                    # (nQ, nV).T @ (nQ, nD) = (nV, nD)
                    blis.py.gemm(seq_attn, d_output[h, q_slice], trans1=True,
                        out=d_values[h, v_slice])
                    # (nQ, nD) @ (nV, nD).T = (nQ, nV)
                    blis.py.gemm(d_output[h, q_slice], values[h, v_slice], trans2=True,
                        out=d_attn[h, attn_slice].reshape(attn_shape))
            return d_values, d_attn
        return output, backprop_apply_attention


def prepare_self_attention(affine, nM=300, nH=6):
    nD = nM // nH
    
    def qkv_sa_forward(Xs, drop=0.0):
        (X, lengths), was_ragged = _get_ragged_array(affine.ops, Xs)
        assert X.shape[1] == nM
        assert X.shape == (sum(lengths), Xs[0].shape[-1]), (X.shape, Xs[0].shape)
        QKV, get_dX = affine.begin_update(X, drop=drop)
        output = AttentionInputs(QKV.reshape((X.shape[0], 3, nH, -1)), lengths,
            dims=("nN", "qkv", "nH", "nD"))
        assert output.nH == nH
        assert output.nD == nD

        def qkv_sa_backward(d_output, sgd=None):
            dQKV = d_output.transpose(d_output.QKV,
                current=d_output.qkv_dims, target=("nN", "qkv", "nH", "nD"),
                contiguous=True)
            dQKV = dQKV.reshape((dQKV.shape[0], -1))
            dX = get_dX(dQKV, sgd=sgd)
            if was_ragged:
                return dX
            else:
                return affine.ops.unflatten(dX, lengths)

        return output, qkv_sa_backward

    return wrap(qkv_sa_forward, affine)


def _get_ragged_array(ops, Xs):
    if hasattr(Xs, "__len__") and len(Xs) == 2 and Xs[1].dtype == "i":
        return Xs, True
    else:
        return (ops.flatten(Xs), [len(x) for x in Xs]), False


class MultiHeadedAttention(Model):
    """Multi-headed attention. Requires a preprocessor to prepare (Qs, Ks, Vs, masks)
    triples, such as the prepare_self_attention() preprocessor. A layer
    should run after this to do the projection as well."""
    def __init__(self):
        Model.__init__(self)

    def begin_update(self, qkv, drop=0.0):
        attn, get_dQ_dK = qkv.get_attn(scale=self.ops.xp.sqrt(qkv.nM))
        output0, get_dV_d_attn = qkv.apply_attn(attn)
        output1 = qkv.transpose(output0, current=("nH", "nN", "nD"),
            target=("nN", "nH", "nD"), contiguous=True)
        output2 = output1.reshape((qkv.nN, qkv.nH*qkv.nD))
        lengths = qkv.lengths
        nN, nH, nD = (qkv.nN, qkv.nH, qkv.nD)

        def backprop_multiheaded_attention(d_output2, sgd=None):
            assert d_output2.shape == (nN, nH*nD)
            d_output1 = d_output2.reshape((nN, nH, nD))
            d_output0 = qkv.transpose(d_output1, current=("nN", "nH", "nD"),
                target=("nH", "nN", "nD"), contiguous=True)
            dV, d_attn = get_dV_d_attn(d_output0)
            dQ, dK = get_dQ_dK(d_attn)
            assert dQ.shape == (nH, nN, nD)
            assert dK.shape == (nH, nN, nD)
            assert dV.shape == (nH, nN, nD)
            dQKV = self.ops.xp.vstack((dQ, dK, dV)).reshape((3, nH, nN, nD))
            return AttentionInputs(dQKV, lengths, dims=("qkv", "nH", "nN", "nD"))
        return (output2, lengths), backprop_multiheaded_attention
