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
            assert d_attn.shape == ("nB", "nH", "nL", "nL")
            d_dots = self.ops.backprop_softmax(attn, d_attn, axis=-1)
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
        return context, None


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
        self.QKV = self.transpose(QKV, target=("qkv", "nH", "nN", "nD"),
            current=dims, contiguous=True)

    @property
    def slices(self):
        start = 0
        attn_start = 0
        for length in self.lengths:
            attn_length = length*length
            yield start, start+length, attn_start, attn_start+attn_length
            start += length
            attn_start += attn_length

    def _get_feature(self, index, dims, contiguous):
        data = self.QKV[index]
        return self.transpose(data,
            current=("nH", "nN", "nD"), target=dims, contiguous=contiguous)

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
        if contiguous and not array.flags["C_CONTIGUOUS"]:
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
            for s, e, aS, aE in self.slices:
                blis.py.gemm(Q[h, s:e], K[h, s:e], beta=scale, trans2=True,
                    out=attn[h, aS:aE].reshape((e-s, e-s)))
                self.ops.softmax(attn[h, aS:aE].reshape(e-s, e-s), axis=-1, inplace=True)
        
        def backprop_get_attn_cpu(d_attn):
            dQ = self.ops.allocate((self.nH, self.nN, self.nD))
            dK = self.ops.allocate((self.nH, self.nN, self.nD))
            for h in range(self.nH):
                for s, e, aS, aE in self.slices:
                    # d_dots has shape (qlength, klength)
                    d_dots = self.ops.backprop_softmax(
                        attn[h, s:e], d_attn[h, aS:aE], axis=-1)
                    # Compute (qlength, klength) @ (klength, nD) = (qlength, nD)
                    blis.py.gemm.gemm(d_dots, K[h, s:e], alpha=scale,
                        out=dQ[h, s:e])
                    # Compute (qlength, klength).T @ (qlength, nD) = (klength, nD)
                    blis.py.gemm(d_dots, Q[h, s:e], alpha=scale, trans1=True,
                        out=dK[h, s:e])
            dQ = self.transpose(dQ,
                target=("nN", "nH", "nD"), current=("nH", "nN", "nD"))
            dK = self.transpose(dK,
                target=("nN", "nH", "nD"), current=("nH", "nN", "nD"))
            return dQ, dK
        return attn, backprop_get_attn_cpu

    def _get_attn_gpu(queries, keys, values, out=None):
        A_ptr = self.K.data.ptr
        B_ptr = self.Q.data.ptr
        C_ptr = out.data.ptr
        strideA = self.nL * self.K.shape[1]
        strideB = self.Q.shape[1]
        strideC = self.nL

        sgemmStridedBatched(
            handle,
            False, False,
            Q.shape[0], 1, Q.shape[1],
            alpha,
            A_ptr, K.shape[1], da, strideA,
            B_ptr, Q.shape[1], strideB,
            C_ptr, V.shape[1], strideC,
            batchCount
        )

    def apply_attn(self, attn, attn_dims=("nH", "nP")):
        attn = self.transpose(attn,
            target=("nH", "nP"), current=attn_dims, contiguous=True)
        values = self.get_values(("nH", "nN", "nD"), contiguous=True)
        # Attn is (nH, nQ, nV)
        # V    is (nH, nV, nD)
        # Do: (nH, nQ, nV) @ (nH, nV, nD) = (nH, nQ, nD)
        output = self.ops.allocate((self.nH, self.nN, self.nD))
        for h in range(self.nH):
            for s, e, aS, aE in self.slices:
                seq_attn = attn[h, aS:aE].reshape((e-s, -1))
                self.ops.gemm(seq_attn, values[h, s:e], out=output[h, s:e])
        output = self.transpose(output,
            target=("nN", "nH", "nD"), current=("nH", "nN", "nD"), contiguous=True)

        def backprop_apply_attention(d_output):
            output = self.transpose(d_output,
                target=("nH", "nN", "nD"), current=("nN", "nH", "nD"), contiguous=True)
            d_values = self.ops.allocate((self.nH, self.nN, self.nD))
            d_attn = self.ops.allocate((self.nH, self.nP))
            for h in range(self.nH):
                for s, e, aS, aE in self.slices:
                    n = e-s
                    seq_attn = attn[h, aS:aE].reshape((e-s, -1))
                    # (nQ, nV).T @ (nQ, nD) = (nV, nD)
                    self.ops.gemm(seq_attn, d_output[h, s:e],
                        trans1=True, out=d_values[h, s:e])
                    # (nQ, nD) @ (nV, nD).T = (nQ, nV)
                    self.ops.gemm(d_output[h, s:e], d_values[h, s:e],
                        trans2=True, out=d_attn[h, aS:aE].reshape((e-s, -1)))
            d_values = self.transpose(d_values,
                target=("nN", "nH", "nD"), current=("nH", "nN", "nD"))
            d_attn = self.transpose(d_attn,
                target=("nN", "nH", "nN"), current=("nH", "nN", "nN"))
            return d_values, d_attn
        return output, backprop_apply_attention


def prepare_self_attention(affine, nM=300, nH=6):
    nD = nM // nH
    
    def qkv_sa_forward(Xs, drop=0.0):
        (X, lengths), was_ragged = _get_ragged_array(affine.ops, Xs)
        assert X.shape == (sum(lengths), Xs[0].shape[-1]), (X.shape, Xs[0].shape)
        QKV, get_dX = affine.begin_update(X, drop=drop)
        output = AttentionInputs(QKV.reshape((X.shape[0], 3, nH, -1)), lengths)

        def qkv_sa_backward(d_output, sgd=None):
            dQKV = d_output.QKV.reshape((QKV.shape[0], -1))
            dX = get_dX(dQKV, sgd=sgd)
            if was_ragged:
                return dX, lengths
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
        attn, get_dQ_dK = qkv.queries_dot_keys(scale=self.ops.xp.sqrt(qkv.nM))
        output0, get_dV_d_attn = qkv.apply_attn(attn)
        output1 = output0.reshape((qkv.nN, qkv.nH*qkv.nD))
        lengths = qkv.lengths

        def backprop_multiheaded_attention(d_output1, sgd=None):
            d_output0 = d_output1.reshape((nN, nH, nD))
            dV, d_attn = get_dV_d_attn(d_output0)
            dQ, dK = get_dQ_dK(d_attn)
            dQKV = self.ops.xp.hstack((dQ, dK, dV))
            dQKV = dQKV.reshape((dQKV.shape[0], 3, nH, nD))
            return AttentionInputs(dQKV, lengths)

        return (output1, lengths), backprop_multiheaded_attention
