from typing import Optional, List, Tuple, Sequence, Union, cast
import numpy

from ..types import Xp, Array, Shape, DTypes, DTypesInt, DTypesFloat, Padded
from ..types import Array1d, Array2d, Array3d, Array4d, ArrayTypes, ArrayT
from ..util import copy_array, get_array_module


class Ops:
    device: str = "cpu"
    xp: Xp = numpy

    def __init__(self, xp: Optional[Xp] = None) -> None:
        if xp is not None:
            self.xp = xp

    def to_numpy(self, data):
        if isinstance(data, numpy.ndarray):
            return data
        else:
            raise ValueError("Cannot convert non-numpy from base Ops class.")

    def seq2col(self, seq: ArrayT, nW: int) -> ArrayT:
        """Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1))
        sequence. The new sequence is constructed by concatenating nW preceding
        and succeeding vectors onto each column in the sequence, to extract a
        window of features.
        """
        # This is a test implementation that only supports nW=1
        assert nW == 1
        B = seq.shape[0]
        I = seq.shape[1]
        cols: Array3d = self.alloc_f3d(B, (nW * 2 + 1), I)
        # Copy left contexts. The last words aren't the left-context for anything.
        cols[nW:, :nW] = seq[:-nW].reshape((-1, nW, I))
        cols[:, nW] = seq
        cols[:-nW, nW + 1 :] = seq[nW:].reshape((-1, nW, I))
        return cols.reshape((B, I * (2 * nW + 1)))

    def backprop_seq2col(self, dY: ArrayT, nW: int) -> Array:
        # This is a test implementation that only supports nW=1
        assert nW == 1
        nF = nW * 2 + 1
        B = dY.shape[0]
        I = dY.shape[1] // nF
        # Having trouble getting the kernel to work...
        dX = self.alloc_f2d(B, I)
        dY = dY.reshape((B, nF, I))
        dX[:-nW] += dY[nW:, :nW].reshape((-1, I))
        dX += dY[:, nW]
        dX[nW:] += dY[:-nW, nW + 1 :].reshape((-1, I))
        return dX

    def gemm(
        self,
        x: Array2d,
        y: Array2d,
        out: Optional[Array2d] = None,
        trans1: bool = False,
        trans2: bool = False,
    ) -> Array2d:
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        if out is None:
            return self.xp.dot(x, y)
        else:
            self.xp.dot(x, y, out=out)
            return out

    def affine(self, X, W, b):
        Y = self.gemm(X, W, trans2=True)
        Y += b
        return Y

    def flatten(
        self,
        X: Sequence[ArrayT],
        dtype: Optional[DTypes] = None,
        pad: int = 0,
        ndim_if_empty: int = 2,
    ) -> ArrayT:
        if X is None or len(X) == 0:
            return self.alloc((0,) * ndim_if_empty, dtype=dtype or "f")
        xp = get_array_module(X[0])
        X = [x for x in X if x.size != 0]
        if int(pad) >= 1:
            padded = []
            for x in X:
                padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
                padded.append(x)
            padded.append(xp.zeros((pad,) + x.shape[1:], dtype=x.dtype))
            X = padded
        result = xp.concatenate(X)
        if dtype is not None:
            result = xp.asarray(result, dtype=dtype)
        return result

    def unflatten(self, X: ArrayT, lengths: Array1d, pad: int = 0) -> List[ArrayT]:
        unflat = []
        pad = int(pad)
        for length in lengths:
            length = int(length)
            if pad >= 1 and length != 0:
                X = X[pad:]
            unflat.append(X[:length])
            X = X[length:]
        if pad >= 1:
            X = X[pad:]
        assert len(X) == 0
        assert len(unflat) == len(lengths)
        return unflat

    def pad(self, seqs: List[Array]) -> Array:
        if not seqs:
            raise ValueError("Cannot pad empty sequence")
        if len(set(seq.ndim for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences with different ndims")
        if len(set(seq.dtype for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences with different dtypes")
        if len(set(seq.shape[1:] for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences that differ on other dimensions")
        shape = [len(seqs)]
        # Find the maximum dimension along each axis. That's what we'll pad to.
        dim_sizes = zip(*[seq.shape for seq in seqs])
        shape.extend(max(sizes) for sizes in dim_sizes)
        # Now copy the data into our new buffer.
        output: Array = self.alloc(tuple(shape), dtype=seqs[0].dtype)
        for i, arr in enumerate(seqs):
            # TODO: It would be nice to generalize this to work along different
            # dimensions. We'd have to handle that in the unpad, though, which
            # could be tricky?
            # I don't know how to do the numpy indexing for multi-dimensions
            # anyway. Need to construct the slice object maybe?
            output[i, : arr.shape[0]] = arr
        return output

    def unpad(self, padded: Array, lengths: List[int]) -> List[Array]:
        output = []
        for i, length in enumerate(lengths):
            output.append(padded[i, :length])
        return output

    def list2padded(self, seqs: List[Array2d]) -> Padded:
        """Pack a sequence of 2d arrays into a Padded datatype."""
        if not seqs:
            return Padded(self.alloc_f3d(0, 0, 0), self.alloc_i1d(0), [], [])
        lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
        lengths_indices.sort(reverse=True)
        indices = [i for length, i in lengths_indices]
        lengths = [length for length, i in lengths_indices]
        nB = len(seqs)
        nS = max([len(seq) for seq in seqs])
        arr: Array3d = self.alloc_f3d(nB, nS, seqs[0].shape[1])
        for arr_i, (length, seqs_i) in enumerate(lengths_indices):
            arr[arr_i, :length] = self.asarray(seqs[seqs_i])
        arr = self.xp.ascontiguousarray(arr.transpose((1, 0, 2)))
        # Build a lookup table so we can find how big the batch is at point t.
        batch_size_at_t = self.alloc_i1d(nS, dtype="i")
        batch_size_at_t += 1
        i = len(lengths)
        for t in range(nS):
            if t == lengths[i - 1]:
                i -= 1
                if i == 0:
                    break
            batch_size_at_t[t] = i
        return Padded(arr, batch_size_at_t, lengths, indices)

    def padded2list(self, padded: Padded) -> List[Array2d]:
        indices = padded.indices
        data = padded.data
        lengths = padded.lengths
        unpadded = [None] * len(lengths)
        data = self.xp.ascontiguousarray(data.transpose((1, 0, 2)))
        for i in range(data.shape[0]):
            unpadded[indices[i]] = data[i, : lengths[i]]
        return cast(List[Array2d], unpadded)

    def get_dropout_mask(self, shape: Shape, drop: float) -> Array:
        if drop is None or drop <= 0:
            return self.xp.ones(shape, dtype="f")
        elif drop >= 1.0:
            return self.alloc(shape)
        coinflips = self.xp.random.uniform(0.0, 1.0, shape)
        mask = (coinflips >= drop) / (1.0 - drop)
        return self.asarray(mask, dtype="float32")

    def alloc_f1d(
        self, d0: int, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> Array1d:
        return self.alloc((d0,), dtype=dtype)

    def alloc_f2d(
        self, d0: int, d1: int, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> Array2d:
        return self.alloc((d0, d1), dtype=dtype)

    def alloc_f3d(
        self, d0: int, d1: int, d2: int, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> Array3d:
        return self.alloc((d0, d1, d2), dtype=dtype)

    def alloc_f4d(
        self,
        d0: int,
        d1: int,
        d2: int,
        d3: int,
        *,
        dtype: Optional[DTypesFloat] = "float32",
    ) -> Array4d:
        return self.alloc((d0, d1, d2, d3), dtype=dtype)

    def alloc_f(
        self, shape: Shape, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> ArrayTypes:
        return self.alloc(shape, dtype=dtype)

    def alloc_i1d(self, d0: int, *, dtype: Optional[DTypesInt] = "int32") -> Array1d:
        return self.alloc((d0,), dtype=dtype)

    def alloc_i2d(
        self, d0: int, d1: int, *, dtype: Optional[DTypesInt] = "int32"
    ) -> Array2d:
        return self.alloc((d0, d1), dtype=dtype)

    def alloc_i3d(
        self, d0: int, d1: int, d2: int, *, dtype: Optional[DTypesInt] = "int32"
    ) -> Array3d:
        return self.alloc((d0, d1, d2), dtype=dtype)

    def alloc_i4d(
        self,
        d0: int,
        d1: int,
        d2: int,
        d3: int,
        *,
        dtype: Optional[DTypesInt] = "int32",
    ) -> Array4d:
        return self.alloc((d0, d1, d2, d3), dtype=dtype)

    def alloc_i(
        self, shape: Shape, *, dtype: Optional[DTypesInt] = "int32"
    ) -> ArrayTypes:
        return self.alloc(shape, dtype=dtype)

    def alloc(self, shape: Shape, *, dtype: Optional[DTypes] = "float32") -> ArrayT:
        if isinstance(shape, int):
            shape = (shape,)
        return self.xp.zeros(shape, dtype=dtype)

    def unzip(self, data: Tuple[Array, Array]) -> Tuple[Array, Array]:
        X, y = zip(*data)
        return self.asarray(X), self.asarray(y)

    def asarray(
        self,
        data: Union[ArrayT, Sequence[ArrayT], Sequence[int]],
        *,
        dtype: Optional[DTypes] = None,
    ) -> ArrayT:
        if isinstance(data, self.xp.ndarray):
            if dtype is not None:
                return self.xp.asarray(data, dtype=dtype)
            else:
                return self.xp.asarray(data)
        elif hasattr(data, "numpy"):
            # Handles PyTorch Tensor
            return data.numpy()  # type: ignore
        elif dtype is not None:
            return self.xp.array(data, dtype=dtype)
        else:
            return self.xp.array(data)

    def as_contig(self, data: ArrayT, dtype: Optional[DTypes] = None) -> ArrayT:
        kwargs = {"dtype": dtype} if dtype is not None else {}
        return self.xp.ascontiguousarray(data, **kwargs)

    def sigmoid(self, X: ArrayT, *, inplace: bool = False) -> ArrayT:
        if inplace:
            self.xp.exp(-X, out=X)
            X += 1.0
            X **= -1.0
            return X
        else:
            return 1.0 / (1.0 + self.xp.exp(-X))

    def dsigmoid(self, Y: ArrayT, *, inplace: bool = False) -> ArrayT:
        if inplace:
            Y *= 1 - Y
            return Y
        else:
            return Y * (1.0 - Y)

    def cosine(self, X: Array, Y: ArrayT) -> float:
        # Add a small constant to avoid 0 vectors
        X = X + 1e-8
        Y = Y + 1e-8
        normX = self.xp.linalg.norm(X, axis=1, keepdims=True)
        normY = self.xp.linalg.norm(Y, axis=1, keepdims=True)
        mul_norms = normX * normY
        cosine = (X * Y).sum(axis=1, keepdims=True) / mul_norms
        return cosine

    def cosine_abs_loss(
        self, X: Array, Y: ArrayT, *, ignore_zeros: bool = False
    ) -> float:
        cosine = self.cosine(X, Y)
        losses = self.xp.abs(cosine - 1)
        if ignore_zeros:
            # If the target was a zero vector, don't count it in the loss.
            zero_indices = self.xp.abs(Y).sum(axis=1) == 0
            losses[zero_indices] = 0
        loss = losses.sum()
        return loss

    def get_norm(self, X: Array) -> Array:
        norms = self.xp.linalg.norm(X, axis=1)
        norms[norms == 0] = 1
        return norms

    def dtanh(self, Y: ArrayT, *, inplace: bool = False) -> ArrayT:
        if inplace:
            Y **= 2
            Y *= -1.0
            Y += 1.0
            return Y
        else:
            return 1 - Y ** 2

    def softmax(self, x: Array, *, inplace: bool = False, axis: int = -1) -> Array:
        maxes = self.xp.max(x, axis=axis, keepdims=True)
        shifted = x - maxes
        new_x = self.xp.exp(shifted)
        new_x /= new_x.sum(axis=axis, keepdims=True)
        return new_x

    def softmax_sequences(
        self, Xs: Array2d, lengths: Array1d, *, inplace: bool = False, axis: int = -1
    ) -> Array2d:
        if Xs.ndim >= 3:
            err = f"Softmax currently only supports 2d. Got: {Xs.ndim}"
            raise NotImplementedError(err)
        # This loses almost no fidelity, and helps the numerical stability.
        Xs = self.xp.clip(Xs, -20.0, 20.0)
        new_x = self.xp.exp(Xs)
        summed = self.backprop_sum_pool(self.sum_pool(new_x, lengths), lengths)
        new_x /= summed
        return new_x

    def backprop_softmax(self, Y: Array, dY: Array, *, axis: int = -1) -> Array:
        dX = Y * dY
        dX -= Y * dX.sum(axis=axis, keepdims=True)
        return dX

    def backprop_softmax_sequences(
        self, dY: Array2d, Y: Array2d, lengths: Array1d
    ) -> Array2d:
        dX = Y * dY
        sum_dX = self.backprop_sum_pool(self.sum_pool(dX, lengths), lengths)
        dX -= Y * sum_dX
        return dX

    def lstm(
        self, acts: Array3d, prevcells: Array2d
    ) -> Tuple[Array2d, Array2d, Array3d]:
        gates = self.as_contig(acts.transpose((2, 0, 1)))
        # Activations is: hf, hi, ho, hc
        self.sigmoid(gates[0], inplace=True)
        self.sigmoid(gates[1], inplace=True)
        self.sigmoid(gates[2], inplace=True)
        self.xp.tanh(gates[3], out=gates[3])
        cells = gates[0] * prevcells
        cells += gates[1] * gates[3]
        hiddens = self.xp.tanh(cells)
        hiddens *= gates[2]
        return hiddens, cells, gates

    def backprop_lstm(
        self,
        d_cells: Array2d,
        d_hiddens: Array2d,
        gates: Array3d,
        cells: Array2d,
        prevcells: Array2d,
    ) -> Tuple[Array3d, Array2d]:
        (hf, hi, ho, hc) = (0, 1, 2, 3)
        cells_tanh = self.xp.tanh(cells)
        # Gradient for ho and c in h = sigmoid(ho) * tanh(c)
        d_ho = cells_tanh
        d_ho *= d_hiddens * self.dsigmoid(gates[ho])
        d_prevcells = gates[ho] * d_hiddens * self.dtanh(cells_tanh)
        d_prevcells += d_cells  # Carry gradient from timestep
        # Gradient for hf, hi, hc, prev[i]
        # in c = sigmoid(hf) * prev[i] + sigmoid(hi) * tanh(hc)
        d_hf = self.dsigmoid(gates[hf]) * d_prevcells * prevcells
        d_hi = self.dsigmoid(gates[hi]) * d_prevcells * gates[hc]
        d_hc = self.dtanh(gates[hc]) * d_prevcells * gates[hi]
        d_prevcells *= gates[hf]
        copy_array(d_cells, d_prevcells)  # TOOD: Wtf is this?
        d_acts = self.xp.concatenate((d_hf, d_hi, d_ho, d_hc), axis=-1)
        return d_acts, d_prevcells

    # TODO: types
    def maxout(self, X):
        which = X.argmax(axis=-1)
        return X.max(axis=-1), which

    # TODO: types
    def backprop_maxout(self, dY, which, P):
        dX = self.alloc((dY.shape[0], dY.shape[1], P), dtype="float32")
        for b in range(dY.shape[0]):
            for o in range(dY.shape[1]):
                dX[b, o, which[b, o]] = dY[b, o]
        return dX

    def relu(self, X: Array, inplace: bool = False) -> Array:
        if not inplace:
            return X * (X > 0)
        else:
            X *= X > 0
            return X

    def backprop_relu(self, dY: Array, Y: Array, inplace: bool = False) -> Array:
        if not inplace:
            return dY * (Y > 0)
        dY *= Y > 0
        return dY

    def mish(self, X: Array2d, threshold: float = 20.0) -> Array2d:
        Y = self.alloc_f2d(*X.shape, dtype=X.dtype)
        tmp = X * self.xp.tanh(self.xp.log(1.0 + self.xp.exp(X)))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] >= threshold:
                    Y[i, j] = X[i, j]
                else:
                    Y[i, j] = tmp[i, j]
        return Y

    def backprop_mish(
        self,
        dY: Array2d,
        X: Array2d,
        threshold: float = 20.0,
        out: Optional[Array2d] = None,
    ):
        xp = get_array_module(X)
        indices = X < threshold
        Xsub = X[indices]
        dYsub = dY[indices]
        omega = 4.0 * (Xsub + 1.0)
        omega += 4.0 * xp.exp(2.0 * Xsub)
        omega += xp.exp(Xsub) * ((4.0 * Xsub) + 6.0)
        delta = 2.0 * xp.exp(Xsub)
        delta += xp.exp(2.0 * Xsub)
        delta += 2.0
        dXsub = dYsub * ((xp.exp(Xsub) * omega) / (delta ** 2))
        if out is None:
            out = xp.zeros(dY.shape, dtype="f")
        # Gradient when above threshold will ignore softplus.
        out[:] = dY + dY * self.dtanh(X)
        out[indices] = dXsub
        return out

    def update_averages(
        self, ema: Array, weights: Array, t: int, max_decay: float = 0.9999
    ) -> None:
        decay = (1.0 + t) / (10.0 + t)
        if decay > max_decay:
            decay = max_decay
        ema -= (1 - decay) * (ema - weights)

    def adam(
        self,
        weights: Array1d,
        gradient: Array1d,
        mom1: Array1d,
        mom2: Array1d,
        beta1: float,
        beta2: float,
        eps: float,
        learn_rate: float,
        mod_rate: float = 1.0,
    ) -> Tuple[Array1d, Array1d, Array1d, Array1d]:
        mom1 *= beta1
        mom2 *= beta2
        mom1 += gradient * (1.0 - beta1)
        mom2 += gradient * gradient * (1.0 - beta2)
        # Here we assume learn rate is calculated by the caller.
        # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
        weights -= learn_rate * (mom1 / (mod_rate * self.xp.sqrt(mom2) + eps))
        return weights, gradient, mom1, mom2

    def clip_gradient(self, gradient: Array, threshold: float) -> Array:
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm
        return gradient

    def logloss(self, y_true: Array, y_pred: Array):
        log_yp = self.xp.log(y_pred + 1e-8)
        loss = (y_true * log_yp) + (1 - y_true) * self.xp.log((1 - y_pred) + 1e-8)
        return -loss

    def sum_pool(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].sum(axis=0)
            start += length
        return Y

    def mean_pool(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].mean(axis=0)
            start += length
        return Y

    def max_pool(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        which = self.alloc_i2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            which[i] = X[start : start + length].argmax(axis=0)
            Y[i] = X[start : start + length].max(axis=0)
            start += length
        return Y, which

    def backprop_sum_pool(self, d_sums: Array2d, lengths: Array1d) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_sums.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_sums[i]
            start += length
        return dX

    def backprop_mean_pool(self, d_means: Array2d, lengths: Array1d) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_means.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_means[i] / length
            start += length
        return dX

    def backprop_max_pool(
        self, d_maxes: Array2d, which: Array2d, lengths: Array1d
    ) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_maxes.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length, which[i]] = d_maxes[i]
            start += length
        return dX

    def hash(self, ids: Array, seed: Array) -> Array:
        # TODO: fix
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        return self.asarray(
            numpy_ops.hash(numpy_ops.asarray(ids, dtype="uint64"), seed)
        )

    def ngrams(self, n: int, keys: Array) -> Array:
        # TODO: fix
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        return self.asarray(
            numpy_ops.ngrams(n, numpy_ops.asarray(keys, dtype="uint64"))
        )

    def position_encode(
        self, N: int, D: int, period: int = 10000, out: Optional[Array] = None
    ) -> Array:
        # TODO: fix
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        return self.asarray(numpy_ops.position_encode(N, D, period, out))

    def scatter_add(self, out: Array, ids: Array, inputs: Array) -> Array:
        return self.xp.add.at(out, ids, inputs)
