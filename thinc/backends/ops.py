from typing import Optional, List, Callable, Tuple, Sequence, Union, cast

from ..types import Xp, Array, Shape, DTypes, DTypesInt, DTypesFloat
from ..types import Floats1d, Floats2d, Floats3d, Floats4d
from ..types import Ints1d, Ints2d, Ints3d, Ints4d, IntsNd
from ..types import ArrayTypesInt, ArrayTypesFloat, ArrayT
from ..types import Array2d, Array3d
from ..util import copy_array, get_array_module


class Ops:
    device: str = "cpu"
    xp: Xp = None

    def __init__(self, xp: Optional[Xp] = None) -> None:
        if xp is not None:
            self.xp = xp

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
        cols: Floats3d = self.alloc_f3d(B, (nW * 2 + 1), I)
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
        x: Floats2d,
        y: Floats2d,
        out: Optional[Floats2d] = None,
        trans1: bool = False,
        trans2: bool = False,
    ) -> Floats2d:
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        if out is None:
            return self.xp.dot(x, y)
        else:
            self.xp.dot(x, y, out=out)
            return out

    def flatten(
        self, X: Sequence[ArrayT], dtype: Optional[DTypes] = None, pad: int = 0
    ) -> ArrayT:
        if X is None or len(X) == 0:
            return self.alloc((0,), dtype=dtype or "f")
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

    def unflatten(self, X: ArrayT, lengths: Ints1d, pad: int = 0) -> List[ArrayT]:
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

    def pad_sequences(
        self, seqs_in: Sequence[Array2d], pad_to: Optional[int] = None
    ) -> Tuple[Array3d, Callable]:
        lengths: IntsNd = self.asarray([len(seq) for seq in seqs_in], dtype="i")
        nB = len(seqs_in)
        if pad_to is None:
            pad_to = int(lengths.max())
        arr: Floats3d = self.alloc_f3d(
            nB, int(pad_to), seqs_in[0].shape[1], dtype="float32"
        )
        for arr_i, seq in enumerate(seqs_in):
            arr[arr_i, : seq.shape[0]] = self.asarray(seq)

        def unpad(padded: Floats3d) -> Sequence[Floats2d]:
            unpadded = [None] * len(lengths)
            for i in range(padded.shape[0]):
                unpadded[i] = padded[i, : lengths[i]]
            return cast(Sequence[Floats2d], unpadded)

        return arr, unpad

    def square_sequences(
        self, seqs: Sequence[Array2d]
    ) -> Tuple[Array3d, Ints1d, Callable]:
        """Sort a batch of sequence by decreasing length, pad, and transpose
        so that the outer dimension is the timestep. Return the padded batch,
        along with an array indicating the actual length at each step, and a callback
        to reverse the transformation.
        """
        lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
        lengths_indices.sort(reverse=True)
        indices = [i for length, i in lengths_indices]
        lengths = [length for length, i in lengths_indices]
        nB = len(seqs)
        nS = max([len(seq) for seq in seqs])
        arr: Floats3d = self.alloc_f3d(nB, nS, seqs[0].shape[1])
        for arr_i, (length, seqs_i) in enumerate(lengths_indices):
            arr[arr_i, :length] = self.asarray(seqs[seqs_i])
        extra_dims = tuple(range(2, len(arr.shape)))
        arr = self.xp.ascontiguousarray(arr.transpose((1, 0) + extra_dims))
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

        def unpad(padded: Array3d) -> Sequence[Array2d]:
            unpadded = [None] * len(lengths)
            padded = self.xp.ascontiguousarray(padded.transpose((1, 0) + extra_dims))
            for i in range(padded.shape[0]):
                unpadded[indices[i]] = padded[i, : lengths[i]]
            return cast(Sequence[Floats2d], unpadded)

        return arr, batch_size_at_t, unpad

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
    ) -> Floats1d:
        return self.alloc((d0,), dtype=dtype)

    def alloc_f2d(
        self, d0: int, d1: int, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> Floats2d:
        return self.alloc((d0, d1), dtype=dtype)

    def alloc_f3d(
        self, d0: int, d1: int, d2: int, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> Floats3d:
        return self.alloc((d0, d1, d2), dtype=dtype)

    def alloc_f4d(
        self,
        d0: int,
        d1: int,
        d2: int,
        d3: int,
        *,
        dtype: Optional[DTypesFloat] = "float32",
    ) -> Floats4d:
        return self.alloc((d0, d1, d2, d3), dtype=dtype)

    def alloc_f(
        self, shape: Shape, *, dtype: Optional[DTypesFloat] = "float32"
    ) -> ArrayTypesFloat:
        return self.alloc(shape, dtype=dtype)

    def alloc_i1d(self, d0: int, *, dtype: Optional[DTypesInt] = "int32") -> Ints1d:
        return self.alloc((d0,), dtype=dtype)

    def alloc_i2d(
        self, d0: int, d1: int, *, dtype: Optional[DTypesInt] = "int32"
    ) -> Ints2d:
        return self.alloc((d0, d1), dtype=dtype)

    def alloc_i3d(
        self, d0: int, d1: int, d2: int, *, dtype: Optional[DTypesInt] = "int32"
    ) -> Ints3d:
        return self.alloc((d0, d1, d2), dtype=dtype)

    def alloc_i4d(
        self,
        d0: int,
        d1: int,
        d2: int,
        d3: int,
        *,
        dtype: Optional[DTypesInt] = "int32",
    ) -> Ints4d:
        return self.alloc((d0, d1, d2, d3), dtype=dtype)

    def alloc_i(
        self, shape: Shape, *, dtype: Optional[DTypesInt] = "int32"
    ) -> ArrayTypesInt:
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
        if inplace:
            copy_array(x, new_x)
            return x
        else:
            return new_x

    def softmax_sequences(
        self, Xs: Floats2d, lengths: Ints1d, *, inplace: bool = False, axis: int = -1
    ) -> Floats2d:
        if Xs.ndim >= 3:
            err = f"Softmax currently only supports 2d. Got: {Xs.ndim}"
            raise NotImplementedError(err)
        # This loses almost no fidelity, and helps the numerical stability.
        Xs = self.xp.clip(Xs, -20.0, 20.0)
        new_x = self.xp.exp(Xs)
        summed = self.backprop_sum_pool(self.sum_pool(new_x, lengths), lengths)
        new_x /= summed
        if inplace:
            copy_array(Xs, new_x)
            return Xs
        else:
            return new_x

    def backprop_softmax(self, Y: Array, dY: Array, *, axis: int = -1) -> Array:
        dX = Y * dY
        dX -= Y * dX.sum(axis=axis, keepdims=True)
        return dX

    def backprop_softmax_sequences(
        self, dY: Floats2d, Y: Floats2d, lengths: Ints1d
    ) -> Floats2d:
        dX = Y * dY
        sum_dX = self.backprop_sum_pool(self.sum_pool(dX, lengths), lengths)
        dX -= Y * sum_dX
        return dX

    def clip_low(self, x: ArrayT, value: ArrayT, *, inplace: bool = False) -> ArrayT:
        if inplace:
            return self.xp.maximum(x, value, out=x)
        else:
            return self.xp.maximum(x, value)

    def take_which(self, x: ArrayT, which: ArrayT, *, axis: int = -1) -> ArrayT:
        output: ArrayT = self.alloc(which.shape)
        for i in range(x.shape[axis]):
            output += x[:, :, i] * (which == i)
        return output

    def backprop_take(self, dX: Array, which: Array, nP: int) -> Array:
        dX__bop = self.alloc_f3d(dX.shape[0], dX.shape[1], nP)
        for i in range(nP):
            dX__bop[:, :, i] += dX * (which == i)
        return dX__bop

    def lstm(
        self, output: Floats2d, cells: Floats2d, acts: Floats3d, prev: Floats2d
    ) -> None:
        # Activations is: hf, hi, ho, hc
        self.sigmoid(acts[0], inplace=True)
        self.sigmoid(acts[1], inplace=True)
        self.sigmoid(acts[2], inplace=True)
        self.xp.tanh(acts[3], out=acts[3])
        cells[:] = acts[0]
        cells *= prev
        cells += acts[1] * acts[3]
        self.xp.tanh(cells, out=output)
        output *= acts[2]

    # TODO: types
    def backprop_lstm(
        self,
        d_cells: Floats2d,
        d_prev: Floats2d,
        d_gates: Floats3d,
        d_output: Floats2d,
        gates: Floats3d,
        cells: Floats2d,
        prev: Floats2d,
    ) -> None:
        (hf, hi, ho, hc) = (0, 1, 2, 3)
        cells_tanh = self.xp.tanh(cells)
        # Gradient for ho and c in h = sigmoid(ho) * tanh(c)
        d_gates[ho] = cells_tanh
        d_gates[ho] *= d_output * self.dsigmoid(gates[ho])
        d_prevcells = gates[ho] * d_output * self.dtanh(cells_tanh)
        d_prevcells += d_cells  # Carry gradient from timestep
        # Gradient for hf, hi, hc, prev[i]
        # in c = sigmoid(hf) * prev[i] + sigmoid(hi) * tanh(hc)
        d_gates[hf] = self.dsigmoid(gates[hf]) * d_prevcells * prev
        d_gates[hi] = self.dsigmoid(gates[1]) * d_prevcells * gates[hc]
        d_gates[hc] = self.dtanh(gates[hc]) * d_prevcells * gates[hi]
        d_prev[:] = d_prevcells * gates[hf]
        copy_array(d_cells, d_prevcells)

    def softplus(
        self, X: Floats2d, threshold: float = 20.0, out: Optional[Floats2d] = None
    ) -> Array:
        xp = get_array_module(X)
        log1p_exp = xp.log1p(xp.exp(X))
        indices = X >= threshold
        log1p_exp[indices] = X[indices]
        if out is None:
            return log1p_exp
        else:
            out[:] = log1p_exp
            return out

    def backprop_softplus(
        self,
        dY: Floats2d,
        X: Floats2d,
        threshold: float = 20.0,
        out: Optional[Floats2d] = None,
    ) -> Array:
        xp = get_array_module(X)
        out_: Array
        if out is None:
            out_ = xp.zeros(X.shape, dtype="f")
        else:
            out_ = out
        out_[:] = 1 - 1 / (1 + xp.exp(X))
        out_ *= dY
        indices = X >= threshold
        out_[indices] = dY[indices]
        return out_

    def mish(
        self, X: Floats2d, threshold: float = 20.0, out: Optional[Floats2d] = None
    ) -> Floats2d:
        Xsoft = self.softplus(X, threshold=threshold, out=out)
        Y = self.xp.tanh(Xsoft, out=out)
        Y *= X
        return Y

    def backprop_mish(
        self,
        dY: Floats2d,
        X: Floats2d,
        threshold: float = 20.0,
        out: Optional[Floats2d] = None,
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

    # TODO: types
    def adam(
        self,
        weights: Floats1d,
        gradient: Floats1d,
        mom1: Floats1d,
        mom2: Floats1d,
        beta1: float,
        beta2: float,
        eps: float,
        learn_rate: float,
        mod_rate: float = 1.0,
    ) -> None:
        mom1 *= beta1
        mom2 *= beta2
        mom1 += gradient * (1.0 - beta1)
        mom2 += gradient * gradient * (1.0 - beta2)
        # Here we assume learn rate is calculated by the caller.
        # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
        weights -= learn_rate * (mom1 / (mod_rate * self.xp.sqrt(mom2) + eps))
        gradient.fill(0)

    def clip_gradient(self, gradient: Floats1d, threshold: float) -> None:
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm

    def logloss(self, y_true: Array, y_pred: Array):
        log_yp = self.xp.log(y_pred + 1e-8)
        loss = (y_true * log_yp) + (1 - y_true) * self.xp.log((1 - y_pred) + 1e-8)
        return -loss

    def sum_pool(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].sum(axis=0)
            start += length
        return Y

    def mean_pool(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].mean(axis=0)
            start += length
        return Y

    def max_pool(self, X: Floats2d, lengths: Ints1d) -> Floats2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].max(axis=0)
            start += length
        return Y

    def backprop_sum_pool(self, d_sums: Floats2d, lengths: Ints1d) -> Floats2d:
        dX = self.alloc_f2d(lengths.sum(), d_sums.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_sums[i]
            start += length
        return dX

    def backprop_mean_pool(self, d_means: Floats2d, lengths: Ints1d) -> Floats2d:
        dX = self.alloc_f2d(lengths.sum(), d_means.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_means[i] / length
            start += length
        return dX

    def backprop_max_pool(
        self, d_maxes: Floats2d, which: Ints2d, lengths: Ints1d
    ) -> Floats2d:
        dX = self.alloc_f2d(lengths.sum(), d_maxes.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length, which[i]] = d_maxes[i]
            start += length
        return dX
