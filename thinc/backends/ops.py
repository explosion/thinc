from typing import Optional, List, Tuple, Sequence, Union, cast
from typing import Iterator
import numpy
import itertools

from ..types import Xp, Array, Shape, DTypes, DTypesInt, DTypesFloat
from ..types import Array1d, Array2d, Array3d, Array4d, ArrayTypes, ArrayT
from ..types import DeviceTypes, Generator, Padded, Batchable, SizedGenerator
from ..util import get_array_module, is_xp_array


class Ops:
    name: str = "base"
    xp: Xp = numpy

    def __init__(
        self,
        device_type: DeviceTypes = "cpu",
        device_id: int = -1,
        **kwargs
    ) -> None:
        self.device_type = device_type
        self.device_id = device_id

    def to_numpy(self, data):  # pragma: no cover
        if isinstance(data, numpy.ndarray):
            return data
        else:
            raise ValueError("Cannot convert non-numpy from base Ops class")

    def minibatch(
        self,
        size: Union[int, Generator],
        sequence: Batchable,
        *,
        shuffle: bool = False,
        buffer: int = 1,
    ) -> SizedGenerator:
        """Iterate slices from a sequence, optionally shuffled. Slices
        may be either views or copies of the underlying data.

        The `size` argument may be either an integer, or a sequence of integers.
        If a sequence, a new size is drawn before every output.

        If shuffle is True, shuffled batches are produced by first generating
        an index array, shuffling it, and then using it to slice into the
        sequence.

        An internal queue of `buffer` items is accumulated before being each
        output. Buffering is useful for some devices, to allow the
        network to run asynchronously without blocking on every batch.
        """
        if not hasattr(sequence, "__len__"):
            err = f"Can't minibatch data. Expected sequence, got {type(sequence)}"
            raise ValueError(err)
        sizes = self._get_batch_sizes(
            len(sequence), itertools.repeat(size) if isinstance(size, int) else size
        )
        indices = numpy.arange(len(sequence))

        # This is a bit convoluted, but it's a time where convenience makes
        # trickery worthwhile: instead of being an actual generator, we
        # return our SizedGenerator object, which provides a __len__.
        def _iter_items():
            if shuffle:
                numpy.random.shuffle(indices)
            queue = []
            i = 0
            for size in sizes:
                queue.append(self._get_batch(sequence, indices[i : i + size]))
                if len(queue) >= buffer:
                    yield from queue
                    queue = []
                i += size
            yield from queue

        return SizedGenerator(_iter_items, len(sizes))

    def multibatch(
        self,
        size: Union[int, Generator],
        sequence: Batchable,
        *others: Batchable,
        shuffle: bool = False,
        buffer: int = 1,
    ) -> SizedGenerator:
        """Minibatch one or more sequences of data, and yield
        lists with one batch per sequence. See ops.minibatch.
        """
        # You'd think we could just do this by calling into minibatch and zip...
        # But the shuffling makes it really hard.
        sequences = (sequence,) + tuple(others)
        if not all(hasattr(seq, "__len__") for seq in sequences):
            values = ", ".join([f"{type(seq)}" for seq in sequences])
            err = f"Can't multibatch data. Expected sequences, got {values}"
            raise ValueError(err)
        sizes = self._get_batch_sizes(
            len(sequence), itertools.repeat(size) if isinstance(size, int) else size
        )
        indices = numpy.arange(len(sequence))

        def _iter_items():
            if shuffle:
                numpy.random.shuffle(indices)
            queue = []
            i = 0
            for size in sizes:
                idx_batch = indices[i : i + size]
                queue.append([])
                for sequence in sequences:
                    queue[-1].append(self._get_batch(sequence, idx_batch))
                if len(queue) >= buffer:
                    yield from queue
                    queue = []
                i += size
            yield from queue

        return SizedGenerator(_iter_items, len(sizes))

    def _get_batch(self, sequence, indices):
        if isinstance(sequence, list):
            subseq = [sequence[i] for i in indices]
        elif isinstance(sequence, tuple):
            subseq = tuple(sequence[i] for i in indices)  # type: ignore
        else:
            subseq = sequence[indices]  # type: ignore
        if is_xp_array(subseq):
            subseq = self.as_contig(cast(Array, subseq))  # type: ignore
        return subseq

    def _get_batch_sizes(self, length: int, sizes: Iterator[int]):
        output = []
        i = 0
        while i < length:
            output.append(next(sizes))
            i += output[-1]
        return output

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
        """The reverse/backward operation of the `seq2col` function: calculate
        the gradient of the original `(M, N)` sequence, as a function of the
        gradient of the output `(M, N*(nW*2+1))` sequence.
        """
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
        """Perform General Matrix Multiplication (GeMM) and optionally store
        the result in the specified output variable.
        """
        if trans1:
            x = x.T
        if trans2:
            y = y.T
        if out is None:
            return self.xp.dot(x, y)
        else:
            self.xp.dot(x, y, out=out)
            return out

    # TODO: type without type errors :(
    def affine(self, X, W, b):
        """Apply a weights layer and a bias to some inputs, i.e.
        Y = X @ W.T + b
        """
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
        """Flatten a list of arrays into one large array."""
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
        """The reverse/backward operation of the `flatten` function: unflatten
        a large array into a list of arrays according to the given lengths.
        """
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

    def pad(self, seqs: List[Array2d], round_to=1) -> Array3d:
        """Perform padding on a list of arrays so that they each have the same
        length, by taking the maximum dimension across each axis. This only
        works on non-empty sequences with the same `ndim` and `dtype`.
        """
        # TODO: This should be generalized to handle different ranks
        if not seqs:
            raise ValueError("Cannot pad empty sequence")
        if len(set(seq.ndim for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences with different ndims")
        if len(set(seq.dtype for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences with different dtypes")
        if len(set(seq.shape[1:] for seq in seqs)) != 1:
            raise ValueError("Cannot pad sequences that differ on other dimensions")
        # Find the maximum dimension along each axis. That's what we'll pad to.
        length = max(len(seq) for seq in seqs)
        # Round the length to nearest bucket -- helps on GPU, to make similar
        # array sizes.
        length = (length + (round_to - 1)) // round_to * round_to
        final_shape = (len(seqs), length) + seqs[0].shape[1:]
        output: Array3d = self.alloc(final_shape, dtype=seqs[0].dtype)
        for i, arr in enumerate(seqs):
            output[i, : arr.shape[0]] = arr
        return output

    def unpad(self, padded: Array, lengths: List[int]) -> List[Array]:
        """The reverse/backward operation of the `pad` function: transform an
        array back into a list of arrays, each with their original length.
        """
        output = []
        for i, length in enumerate(lengths):
            output.append(padded[i, :length])
        return output

    def list2padded(self, seqs: List[Array2d]) -> Padded:
        """Pack a sequence of 2d arrays into a Padded datatype."""
        if not seqs:
            return Padded(
                self.alloc_f3d(0, 0, 0),
                self.alloc_i1d(0),
                self.alloc_i1d(0),
                self.alloc_i1d(0),
            )
        elif len(seqs) == 1:
            data = seqs[0].reshape((seqs[0].shape[0], 1) + seqs[0].shape[1:])
            size_at_t: Array1d = self.asarray([1] * data.shape[0], dtype="i")
            lengths: Array1d = self.asarray([data.shape[0]], dtype="i")
            indices: Array1d = self.asarray([0], dtype="i")
            return Padded(data, size_at_t, lengths, indices)
        lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
        lengths_indices.sort(reverse=True)
        indices_ = [i for length, i in lengths_indices]
        lengths_ = [length for length, i in lengths_indices]
        nS = max([len(seq) for seq in seqs])
        arr: Array3d = self.pad(seqs)
        arr = arr.transpose((1, 0, 2))
        # Build a lookup table so we can find how big the batch is at point t.
        batch_size_at_t_ = numpy.zeros(nS, dtype="i")
        batch_size_at_t_ += 1
        i = len(lengths_)
        for t in range(nS):
            if t == lengths_[i - 1]:
                i -= 1
                if i == 0:
                    break
            batch_size_at_t_[t] = i
        return Padded(
            arr,
            self.asarray(batch_size_at_t_),
            self.asarray(lengths_),
            self.asarray(indices_),
        )

    def padded2list(self, padded: Padded) -> List[Array2d]:
        """Unpack a Padded datatype to a list of 2-dimensional arrays."""
        indices = padded.indices
        data = padded.data
        lengths = padded.lengths
        unpadded = [None] * len(lengths)
        data = self.xp.ascontiguousarray(data.transpose((1, 0, 2)))
        for i in range(data.shape[0]):
            unpadded[indices[i]] = data[i, : lengths[i]]
        return cast(List[Array2d], unpadded)

    def get_dropout_mask(self, shape: Shape, drop: float) -> Array:
        """Create a random mask for applying dropout, with a certain percent of
        the mask (defined by `drop`) will contain zeros. The neurons at those
        positions will be deactivated during training, resulting in a more
        robust network and less overfitting.
        """
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
        """Allocate an array of a certain shape."""
        if isinstance(shape, int):
            shape = (shape,)
        return self.xp.zeros(shape, dtype=dtype)

    def unzip(self, data: Tuple[Array, Array]) -> Tuple[Array, Array]:
        """Unzip a tuple of two arrays, transform them with `asarray` and return
        them as two separate arrays.
        """
        X, y = zip(*data)
        return self.asarray(X), self.asarray(y)

    def asarray(
        self,
        data: Union[ArrayT, Sequence[ArrayT], Sequence[int]],
        *,
        dtype: Optional[DTypes] = None,
    ) -> ArrayT:
        """Ensure a given array is of the correct type."""
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
        """Allow the backend to make a contiguous copy of an array.
        Implementations of `Ops` do not have to make a copy or make it
        contiguous if that would not improve efficiency for the execution engine.
        """
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
        summed = self.backprop_reduce_sum(self.reduce_sum(new_x, lengths), lengths)
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
        sum_dX = self.backprop_reduce_sum(self.reduce_sum(dX, lengths), lengths)
        dX -= Y * sum_dX
        return dX

    def recurrent_lstm(
        self,
        W: Array2d,
        b: Array1d,
        h_init: Array1d,
        c_init: Array1d,
        inputs: Array3d,
        is_train: bool = True,
    ) -> Tuple[Array3d, Tuple[Array3d, Array3d, Array3d]]:
        Y, (G, C, S) = recurrent_lstm_forward(W, b, h_init, c_init, inputs)
        return Y, (G, C, S)

    def backprop_recurrent_lstm(
        self,
        dY: Array3d,
        fwd_state: Tuple[Array3d, Array3d, Array3d],
        params: Tuple[Array2d, Array1d],
    ) -> Tuple[Array3d, Tuple[Array2d, Array1d, Array1d, Array1d]]:
        dCt = self.alloc_f2d(dY.shape[1], dY.shape[2])
        empty_row = self.alloc_f3d(1, dY.shape[1], dY.shape[2])
        # Offset dY by 1
        dY = self.xp.vstack((empty_row, dY))
        dW, db, dX, dY, dC0 = backprop_recurrent_lstm(dY, dCt, (fwd_state, params))
        return dX, (dW, db, dY[0].sum(axis=0), dC0.sum(axis=0))

    def maxout(self, X: Array3d) -> Tuple[Array2d, Array2d]:
        which = X.argmax(axis=-1)
        return X.max(axis=-1), which

    def backprop_maxout(self, dY: Array2d, which: Array2d, P: int) -> Array3d:
        dX = self.alloc_f3d(dY.shape[0], dY.shape[1], P)
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
    ) -> Array2d:
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
        # Internals for optimizer
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
        # Internals for optimizer
        mom1 *= beta1
        mom2 *= beta2
        mom1 += gradient * (1.0 - beta1)
        mom2 += gradient * gradient * (1.0 - beta2)
        # Here we assume learn rate is calculated by the caller.
        # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
        weights -= learn_rate * (mom1 / (mod_rate * self.xp.sqrt(mom2) + eps))
        return weights, gradient, mom1, mom2

    def clip_gradient(self, gradient: Array, threshold: float) -> Array:
        # Internals for optimizer
        xp = get_array_module(gradient)
        grad_norm = xp.linalg.norm(gradient)
        if grad_norm >= threshold:
            gradient *= threshold / grad_norm
        return gradient

    def logloss(self, y_true: Array, y_pred: Array) -> float:
        # Currently not used
        log_yp = self.xp.log(y_pred + 1e-8)
        loss = (y_true * log_yp) + (1 - y_true) * self.xp.log((1 - y_pred) + 1e-8)
        return -loss

    def reduce_sum(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].sum(axis=0)
            start += length
        return Y

    def reduce_mean(self, X: Array2d, lengths: Array1d) -> Array2d:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            Y[i] = X[start : start + length].mean(axis=0)
            start += length
        return Y

    def reduce_max(self, X: Array2d, lengths: Array1d) -> Tuple[Array2d, Array2d]:
        Y = self.alloc_f2d(lengths.shape[0], X.shape[1])
        which = self.alloc_i2d(lengths.shape[0], X.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            which[i] = X[start : start + length].argmax(axis=0)
            Y[i] = X[start : start + length].max(axis=0)
            start += length
        return Y, which

    def backprop_reduce_sum(self, d_sums: Array2d, lengths: Array1d) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_sums.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_sums[i]
            start += length
        return dX

    def backprop_reduce_mean(self, d_means: Array2d, lengths: Array1d) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_means.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length] = d_means[i] / length
            start += length
        return dX

    def backprop_reduce_max(
        self, d_maxes: Array2d, which: Array2d, lengths: Array1d
    ) -> Array2d:
        dX = self.alloc_f2d(lengths.sum(), d_maxes.shape[1])
        start = 0
        for i, length in enumerate(lengths):
            dX[start : start + length, which[i]] = d_maxes[i]
            start += length
        return dX

    def hash(self, ids: Array, seed: int) -> Array:
        """Hash a sequence of 64-bit keys into a table with 4 32-bit keys, using
        murmurhash3.
        """
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        return self.asarray(
            numpy_ops.hash(numpy_ops.asarray(ids, dtype="uint64"), seed)
        )

    def ngrams(self, n: int, keys: Array) -> Array:
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        return self.asarray(
            numpy_ops.ngrams(n, numpy_ops.asarray(keys, dtype="uint64"))
        )

    def position_encode(
        self, N: int, D: int, period: int = 10000, out: Optional[Array] = None
    ) -> Array:
        # Currently internals only
        from .numpy_ops import NumpyOps

        numpy_ops = NumpyOps()
        return self.asarray(numpy_ops.position_encode(N, D, period, out))

    def scatter_add(self, table: Array, indices: Array, values: Array) -> Array:
        return self.xp.add.at(table, indices, values)

    def insert_into(self, shape, Xs):
        """Maybe don't need this? Just a quicky to get Jax working."""
        output = self.alloc(shape, dtype=Xs[0].dtype)
        for i, x in enumerate(Xs):
            output[i, : x.shape[0]] = x
        return output


# This code is intentionally almost-duplicate with the Jax one. It's kind
# of hard to condition on jax vs not jax without messing up the jax JIT,
# and we'll want to have a more specialised implementation for non-Jax
# versions. But for now this has been tested and works, so we'll just leave
# it as a reference implementation.
"""
LSTM Notation (kind of involved, but made it a lot easier to write)

X: Inputs
Y: Outputs (aka hiddens)
C: Cells
G: Gates (Output of non-linearity, i.e. lstm_gates(X @ W.T)
A: Activations (X @ W.T, before non-linearity)

Imagine we have the input:
batch = [
    ["apple", "banana", "cantaloupe", "date", "elderberry"],
    ["aardvark", "bat", "capybara", "dingo", "elephant"]
]

The input variable X will have one vector per word, so X[0, 1] will be banana's
vector, X[0, 1, 0] will be a float, the first element of that vector.

We're computing an output variable Y of shape (nL, nB, nO), so that Y[0, 1] is
the output variable of banana.

A problem with variables for RNNs is keeping the timesteps straight. It's hard
to distinguish the current, previous, and next timesteps. To solve this problem,
we follow the convention that **we are at timestep 3**.

Additionally, the variables for Y and C are offset by one, as the 0th elements
have the initial hiddens and initial cells. So:

    t=3
    Xt3: The input vectors for 'dingo' and 'date', i.e. X[t]
    Yt3: The output vectors for 'dingo' and 'date', i.e. Y[t+1] (Y is offset.)
    Ct2: The cells calculated at 'c...', that are the input for 'd...'
    Ct3: The cells calculated at 'd...', that are the input for 'e...'
    At3: The activations at 'd...'
    Gt3: The gates at 'd...'
"""


def recurrent_lstm_forward(W, b, c_init, h_init, X):
    xp = get_array_module(W)
    nL, nB, nI = X.shape
    nO = h_init.shape[0]
    # Preallocate these so we can pass them through for loop.
    Y = xp.zeros((nL + 1, nB, nO), dtype="f")
    G = xp.zeros((nL, nB, nO * 4), dtype="f")
    C = xp.zeros((nL + 1, nB, nO), dtype="f")
    # Set initial hidden and cell states. The Y and C will be shifted 1,
    # so that we can have fewer arrays.
    Y[0] = h_init
    C[0] = c_init
    state = ((W, b, X), (Y, C, G))
    for i in range(X.shape[0]):
        state = lstm_stepper_forward(i, state)
    (W, b, X), (Y, C, G) = state
    # Recall that Y and C are both offset by 1. Y[1] is the output for
    # X[1], while Y[0] was used as an input for Y[1]. We use
    # the S values to backprop the weights, so we need X the previous Ys.
    S = xp.concatenate((X, Y[:-1]), axis=-1)
    return Y[1:], (G, C, S)


def lstm_stepper_forward(t, state):
    (W, b, X), (Y, C, G) = state
    # Get the activations for this timestep.
    At3 = lstm_weights_forward(X[t], Y[t], W, b)
    # The offsets here are a bit unintuitive, because Y and C are 1-offset.
    Ct2 = C[t]
    Yt3, Ct3, Gt3 = lstm_gates_forward(At3, Ct2)
    Y[t + 1] = Yt3
    C[t + 1] = Yt3
    G[t] = Gt3
    return (W, b, X), (Y, C, G)


def backprop_recurrent_lstm(dY, dCt, fwd_vars):
    xp = get_array_module(dY)
    (G, C, S), (W, b) = fwd_vars
    nL = S.shape[0]
    nB = dY.shape[1]
    nI = S.shape[2] - dY.shape[2]
    # Preallocate these so we can pass them through for loop.
    dX = xp.zeros((nL, nB, nI), dtype="f")
    dW = xp.zeros(W.shape, dtype="f")
    db = xp.zeros(b.shape, dtype="f")
    state = (
        (dW, db, dX),  # The gradi-outs (Write-only)
        (dY, dCt),  # The gradi-ins  (Read and write)
        (G, C, S),  # Forward state  (Read-only)
        (W, b),  # Params         (Read-only)
    )
    for t in range(nL - 1, -1, -1):
        state = backprop_lstm_stepper(t, state)
    (dW, db, dX), (dY, dCt), (G, C, S), (W, b) = state
    return dW, db, dX, dY, dCt


def backprop_lstm_stepper(t, state):
    (dW, db, dX), (dY, dCt3), (G, C, S), (W, b) = state
    # Recall, we're at step 3, Y and C are offset by 1. See above.
    dYt3 = dY[t + 1]
    Ct3 = C[t + 1]
    St3 = S[t]
    Gt3 = G[t]
    Ct2 = C[t]
    dAt3, dCt2 = backprop_lstm_gates(dCt3, dYt3, Gt3, Ct3, Ct2)
    dXt3, dYt2, dW3, db3 = backprop_lstm_weights(dAt3, (St3, W, b))
    dX[t] = dXt3
    dY[t] = dYt2
    return (dW + dW3, db + db3, dX), (dY, dCt2), (G, C, S), (W, b)


def lstm_weights_forward(Xt3, Yt2, W, b):
    xp = get_array_module(Yt2)
    St3 = xp.concatenate((Xt3, Yt2), axis=-1)
    At3 = St3 @ W.T + b
    return At3


def backprop_lstm_weights(dAt3, fwd_state):
    St3, W, b = fwd_state
    dW = dAt3.T @ St3
    db = dAt3.sum(axis=0)
    dSt3 = dAt3 @ W
    nO = W.shape[0] // 4
    nI = St3.shape[1] - nO
    dXt3 = dSt3[:, :nI]
    dYt2 = dSt3[:, nI:]
    return dXt3, dYt2, dW, db


def lstm_gates_forward(At3, Ct2):
    xp = get_array_module(At3)
    # hf, hi, ho, hc: Forget, input, output, cell gates.
    At3_hf, At3_hi, At3_ho, At3_hc = xp.split(At3, 4, axis=-1)
    # Number the steps here, to refer back for backward pass.
    # 1. Activations
    hf = sigmoid(At3_hf)  # 1a
    hi = sigmoid(At3_hi)  # 1b
    ho = sigmoid(At3_ho)  # 1c
    hc = xp.tanh(At3_hc)  # 1d

    Ct3 = hf * Ct2  # 2a
    Ct3 += hi * hc  # 2b
    tanhCt3 = xp.tanh(Ct3)  # 3a
    Yt3 = tanhCt3 * ho  # 3b
    # We don't need the gradient for this, it's just for backprop calculation.
    Gt3 = xp.concatenate((hf, hi, ho, hc), axis=-1)
    return Yt3, Ct3, Gt3


def backprop_lstm_gates(
    dYt3: Array2d, dCt3: Array2d, Gt3: Array2d, Ct3: Array2d, Ct2: Array2d
) -> Tuple[Array3d, Array2d]:
    # See above for notation. Step numbering refers to forward_lstm_gates
    xp = get_array_module(dYt3)
    hf, hi, ho, hc = xp.split(Gt3, 4, axis=-1)
    tanhCt3 = xp.tanh(Ct3)
    # 3b: Yt3 = tanhCt3 * ho
    d_ho = dYt3 * tanhCt3
    d_tanhCt3 = dYt3 * ho
    # 3a: tanhCt3 = tanh(Ct3)
    dCt3 += d_tanhCt3 * dtanh(tanhCt3)
    # 2b: Ct3 += hi * hc
    d_hi = dCt3 * hc
    d_hc = dCt3 * hi
    # 2a: Ct3 = hf * Ct2
    d_hf = dCt3 * Ct2
    dCt2 = dCt3 * hf
    d_At3_hc = d_hc * dtanh(hc)  # 1d
    d_At3_ho = d_ho * dsigmoid(ho)  # 1c
    d_At3_hi = d_hi * dsigmoid(hi)  # 1b
    d_At3_hf = d_hf * dsigmoid(hf)  # 1a
    dAt3 = xp.concatenate((d_At3_hf, d_At3_hi, d_At3_ho, d_At3_hc), axis=-1)
    return dAt3, dCt2


def sigmoid(X):
    xp = get_array_module(X)
    return 1.0 / (1.0 + xp.exp(-X))


def dsigmoid(Y: ArrayT) -> ArrayT:
    return Y * (1.0 - Y)


def dtanh(Y: ArrayT) -> ArrayT:
    return 1 - Y ** 2
