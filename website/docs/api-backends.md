---
title: Backends & Math
next: /docs/api-util
---

All Thinc models have a reference to an `Ops` instance, that provides access to
**memory allocation** and **mathematical routines**. The `Model.ops` instance
also keeps track of state and settings, so that you can have different models in
your network executing on different devices or delegating to different
underlying libraries.

Each `Ops` instance holds a reference to a numpy-like module (`numpy` or
`cupy`), which you can access at `Model.ops.xp`. This is enough to make most
layers work on **both CPU and GPU devices**. Additionally, there are several
routines that we have implemented as methods on the `Ops` object, so that
specialized versions can be called for different backends. You can also create
your own `Ops` subclasses with specialized routines for your layers, and use the
[`set_current_ops`](#set_current_ops) function to change the default.

| Backend    |        CPU         |        GPU         |        TPU        | Description                                                                                           |
| ---------- | :----------------: | :----------------: | :---------------: | ----------------------------------------------------------------------------------------------------- |
| `NumpyOps` | <i name="yes"></i> | <i name="no"></i>  | <i name="no"></i> | Execute via `numpy`, [`blis`](https://github.com/explosion/cython-blis) (optional) and custom Cython. |
| `CupyOps`  | <i name="no"></i>  | <i name="yes"></i> | <i name="no"></i> | Execute via [`cupy`](https://cupy.chainer.org/) and custom CUDA.                                      |

## Ops {#ops tag="class"}

The `Ops` class is typically not used directly but via `NumpyOps` or `CupyOps`,
which are subclasses of `Ops` and implement a **more efficient subset of the
methods**. You also have access to the ops via the
[`Model.ops`](/docs/api-model#attributes) attribute. The documented methods
below list which backends provide optimized and more efficient versions
(indicated by <i name="yes"></i>), and which use the default implementation.
Thinc also provides various [helper functions](#util) for getting and setting
different backends.

<infobox variant="warning">

The current set of implemented methods is somewhat arbitrary and **subject to
change**. Methods are moved to the `Ops` object if we want different
implementations for different backends, e.g. cythonized CPU versions or custom
CUDA kernels.

</infobox>

```python
### Example
from thinc.api import Linear, get_ops, use_ops

model = Linear(4, 2)
X = model.ops.alloc2f(10, 2)
blis_ops = get_ops("numpy", use_blis=True)
use_ops(blis_ops)
```

### Attributes {#attributes}

| Name          | Type         | Description                                                                              |
| ------------- | ------------ | ---------------------------------------------------------------------------------------- |
| `name`        | <tt>str</tt> | **Class attribute:** Backend name, `"numpy"` or `"cupy"`.                                |
| `xp`          | <tt>Xp</tt>  | **Class attribute:** `numpy` or `cupy`.                                                  |
| `device_type` | <tt>str</tt> | The device type to use, if available for the given backend: `"cpu"`, `"gpu"` or `"tpu"`. |
| `device_id`   | <tt>int</tt> | The device ID to use, if available for the given backend.                                |

### Ops.\_\_init\_\_ {#init tag="method"}

| Argument       | Type          | Description                                                                                                   |
| -------------- | ------------- | ------------------------------------------------------------------------------------------------------------- |
| `device_type`  | <tt>str</tt>  | The device type to use, if available for the given backend: `"cpu"`, `"gpu"` or `"tpu"`.                      |
| `device_id`    | <tt>int</tt>  | The device ID to use, if available for the given backend.                                                     |
| _keyword-only_ |               |                                                                                                               |
| `use_blis`     | <tt>bool</tt> | `NumpyOps`: Use [`blis`](https://github.com/explosion/cython-blis) for single-threaded matrix multiplication. |

### Ops.minibatch {#minibatch tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Iterate slices from a sequence, optionally shuffled. Slices may be either views
or copies of the underlying data. Supports the batchable data types
[`Pairs`](/docs/api-types#pairs), [`Ragged`](/docs/api-types#ragged) and
[`Padded`](/docs/api-types#padded), as well as arrays, lists and tuples. The
`size` argument may be either an integer, or a sequence of integers. If a
sequence, a new size is drawn before every output. If `shuffle` is `True`,
shuffled batches are produced by first generating an index array, shuffling it,
and then using it to slice into the sequence. An internal queue of `buffer`
items is accumulated before being each output. Buffering is useful for some
devices, to allow the network to run asynchronously without blocking on every
batch.

The method returns a [`SizedGenerator`](/docs/api-types#sizedgenerator) that
exposes a `__len__` and is rebatched and reshuffled every time it's executed,
allowing you to move the batching outside of the training loop.

```python
### Example
batches = model.ops.minibatch(128, train_X, shuffle=True)
```

| Argument       | Type                           | Description                                                        |
| -------------- | ------------------------------ | ------------------------------------------------------------------ |
| `size`         | <tt>Union[int, Generator]</tt> | The batch size(s).                                                 |
| `sequence`     | <tt>Batchable</tt>             | The sequence to batch.                                             |
| _keyword-only_ |                                |                                                                    |
| `shuffle`      | <tt>bool</tt>                  | Whether to shuffle the items.                                      |
| `buffer`       | <tt>int</tt>                   | Number of items to accumulate before each output. Defaults to `1`. |
| **RETURNS**    | <tt>SizedGenerator</tt>        | The batched items.                                                 |

### Ops.multibatch {#multibatch tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Minibatch one or more sequences of data, and return lists with one batch per
sequence. Otherwise identical to [`Ops.minibatch`](#minibatch).

```python
### Example
batches = model.ops.multibatch(128, train_X, train_Y, shuffle=True)
```

| Argument       | Type                           | Description                                                        |
| -------------- | ------------------------------ | ------------------------------------------------------------------ |
| `size`         | <tt>Union[int, Generator]</tt> | The batch size(s).                                                 |
| `sequence`     | <tt>Batchable</tt>             | The sequence to batch.                                             |
| `*other`       | <tt>Batchable</tt>             | The other sequences to batch.                                      |
| _keyword-only_ |                                |                                                                    |
| `shuffle`      | <tt>bool</tt>                  | Whether to shuffle the items.                                      |
| `buffer`       | <tt>int</tt>                   | Number of items to accumulate before each output. Defaults to `1`. |
| **RETURNS**    | <tt>SizedGenerator</tt>        | The batched items.                                                 |

### Ops.seq2col {#seq2col tag="method"}

<inline-list>

- **default:** <i name="yes"></i> (`nW=1` only)
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Given an `(M, N)` sequence of vectors, return an `(M, N*(nW*2+1))` sequence. The
new sequence is constructed by concatenating `nW` preceding and succeeding
vectors onto each column in the sequence, to extract a window of features.

| Argument    | Type              | Description                                                       |
| ----------- | ----------------- | ----------------------------------------------------------------- |
| `seq`       | <tt>Floats2d</tt> | The original sequence .                                           |
| `nW`        | <tt>int</tt>      | The window size.                                                  |
| **RETURNS** | <tt>Floats2d</tt> | The created sequence containing preceding and succeeding vectors. |

### Ops.backprop_seq2col {#backprop_seq2col tag="method"}

<inline-list>

- **default:** <i name="yes"></i> (`nW=1` only)
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

The reverse/backward operation of the `seq2col` function: calculate the gradient
of the original `(M, N)` sequence, as a function of the gradient of the output
`(M, N*(nW*2+1))` sequence.

| Argument    | Type              | Description                        |
| ----------- | ----------------- | ---------------------------------- |
| `dY`        | <tt>Floats2d</tt> | Gradient of the output sequence.   |
| `nW`        | <tt>int</tt>      | The window size.                   |
| **RETURNS** | <tt>Floats2d</tt> | Gradient of the original sequence. |

### Ops.gemm {#gemm tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Perform General Matrix Multiplication (GeMM) and optionally store the result in
the specified output variable.

| Argument    | Type                        | Description                                                   |
| ----------- | --------------------------- | ------------------------------------------------------------- |
| `x`         | <tt>Floats2d</tt>           | First array.                                                  |
| `y`         | <tt>Floats2d</tt>           | Second array.                                                 |
| `out`       | <tt>Optional[Floats2d]</tt> | Variable to store the result of the matrix multiplication in. |
| `trans1`    | <tt>bool</tt>               | Whether or not to transpose array `x`.                        |
| `trans2`    | <tt>bool</tt>               | Whether or not to transpose array `y`.                        |
| **RETURNS** | <tt>Floats2d</tt>           | The result of the matrix multiplication.                      |

### Ops.affine {#affine tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Apply a weights layer and a bias to some inputs, i.e. `Y = X @ W.T + b`.

| Argument    | Type              | Description      |
| ----------- | ----------------- | ---------------- |
| `X`         | <tt>Floats2d</tt> | The inputs.      |
| `W`         | <tt>Floats2d</tt> | The weights.     |
| `b`         | <tt>Floats1d</tt> | The bias vector. |
| **RETURNS** | <tt>Floats2d</tt> | The output.      |

### Ops.flatten {#flatten tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Flatten a list of arrays into one large array.

| Argument        | Type                       | Description                                                   |
| --------------- | -------------------------- | ------------------------------------------------------------- |
| `X`             | <tt>Sequence[ArrayXd]</tt> | The original list of arrays.                                  |
| `dtype`         | <tt>Optional[DTypes]</tt>  | The data type to cast the resulting array in.                 |
| `pad`           | <tt>int</tt>               | The number of zeros to add as padding to `X` (default 0).     |
| `ndim_if_empty` | <tt>int</tt>               | The dimension of the output result if `X` is `None` or empty. |
| **RETURNS**     | <tt>ArrayXd</tt>           | One large array storing all original information.             |

### Ops.unflatten {#unflatten tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

The reverse/backward operation of the `flatten` function: unflatten a large
array into a list of arrays according to the given lengths.

| Argument    | Type                   | Description                                                           |
| ----------- | ---------------------- | --------------------------------------------------------------------- |
| `X`         | <tt>ArrayXd</tt>       | The flattened array.                                                  |
| `lengths`   | <tt>Ints1d</tt>        | The lengths of the original arrays before they were flattened.        |
| `pad`       | <tt>int</tt>           | The padding that was applied during the `flatten` step (default 0).   |
| **RETURNS** | <tt>List[ArrayXd]</tt> | A list of arrays storing the same information as the flattened array. |

### Ops.pad {#pad tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Perform padding on a list of arrays so that they each have the same length, by
taking the maximum dimension across each axis. This only works on non-empty
sequences with the same `ndim` and `dtype`.

| Argument    | Type                   | Description                                                                                      |
| ----------- | ---------------------- | ------------------------------------------------------------------------------------------------ |
| `seqs`      | <tt>List[Array2d]</tt> | The sequences to pad.                                                                            |
| `round_to`  | <tt>int</tt>           | Round the length to nearest bucket (helps on GPU, to make similar array sizes). Defaults to `1`. |
| **RETURNS** | <tt>Array3d</tt>       | The padded sequences, stored in one array.                                                       |

### Ops.unpad {#unpad tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

The reverse/backward operation of the `pad` function: transform an array back
into a list of arrays, each with their original length.

| Argument    | Type                   | Description                                     |
| ----------- | ---------------------- | ----------------------------------------------- |
| `padded`    | <tt>ArrayXd</tt>       | The padded sequences, stored in one array.      |
| `lengths`   | <tt>List[int]</tt>     | The original lengths of the unpadded sequences. |
| **RETURNS** | <tt>List[ArrayXd]</tt> | The unpadded sequences.                         |

### Ops.list2padded {#list2padded tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Pack a sequence of two-dimensional arrays into a
[`Padded`](/docs/api-types#padded) datatype.

| Argument    | Type                   | Description            |
| ----------- | ---------------------- | ---------------------- |
| `seqs`      | <tt>List[Array2d]</tt> | The sequences to pack. |
| **RETURNS** | <tt>Padded</tt>        | The packed arrays.     |

### Ops.padded2list {#padded2list tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Unpack a [`Padded`](/docs/api-types#padded) datatype to a list of
two-dimensional arrays.

| Argument    | Type                   | Description             |
| ----------- | ---------------------- | ----------------------- |
| `padded`    | <tt>Padded</tt>        | The object to unpack.   |
| **RETURNS** | <tt>List[Array2d]</tt> | The unpacked sequences. |

### Ops.get_dropout_mask {#get_dropout_mask tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Create a random mask for applying dropout, with a certain percent of the mask
(defined by `drop`) will contain zeros. The neurons at those positions will be
deactivated during training, resulting in a more robust network and less
overfitting.

| Argument    | Type                     | Description                                                |
| ----------- | ------------------------ | ---------------------------------------------------------- |
| `shape`     | <tt>Shape</tt>           | The input shape.                                           |
| `drop`      | <tt>Optional[float]</tt> | The dropout rate.                                          |
| **RETURNS** | <tt>Floats</tt>          | A mask specifying a 0 were a neuron should be deactivated. |

### Ops.alloc {#alloc tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** default

</inline-list>

Allocate an array of a certain shape. If possible, you should always use the
**type-specific methods** listed below, as they make the code more readable and
allow more sophisticated static [type checking](/docs/usage-type-checking) of
the inputs and outputs.

| Argument       | Type             | Description                                                     |
| -------------- | ---------------- | --------------------------------------------------------------- |
| `shape`        | <tt>Shape</tt>   | The shape.                                                      |
| _keyword-only_ |                  |                                                                 |
| `dtype`        | <tt>DTypes</tt>  | The data type (default: `float32`).                             |
| **RETURNS**    | <tt>ArrayXd</tt> | An array of the correct shape and data type, filled with zeros. |

#### Type-specific methods

<inline-list>

- **Floats:** `Ops.alloc_f`, `Ops.alloc1f`, `Ops.alloc2f`, `Ops.alloc3f`,
  `Ops.alloc4f`
- **Ints:** `Ops.alloc_i`, `Ops.alloc1i`, `Ops.alloc2i`, `Ops.alloc3i`,
  `Ops.alloc4i`

</inline-list>

Shortcuts to allocate an array of a certain shape and data type (`f` refers to
`float32` and `i` to `int32`). For instance, `Ops.alloc2f` will allocate an
two-dimensional array of floats.

```python
### Example
X = model.ops.alloc2f(10, 2)  # Floats2d
Y = model.ops.alloc1i(4)  # Ints1d
```

| Argument       | Type                                      | Description                                                                |
| -------------- | ----------------------------------------- | -------------------------------------------------------------------------- |
| `*shape`       | <tt>int</tt>                              | The shape, one positional argument per dimension.                          |
| _keyword-only_ |                                           |                                                                            |
| `dtype`        | <tt>DTypesInt</tt> / <tt>DTypesFloat</tt> | The data type (float type for float methods and int type for int methods). |
| **RETURNS**    | <tt>ArrayXd</tt>                          | An array of the correct shape and data type, filled with zeros.            |

### Ops.reshape {#reshape tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Reshape an array and return an array containing the same data with the given
shape. If possible, you should always use the **type-specific methods** listed
below, as they make the code more readable and allow more sophisticated static
[type checking](/docs/usage-type-checking) of the inputs and outputs.

| Argument    | Type             | Description           |
| ----------- | ---------------- | --------------------- |
| `array`     | <tt>ArrayXd</tt> | The array to reshape. |
| `shape`     | <tt>Shape</tt>   | The shape.            |
| **RETURNS** | <tt>ArrayXd</tt> | The reshaped array.   |

#### Type-specific methods

<inline-list>

- **Floats:** `Ops.reshape_f`, `Ops.reshape1f`, `Ops.reshape2f`,
  `Ops.reshape3f`, `Ops.reshape4f`
- **Ints:** `Ops.reshape_i`, `Ops.reshape1i`, `Ops.reshape2i`, `Ops.reshape3i`,
  `Ops.reshape4i`

</inline-list>

Shortcuts to reshape an array of a certain shape and data type (`f` refers to
`float32` and `i` to `int32`). For instance, `reshape2f` can be used to reshape
an array of floats to a 2d-array of floats.

<infobox variant="warning">

Note that the data type-specific methods mostly exist for **static type checking
purposes**. They do **not** change the data type of the array. For example,
`Ops.reshape2f` expects an array of floats and expects to return an array of
floats – but it won't convert an array of ints to an array of floats. However,
using the specific method will tell the static type checker what array to
expect, and passing in an array thats _typed_ as an int array will result in a
type error.

</infobox>

```python
### Example {small="true"}
X = model.ops.reshape2f(X, 10, 2)  # Floats2d
Y = model.ops.reshape1i(Y, 4)  # Ints1d
```

| Argument    | Type             | Description                                                    |
| ----------- | ---------------- | -------------------------------------------------------------- |
| `array`     | <tt>ArrayXd</tt> | The array to reshape (of the same data type).                  |
| `*shape`    | <tt>int</tt>     | The shape, one positional argument per dimension.              |
| **RETURNS** | <tt>ArrayXd</tt> | The reshaped array (of the same data type as the input array). |

### Ops.asarray {#asarray tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Ensure a given array is of the correct type, e.g. `numpy.ndarray` for `NumpyOps`
or `cupy.ndarray` for `CupyOps`. If possible, you should always use the
**type-specific methods** listed below, as they make the code more readable and
allow more sophisticated static [type checking](/docs/usage-type-checking) of
the inputs and outputs.

| Argument       | Type                                                      | Description                                |
| -------------- | --------------------------------------------------------- | ------------------------------------------ |
| `data`         | <tt>Union[ArrayXd, Sequence[ArrayXd], Sequence[int]]</tt> | The original array.                        |
| _keyword-only_ |                                                           |                                            |
| `dtype`        | <tt>Optional[DTypes]</tt>                                 | The data type                              |
| **RETURNS**    | <tt>ArrayXd</tt>                                          | The array transformed to the correct type. |

### Type-specific methods

<inline-list>

- **Floats:** `Ops.asarray_f`, `Ops.asarray1f`, `Ops.asarray2f`,
  `Ops.asarray3f`, `Ops.asarray4f`
- **Ints:** `Ops.asarray_i`, `Ops.asarray1i`, `Ops.asarray2i`, `Ops.asarray3i`,
  `Ops.asarray4i`

</inline-list>

Shortcuts for specific dimensions and data types (`f` refers to `float32` and
`i` to `int32`). For instance, `Ops.asarray2f` will return a two-dimensional
array of floats.

```python
### Example
X = model.ops.asarray2f(X, 10, 2)  # Floats2d
Y = model.ops.asarray1i(Y, 4)  # Ints1d
```

| Argument       | Type                                      | Description                                                                |
| -------------- | ----------------------------------------- | -------------------------------------------------------------------------- |
| `*shape`       | <tt>int</tt>                              | The shape, one positional argument per dimension.                          |
| _keyword-only_ |                                           |                                                                            |
| `dtype`        | <tt>DTypesInt</tt> / <tt>DTypesFloat</tt> | The data type (float type for float methods and int type for int methods). |
| **RETURNS**    | <tt>ArrayXd</tt>                          | An array of the correct shape and data type, filled with zeros.            |

### Ops.as_contig {#as_contig tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Allow the backend to make a contiguous copy of an array. Implementations of
`Ops` do not have to make a copy or make it contiguous if that would not improve
efficiency for the execution engine.

| Argument       | Type                      | Description                                   |
| -------------- | ------------------------- | --------------------------------------------- |
| `data`         | <tt>ArrayXd</tt>          | The array.                                    |
| _keyword-only_ |                           |                                               |
| `dtype`        | <tt>Optional[DTypes]</tt> | The data type                                 |
| **RETURNS**    | <tt>ArrayXd</tt>          | An array with the same contents as the input. |

### Ops.unzip {#unzip tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Unzip a tuple of two arrays, transform them with `asarray` and return them as
two separate arrays.

| Argument    | Type                             | Description                                 |
| ----------- | -------------------------------- | ------------------------------------------- |
| `data`      | <tt>Tuple[ArrayXd, ArrayXd]      | The tuple of two arrays.                    |
| **RETURNS** | <tt>Tuple[ArrayXd, ArrayXd]</tt> | The two arrays, transformed with `asarray`. |

### Ops.sigmoid {#sigmoid tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Calculate the sigmoid function.

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `X`            | <tt>FloatsXd</tt> | The input values.                              |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| **RETURNS**    | <tt>FloatsXd</tt> | The output values, i.e. `S(X)`.                |

### Ops.dsigmoid {#dsigmoid tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Calculate the derivative of the `sigmoid` function.

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `Y`            | <tt>FloatsXd</tt> | The input values.                              |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| **RETURNS**    | <tt>FloatsXd</tt> | The output values, i.e. `dS(Y)`.               |

### Ops.dtanh {#dtanh tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Calculate the derivative of the `tanh` function.

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `Y`            | <tt>FloatsXd</tt> | The input values.                              |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| **RETURNS**    | <tt>FloatsXd</tt> | The output values, i.e. `dtanh(Y)`.            |

### Ops.softmax {#softmax tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Calculate the softmax function. The resulting array will sum up to 1.

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `x`            | <tt>FloatsXd</tt> | The input values.                              |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| `axis`         | <tt>int</tt>      | The dimension to normalize over.               |
| **RETURNS**    | <tt>FloatsXd</tt> | The normalized output values.                  |

### Ops.backprop_softmax {#backprop_softmax tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

| Argument       | Type              | Description                             |
| -------------- | ----------------- | --------------------------------------- |
| `Y`            | <tt>FloatsXd</tt> | Output array.                           |
| `dY`           | <tt>FloatsXd</tt> | Gradients of the output array.          |
| _keyword-only_ |                   |                                         |
| `axis`         | <tt>int</tt>      | The dimension that was normalized over. |
| **RETURNS**    | <tt>FloatsXd</tt> | The gradients of the input array.       |

### Ops.softmax_sequences {#softmax_sequences tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `Xs`           | <tt>Floats2d</tt> | An 2d array of input sequences.                |
| `lengths`      | <tt>Ints1d</tt>   | The lengths of the input sequences.            |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| `axis`         | <tt>int</tt>      | The dimension to normalize over.               |
| **RETURNS**    | <tt>Floats2d</tt> | The normalized output values.                  |

### Ops.backprop_softmax_sequences {#backprop_softmax_sequences tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

The reverse/backward operation of the `softmax` function.

| Argument    | Type              | Description                           |
| ----------- | ----------------- | ------------------------------------- |
| `dY`        | <tt>Floats2d</tt> | Gradients of the output array.        |
| `Y`         | <tt>Floats2d</tt> | Output array.                         |
| `lengths`   | <tt>Ints1d</tt>   | The lengths of the input sequences.   |
| **RETURNS** | <tt>Floats2d</tt> | The gradients of the input sequences. |

### Ops.recurrent_lstm {#recurrent_lstm tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Encode a padded batch of inputs into a padded batch of outputs using an LSTM.

| Argument    | Type                                                          | Description                                                                                                                               |
| ----------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `W`         | <tt>Floats2d</tt>                                             | The weights, shaped `(nO * 4, nO + nI)`.                                                                                                  |
| `b`         | <tt>Floats1d</tt>                                             | The bias vector, shaped `(nO * 4,)`.                                                                                                      |
| `h_init`    | <tt>Floats1d</tt>                                             | Initial value for the previous hidden vector.                                                                                             |
| `c_init`    | <tt>Floats1d</tt>                                             | Initial value for the previous cell state.                                                                                                |
| `inputs`    | <tt>Floats3d</tt>                                             | A batch of inputs, shaped `(nL, nB, nI)`, where `nL` is the sequence length and `nB` is the batch size.                                   |
| `is_train`  | <tt>bool</tt>                                                 | Whether the model is running in a training context.                                                                                       |
| **RETURNS** | <tt>Tuple[Floats3d, Tuple[Floats3d, Floats3d, Floats3d]]</tt> | A tuple consisting of the outputs and the intermediate activations required for the backward pass. The outputs are shaped `(nL, nB, nO)`. |

### Ops.backprop_recurrent_lstm {#backprop_recurrent_lstm tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** default
- **cupy:** default

</inline-list>

Compute the gradients for the `recurrent_lstm` operation via backpropagation.

| Argument    | Type                                                                    | Description                                                                                           |
| ----------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `dY`        | <tt>Floats3d</tt>                                                       | The gradient w.r.t. the outputs.                                                                      |
| `fwd_state` | <tt>Tuple[Floats3d, Floats3d, Floats3d]</tt>                            | The tuple of gates, cells and inputs, returned by the forward pass.                                   |
| `params`    | <tt>Tuple[Floats2d, Floats1d]</tt>                                      | A tuple of the weights and biases.                                                                    |
| **RETURNS** | <tt>Tuple[Floats3d, Tuple[Floats2d, Floats1d, Floats1d, Floats1d]]</tt> | The gradients for the inputs and parameters (the weights, biases, initial hiddens and initial cells). |

### Ops.maxout {#maxout tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

| Argument    | Type                             | Description                                                                     |
| ----------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `X`         | <tt>Floats3d</tt>                | The inputs.                                                                     |
| **RETURNS** | <tt>Tuple[Floats2d, Ints2d]</tt> | The outputs and an array indicating which elements in the final axis were used. |

### Ops.backprop_maxout {#backprop_maxout tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

| Argument    | Type              | Description                                 |
| ----------- | ----------------- | ------------------------------------------- |
| `dY`        | <tt>Floats2d</tt> | Gradients of the output array.              |
| `which`     | <tt>Ints2d</tt>   | The positions selected in the forward pass. |
| `P`         | <tt>int</tt>      | The size of the final dimension.            |
| **RETURNS** | <tt>Floats3d</tt> | The gradient of the inputs.                 |

### Ops.relu {#relu tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `X`            | <tt>Floats2d</tt> | The inputs.                                    |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| **RETURNS**    | <tt>Floats2d</tt> | The outputs.                                   |

### Ops.backprop_relu {#relu tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

| Argument       | Type              | Description                                    |
| -------------- | ----------------- | ---------------------------------------------- |
| `dY`           | <tt>Floats2d</tt> | Gradients of the output array.                 |
| `Y`            | <tt>Floats2d</tt> | The output from the forward pass.              |
| _keyword-only_ |                   |                                                |
| `inplace`      | <tt>bool</tt>     | If `True`, the array may be modified in place. |
| **RETURNS**    | <tt>Floats2d</tt> | The gradient of the input.                     |

### Ops.mish {#mish tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Compute the Mish activation
([Misra, 2019](https://arxiv.org/pdf/1908.08681.pdf)).

| Argument    | Type              | Description                                     |
| ----------- | ----------------- | ----------------------------------------------- |
| `X`         | <tt>Floats2d</tt> | The inputs.                                     |
| `threshold` | <tt>float</tt>    | Maximum value at which to apply the activation. |
| **RETURNS** | <tt>Floats2d</tt> | The outputs.                                    |

### Ops.backprop_mish {#backprop_softplus tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Backpropagate the Mish activation
([Misra, 2019](https://arxiv.org/pdf/1908.08681.pdf)).

| Argument    | Type              | Description                      |
| ----------- | ----------------- | -------------------------------- |
| `dY`        | <tt>Floats2d</tt> | Gradients of the output array.   |
| `X`         | <tt>Floats2d</tt> | The inputs to the forward pass.  |
| `threshold` | <tt>float</tt>    | Threshold from the forward pass. |
| **RETURNS** | <tt>Floats2d</tt> | The gradient of the input.       |

### Ops.reduce_sum {#reduce_sum tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Perform sequence-wise summation for data in the ragged format.

| Argument    | Type              | Description                   |
| ----------- | ----------------- | ----------------------------- |
| `X`         | <tt>Floats2d</tt> | The concatenated sequences.   |
| `lengths`   | <tt>Ints1d</tt>   | The sequence lengths.         |
| **RETURNS** | <tt>Floats2d</tt> | The sequence-wise summations. |

### Ops.backprop_reduce_sum {#backprop_reduce_sum tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Backpropagate the `reduce_sum` operation.

| Argument    | Type              | Description                                 |
| ----------- | ----------------- | ------------------------------------------- |
| `d_sums`    | <tt>Floats2d</tt> | The gradient of the outputs.                |
| `lengths`   | <tt>Ints1d</tt>   | The sequence lengths.                       |
| **RETURNS** | <tt>Floats2d</tt> | The gradient of the concatenated sequences. |

### Ops.reduce_mean {#reduce_mean tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Perform sequence-wise averaging for data in the ragged format.

| Argument    | Type              | Description                 |
| ----------- | ----------------- | --------------------------- |
| `X`         | <tt>Floats2d</tt> | The concatenated sequences. |
| `lengths`   | <tt>Ints1d</tt>   | The sequence lengths.       |
| **RETURNS** | <tt>Floats2d</tt> | The sequence-wise averages. |

### Ops.backprop_reduce_mean {#backprop_reduce_mean tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Backpropagate the `reduce_mean` operation.

| Argument    | Type              | Description                                 |
| ----------- | ----------------- | ------------------------------------------- |
| `d_means`   | <tt>Floats2d</tt> | The gradient of the outputs.                |
| `lengths`   | <tt>Ints1d</tt>   | The sequence lengths.                       |
| **RETURNS** | <tt>Floats2d</tt> | The gradient of the concatenated sequences. |

### Ops.reduce_max {#reduce_max tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Perform sequence-wise max pooling for data in the ragged format.

| Argument    | Type                             | Description                 |
| ----------- | -------------------------------- | --------------------------- |
| `X`         | <tt>Floats2d</tt>                | The concatenated sequences. |
| `lengths`   | <tt>Ints1d</tt>                  | The sequence lengths.       |
| **RETURNS** | <tt>Tuple[Floats2d, Ints2d]</tt> | The sequence-wise maximums. |

### Ops.backprop_reduce_max {#backprop_reduce_max tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Backpropagate the `reduce_max` operation.

| Argument    | Type              | Description                                 |
| ----------- | ----------------- | ------------------------------------------- |
| `d_maxes`   | <tt>Floats2d</tt> | The gradient of the outputs.                |
| `which`     | <tt>Ints2d</tt>   | The indices selected.                       |
| `lengths`   | <tt>Ints1d</tt>   | The sequence lengths.                       |
| **RETURNS** | <tt>Floats2d</tt> | The gradient of the concatenated sequences. |

### Ops.hash {#hash tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Hash a sequence of 64-bit keys into a table with four 32-bit keys, using
`murmurhash3`.

| Argument    | Type            | Description                         |
| ----------- | --------------- | ----------------------------------- |
| `ids`       | <tt>Ints1d</tt> | The keys, 64-bit unsigned integers. |
| `seed`      | <tt>int</tt>    | The hashing seed.                   |
| **RETURNS** | <tt>Ints2d</tt> | The hashes.                         |

### Ops.ngrams {#ngrams tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** default

</inline-list>

Create hashed ngram features.

| Argument    | Type            | Description                                |
| ----------- | --------------- | ------------------------------------------ |
| `n`         | <tt>int</tt>    | The window to calculate each feature over. |
| `keys`      | <tt>Ints1d</tt> | The input sequence.                        |
| **RETURNS** | <tt>Ints1d</tt> | The hashed ngrams.                         |

### Ops.scatter_add {#scatter_add tag="method"}

<inline-list>

- **default:** <i name="yes"></i>
- **numpy:** <i name="yes"></i>
- **cupy:** <i name="yes"></i>

</inline-list>

Increment entries in the array out using the indices in `ids` and the values in
`inputs`.

| Argument    | Type              | Description             |
| ----------- | ----------------- | ----------------------- |
| `table`     | <tt>FloatsXd</tt> | The array to increment. |
| `indices`   | <tt>IntsXd</tt>   | The indices to use.     |
| `values`    | <tt>FloatsXd</tt> | The inputs.             |
| **RETURNS** | <tt>FloatsXd</tt> | The incremented array.  |

---

## Utilities {#util}

### get_ops {#get_ops tag="function"}

Get a backend object using a string name.

```python
### Example
from thinc.api import get_ops

numpy_ops = get_ops("numpy")
```

| Argument    | Type         | Description                                           |
| ----------- | ------------ | ----------------------------------------------------- |
| `ops`       | <tt>str</tt> | `"numpy"` or `"cupy"`.                                |
| `**kwargs`  |              | Optional arguments passed to [`Ops.__init__`](#init). |
| **RETURNS** | <tt>Ops</tt> | The backend object.                                   |

### use_ops {#use_ops tag="contextmanager"}

Change the backend to execute with for the scope of the block.

```python
### Example
from thinc.api import use_ops, get_current_ops

with use_ops("cupy"):
    current_ops = get_current_ops()
    assert current_ops.name == "cupy"
```

| Argument   | Type         | Description                                           |
| ---------- | ------------ | ----------------------------------------------------- |
| `ops`      | <tt>str</tt> | `"numpy"` or `"cupy"`.                       |
| `**kwargs` |              | Optional arguments passed to [`Ops.__init__`](#init). |

### get_current_ops {#get_current_ops tag="function"}

Get the current backend object.

| Argument    | Type         | Description                 |
| ----------- | ------------ | --------------------------- |
| **RETURNS** | <tt>Ops</tt> | The current backend object. |

### set_current_ops {#set_current_ops tag="function"}

Set the current backend object.

| Argument | Type         | Description         |
| -------- | ------------ | ------------------- |
| `ops`    | <tt>Ops</tt> | The backend object. |
