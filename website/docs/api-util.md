---
title: Utilities & Extras
teaser: Helpers and utility functions
---

### fix_random_seed {#fix_random_seed tag="function"}

Set the random seed for `random`, `numpy.random` and `cupy.random` (if
available). Should be called at the top of a file or function.

```python
### Example
from thinc.api import fix_random_seed
fix_random_seed(0)
```

| Argument | Type         | Description                |
| -------- | ------------ | -------------------------- |
| `seed`   | <tt>int</tt> | The seed. Defaults to `0`. |

### prefer_gpu {#prefer_gpu tag="function"}

Allocate data and perform operations on GPU, if available. If data has already
been allocated on CPU, it will not be moved. Ideally, this function should be
called right after importing Thinc.

```python
### Example
from thinc.api import prefer_gpu
is_gpu = prefer_gpu()
```

| Argument    | Type          | Description                              |
| ----------- | ------------- | ---------------------------------------- |
| `gpu_id`    | <tt>int</tt>  | Device index to select. Defaults to `0`. |
| **RETURNS** | <tt>bool</tt> | Whether the GPU was activated.           |

### require_gpu {#require_gpu tag="function"}

Allocate data and perform operations on GPU. Will raise an error if no GPU is
available. If data has already been allocated on CPU, it will not be moved.
Ideally, this function should be called right after importing Thinc.

```python
### Example
from thinc.api import require_gpu
require_gpu()
```

| Argument    | Type          | Description |
| ----------- | ------------- | ----------- |
| **RETURNS** | <tt>bool</tt> | `True`.     |

### set_active_gpu {#set_active_gpu tag="function"}

Set the current GPU device for `cupy` and `torch` (if available).

```python
### Example
from thinc.api import set_active_gpu
set_active_gpu(0)
```

| Argument    | Type                      | Description             |
| ----------- | ------------------------- | ----------------------- |
| `gpu_id`    | <tt>int</tt>              | Device index to select. |
| **RETURNS** | <tt>cupy.cuda.Device</tt> | The device.             |

### minibatch {#minibatch tag="function"}

Iterate over batches of items. `size` may be an iterator, so that batch-size can
vary on each step.

```python
### Example
from thinc.api import minibatch

items = ("a", "b", "c", "d")
batch_sizes = (8, 16, 32, 8, 64)
batches = minibatch(items, batch_sizes)
```

| Argument   | Type                               | Description         |
| ---------- | ---------------------------------- | ------------------- |
| `items`    | <tt>Iterable[Any]</tt>             | The items to batch. |
| `size`     | <tt>Union[int, Iterable[int]]</tt> | The batch size(s).  |
| **YIELDS** | <tt>Any</tt>                       | The items.          |

### get_shuffled_batches {#get_shuffled_batches tag="function"}

Iterate over paired batches from two arrays, shuffling the indices.

| Argument     | Type                         | Description       |
| ------------ | ---------------------------- | ----------------- |
| `X`          | <tt>Array</tt>               | The first array.  |
| `Y`          | <tt>Array</tt>               | The second array. |
| `batch_size` | <tt>int</tt>                 | The batch size.   |
| **YIELDS**   | <tt>Tuple[Array, Array]</tt> | The batches.      |

### evaluate_model_on_arrays {#evaluate_model_on_arrays tag="function"}

Helper to evaluate accuracy of a model in the simplest cases, where there's one
correct output class and the inputs are arrays. Not guaranteed to cover all
situations – many applications will have to implement their own evaluation
methods.

| Argument     | Type           | Description                                |
| ------------ | -------------- | ------------------------------------------ |
| `model`      | <tt>Model</tt> | The model to evaluate.                     |
| `inputs`     | <tt>Array</tt> | The inputs of the dataset to evaluate on.  |
| `labels`     | <tt>Array</tt> | The outputs of the dataset to evaluate on. |
| `batch_size` | <tt>int</tt>   | The batch size.                            |
| **RETURNS**  | <tt>float</tt> | The score.                                 |

### is_numpy_array {#is_numpy_array tag="function"}

Check whether an array is a `numpy` array.

| Argument    | Type           | Description                           |
| ----------- | -------------- | ------------------------------------- |
| `arr`       | <tt>Array</tt> | The array to check.                   |
| **RETURNS** | <tt>bool</tt>  | Whether the array is a `numpy` array. |

### is_cupy_array {#is_cupy_array tag="function"}

Check whether an array is a `cupy` array.

| Argument    | Type           | Description                          |
| ----------- | -------------- | ------------------------------------ |
| `arr`       | <tt>Array</tt> | The array to check.                  |
| **RETURNS** | <tt>bool</tt>  | Whether the array is a `cupy` array. |

### get_width {#get_width tag="function"}

Infer the width of a batch of data, which could be any of: an n-dimensional
array (use the shape) or a sequence of arrays (use the shape of the first
element).

| Argument       | Type                                                             | Description                                            |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------------ |
| `X`            | <tt>Union[Array, Ragged, Padded, Sequence[Array], RNNState]</tt> | The array(s).                                          |
| _keyword-only_ |                                                                  |                                                        |
| `dim`          | <tt>int</tt>                                                     | Which dimension to get the size for. Defaults to `-1`. |
| **RETURNS**    | <tt>int</tt>                                                     | The array's inferred width.                            |

### to_categorical {#to_categorical tag="function"}

Converts a class vector (integers) to binary class matrix. Based on
[`keras.utils.to_categorical`](https://keras.io/utils/).

| Argument    | Type                   | Description                                                                    |
| ----------- | ---------------------- | ------------------------------------------------------------------------------ |
| `Y`         | <tt>IntsNd</tt>        | Class vector to be converted into a matrix (integers from `0` to `n_classes`). |
| `n_classes` | <tt>Optional[int]</tt> | Total number of classes.                                                       |
| **RETURNS** |  <tt>FloatsNd</tt>     | A binary matrix representation of the input. The classes axis is placed last.  |

### xp2torch {#xp2torch tag="function"}

Convert a `numpy` or `cupy` tensor to a PyTorch tensor.

| Argument        | Type                  | Description                                    |
| --------------- | --------------------- | ---------------------------------------------- |
| `xp_tensor`     | <tt>Array</tt>        | The tensor to convert.                         |
| `requires_grad` | <tt>bool</tt>         | Whether to backpropagate through the variable. |
| **RETURNS**     | <tt>torch.Tensor</tt> | The converted tensor.                          |

### torch2xp {#torch2xp tag="function"}

Convert a PyTorch tensor to a `numpy` or `cupy` tensor.

| Argument    | Type                  | Description            |
| ----------- | --------------------- | ---------------------- |
| `xp_tensor` | <tt>torch.Tensor</tt> | The tensor to convert. |
| **RETURNS** | <tt>Array</tt>        | The converted tensor.  |
