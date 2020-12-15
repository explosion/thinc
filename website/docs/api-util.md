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

### require_cpu {#require_cpu tag="function"}

Allocate data and perform operations on CPU. 
If data has already been allocated on GPU, it will not be moved.
Ideally, this function should be called right after importing Thinc.

```python
### Example
from thinc.api import require_cpu
require_cpu()
```

| Argument    | Type          | Description |
| ----------- | ------------- | ----------- |
| **RETURNS** | <tt>bool</tt> | `True`.     |

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

### use_pytorch_for_gpu_memory {#use_pytorch_for_gpu_memory tag="function"}

Route GPU memory allocation via PyTorch. This is recommended for using PyTorch
and `cupy` together, as otherwise OOM errors can occur when there's available
memory sitting in the other library's pool. We'd like to support routing
TensorFlow memory allocation via PyTorch as well (or vice versa), but do not
currently have an implementation for it.

```python
### Example
from thinc.api import prefer_gpu, use_pytorch_for_gpu_memory

if prefer_gpu():
    use_pytorch_for_gpu_memory()
```

### use_tensorflow_for_gpu_memory {#use_tensorflow_for_gpu_memory tag="function"}

Route GPU memory allocation via TensorFlow. This is recommended for using
TensorFlow and `cupy` together, as otherwise OOM errors can occur when there's
available memory sitting in the other library's pool. We'd like to support
routing PyTorch memory allocation via TensorFlow as well (or vice versa), but do
not currently have an implementation for it.

```python
### Example
from thinc.api import prefer_gpu, use_tensorflow_for_gpu_memory

if prefer_gpu():
    use_tensorflow_for_gpu_memory()
```

### get_width {#get_width tag="function"}

Infer the width of a batch of data, which could be any of: an n-dimensional
array (use the shape) or a sequence of arrays (use the shape of the first
element).

| Argument       | Type                                                       | Description                                            |
| -------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| `X`            | <tt>Union[ArrayXd, Ragged, Padded, Sequence[ArrayXd]]</tt> | The array(s).                                          |
| _keyword-only_ |                                                            |                                                        |
| `dim`          | <tt>int</tt>                                               | Which dimension to get the size for. Defaults to `-1`. |
| **RETURNS**    | <tt>int</tt>                                               | The array's inferred width.                            |

### to_categorical {#to_categorical tag="function"}

Converts a class vector (integers) to binary class matrix. Based on
[`keras.utils.to_categorical`](https://keras.io/utils/).

| Argument    | Type                   | Description                                                                                    |
| ----------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| `Y`         | <tt>IntsXd</tt>        | Class vector to be converted into a matrix (integers from `0` to `n_classes`).                 |
| `n_classes` | <tt>Optional[int]</tt> | Total number of classes.                                                                       |
| **RETURNS** |  <tt>Floats2d</tt>     | A binary matrix representation of the input. The axis representing the classes is placed last. |

### xp2torch {#xp2torch tag="function"}

Convert a `numpy` or `cupy` tensor to a PyTorch tensor.

| Argument        | Type                  | Description                                    |
| --------------- | --------------------- | ---------------------------------------------- |
| `xp_tensor`     | <tt>ArrayXd</tt>      | The tensor to convert.                         |
| `requires_grad` | <tt>bool</tt>         | Whether to backpropagate through the variable. |
| **RETURNS**     | <tt>torch.Tensor</tt> | The converted tensor.                          |

### torch2xp {#torch2xp tag="function"}

Convert a PyTorch tensor to a `numpy` or `cupy` tensor.

| Argument       | Type                  | Description            |
| -------------- | --------------------- | ---------------------- |
| `torch_tensor` | <tt>torch.Tensor</tt> | The tensor to convert. |
| **RETURNS**    | <tt>ArrayXd</tt>      | The converted tensor.  |

### xp2tensorflow {#xp2tensorflow tag="function"}

Convert a `numpy` or `cupy` tensor to a TensorFlow tensor.

| Argument        | Type                       | Description                                           |
| --------------- | -------------------------- | ----------------------------------------------------- |
| `xp_tensor`     | <tt>ArrayXd</tt>           | The tensor to convert.                                |
| `requires_grad` | <tt>bool</tt>              | Whether to backpropagate through the variable.        |
| `as_variable`   | <tt>bool</tt>              | Convert the result to a `tensorflow.Variable` object. |  |
| **RETURNS**     | <tt>tensorflow.Tensor</tt> | The converted tensor.                                 |

### tensorflow2xp {#tensorflow2xp tag="function"}

Convert a TensorFlow tensor to a `numpy` or `cupy` tensor.

| Argument            | Type                       | Description            |
| ------------------- | -------------------------- | ---------------------- |
| `tensorflow_tensor` | <tt>tensorflow.Tensor</tt> | The tensor to convert. |
| **RETURNS**         | <tt>ArrayXd</tt>           | The converted tensor.  |

### xp2mxnet {#xp2mxnet tag="function"}

Convert a `numpy` or `cupy` tensor to an MXNet tensor.

| Argument        | Type                   | Description                                    |
| --------------- | ---------------------- | ---------------------------------------------- |
| `xp_tensor`     | <tt>ArrayXd</tt>       | The tensor to convert.                         |
| `requires_grad` | <tt>bool</tt>          | Whether to backpropagate through the variable. |
| **RETURNS**     | <tt>mx.nd.NDArray</tt> | The converted tensor.                          |

### mxnet2xp {#mxnet2xp tag="function"}

Convert an MXNet tensor to a `numpy` or `cupy` tensor.

| Argument    | Type                   | Description            |
| ----------- | ---------------------- | ---------------------- |
| `mx_tensor` | <tt>mx.nd.NDArray</tt> | The tensor to convert. |
| **RETURNS** | <tt>ArrayXd</tt>       | The converted tensor.  |

### Errors {#errors}

Thinc uses the following custom errors:

| Name                    | Description                                                                                                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ConfigValidationError` | Raised if invalid config settings are encountered by [`Config`](/docs/api-config#config) or the [`registry`](/docs/api-config#registry), or if resolving and validating the referenced functions fails.                           |
| `DataValidationError`   | Raised if [`Model.initialize`](/docs/api-model#initialize) is called with sample input or output data that doesn't match the expected input or output of the network, or leads to mismatched input or output in any of its layer. |
