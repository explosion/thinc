---
title: Backends & Maths
next: /docs/api-util
---

TODO: intro

|                        |                                                    |
| ---------------------- | -------------------------------------------------- |
| [**Ops**](#ops)        | Backends for `numpy` on CPU and `cupy` on GPU.     |
| [**Utilities**](#util) | Helper functions for getting and setting backends. |

## Ops {#ops tag="class"}

The `Ops` class is typically not used directly but via `NumpyOps` and `CupyOps`,
which are subclasses of `Ops` and implement a more-efficient subset of the
methods. You also have access to the ops via the
[`Model.ops`](/docs/api-model#attributes) attribute.

```python
### Example
from thinc.api import Linear, with_list2array

model = with_list2array(Linear(4, 2))
Xs = [model.ops.allocate((10, 2), dtype="f")]
```

### Ops.\_\_init\_\_ {#init tag="method"}

| Argument | Type                  | Description        |
| -------- | --------------------- | ------------------ |
| `xp`     | <tt>Optional[Xp]</tt> | `numpy` or `cupy`. |

### Ops.seq2col {#seq2col tag="method"}

Given an `(M, N)` sequence of vectors, return an `(M, N*(nW*2+1))` sequence. The
new sequence is constructed by concatenating `nW` preceding and succeeding
vectors onto each column in the sequence, to extract a window of features.

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `seq`       | <tt>Array</tt> | TODO: ...   |
| `nW`        | <tt>int</tt>   | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.backprop_seq2col {#backprop_seq2col tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `dY`        | <tt>Array</tt> | TODO: ...   |
| `nW`        | <tt>int</tt>   | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.gemm {#gemm tag="method"}

| Argument    | Type                     | Description |
| ----------- | ------------------------ | ----------- |
| `x`         | <tt>Array</tt>           | TODO: ...   |
| `y`         | <tt>Array</tt>           | TODO: ...   |
| `out`       | <tt>Optional[Array]</tt> | TODO: ...   |
| `trans1`    | <tt>bool</tt>            | TODO: ...   |
| `trans2`    | <tt>bool</tt>            | TODO: ...   |
| **RETURNS** | <tt>Array</tt>           | TODO: ...   |

### Ops.flatten {#flatten tag="method"}

| Argument    | Type                     | Description |
| ----------- | ------------------------ | ----------- |
| `X`         | <tt>Sequence[Array]</tt> | TODO: ...   |
| `dtype`     | <tt>Optional[str]</tt>   | TODO: ...   |
| `pad`       | <tt>int</tt>             | TODO: ...   |
| **RETURNS** | <tt>Array</tt>           | TODO: ...   |

### Ops.unflatten {#unflatten tag="method"}

| Argument    | Type                 | Description |
| ----------- | -------------------- | ----------- |
| `X`         | <tt>Array</tt>       | TODO: ...   |
| `lengths`   | <tt>Array</tt>       | TODO: ...   |
| `pad`       | <tt>int</tt>         | TODO: ...   |
| **RETURNS** | <tt>List[Array]</tt> | TODO: ...   |

### Ops.pad_sequences {#pad_sequences tag="method"}

| Argument    | Type                            | Description |
| ----------- | ------------------------------- | ----------- |
| `seqs_in`   | <tt>Sequence[Array]</tt>        | TODO: ...   |
| `pad_to`    | <tt>Optional[int]</tt>          | TODO: ...   |
| **RETURNS** | <tt>Tuple[Array, Callable]</tt> | TODO: ...   |

### Ops.square_sequences {#square_sequences tag="method"}

Sort a batch of sequence by decreasing length, pad, and transpose so that the
outer dimension is the timestep. Return the padded batch, along with an array
indicating the actual length at each step, and a callback to reverse the
transformation.

| Argument    | Type                                   | Description |
| ----------- | -------------------------------------- | ----------- |
| `seqs`      | <tt>Sequence[Array]</tt>               | TODO: ...   |
| **RETURNS** | <tt>Tuple[Array, Array, Callable]</tt> | TODO: ...   |

### Ops.get_dropout_mask {#get_dropout_mask tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `shape`     | <tt>Shape</tt> | TODO: ...   |
| `drop`      | <tt>float</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.allocate {#allocate tag="method"}

| Argument       | Type           | Description |
| -------------- | -------------- | ----------- |
| `shape`        | <tt>Shape</tt> | TODO: ...   |
| _keyword-only_ |                |             |
| `dtype`        | <tt>str</tt>   | TODO: ...   |
| **RETURNS**    | <tt>Array</tt> | TODO: ...   |

### Ops.unzip {#unzip tag="method"}

| Argument    | Type                         | Description |
| ----------- | ---------------------------- | ----------- |
| `data`      | <tt>Tuple[Array, Array]      | TODO: ...   |
| **RETURNS** | <tt>Tuple[Array, Array]</tt> | TODO: ...   |

### Ops.asarray {#asarray tag="method"}

| Argument       | Type                                                  | Description |
| -------------- | ----------------------------------------------------- | ----------- |
| `data`         | <tt>Union[Array, Sequence[Array], Sequence[int]]</tt> | TODO: ...   |
| _keyword-only_ |                                                       |             |
| `dtype`        |  <tt>Optional[str]</tt>                               | TODO: ...   |
| **RETURNS**    | <tt>Array</tt>                                        | TODO: ...   |

### Ops.sigmoid {#sigmoid tag="method"}

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `X`            | <tt>Array</tt> | TODO: ...                                  |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, the array is modified in place. |
| **RETURNS**    | <tt>Array</tt> | TODO: ...                                  |

### Ops.dsigmoid {#dsigmoid tag="method"}

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `Y`            | <tt>Array</tt> | TODO: ...                                  |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, the array is modified in place. |
| **RETURNS**    | <tt>Array</tt> | TODO: ...                                  |

### Ops.dtanh {#dtanh tag="method"}

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `Y`            | <tt>Array</tt> | TODO: ...                                  |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, the array is modified in place. |
| **RETURNS**    | <tt>Array</tt> | TODO: ...                                  |

### Ops.softmax {#softmax tag="method"}

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `x`            | <tt>Array</tt> | TODO: ...                                  |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, the array is modified in place. |
| `axis`         | <tt>int</tt>   | TODO: ...                                  |
| **RETURNS**    | <tt>Array</tt> | TODO: ...                                  |

### Ops.backprop_softmax {#backprop_softmax tag="method"}

| Argument       | Type           | Description |
| -------------- | -------------- | ----------- |
| `Y`            | <tt>Array</tt> | TODO: ...   |
| `dY`           | <tt>Array</tt> | TODO: ...   |
| _keyword-only_ |                |             |
| `axis`         | <tt>int</tt>   | TODO: ...   |
| **RETURNS**    | <tt>Array</tt> | TODO: ...   |

### Ops.softmax_sequences {#softmax_sequences tag="method"}

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `Xs`           | <tt>Array</tt> | TODO: ...                                  |
| `lengths`      | <tt>Array</tt> | TODO: ...                                  |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, the array is modified in place. |
| `axis`         | <tt>int</tt>   | TODO: ...                                  |
| **RETURNS**    | <tt>Array</tt> | TODO: ...                                  |

### Ops.backprop_softmax_sequences {#backprop_softmax_sequences tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `dy`        | <tt>Array</tt> | TODO: ...   |
| `y`         | <tt>Array</tt> | TODO: ...   |
| `lengths`   | <tt>Array</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.clip_low {#clip_low tag="method"}

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `x`            | <tt>Array</tt> | TODO: ...                                  |
| `value`        | <tt>Array</tt> | TODO: ...                                  |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, the array is modified in place. |
| **RETURNS**    | <tt>Array</tt> | TODO: ...                                  |

### Ops.take_which {#take_which tag="method"}

| Argument       | Type           | Description |
| -------------- | -------------- | ----------- |
| `x`            | <tt>Array</tt> | TODO: ...   |
| `which`        | <tt>Array</tt> | TODO: ...   |
| _keyword-only_ |                |             |
| `axis`         | <tt>int</tt>   | TODO: ...   |
| **RETURNS**    | <tt>Array</tt> | TODO: ...   |

### Ops.backprop_take {#backprop_take tag="method"}

| Argument    | Type                 | Description |
| ----------- | -------------------- | ----------- |
| `dX`        | TODO: <tt>Array</tt> | TODO: ...   |
| `which`     | TODO: <tt>Array</tt> | TODO: ...   |
| `nP`        | <tt>int</tt>         | TODO: ...   |
| **RETURNS** | <tt>Array</tt>       | TODO: ...   |

### Ops.lstm {#lstm tag="method"}

| Argument | Type      | Description |
| -------- | --------- | ----------- |
| `output` | TODO: ... | TODO: ...   |
| `cells`  | TODO: ... | TODO: ...   |
| `acts`   | TODO: ... | TODO: ...   |
| `prev`   | TODO: ... | TODO: ...   |

### Ops.backprop_lstm {#backprop_lstm tag="method"}

| Argument    | Type      | Description |
| ----------- | --------- | ----------- |
| `d_cells`   | TODO: ... | TODO: ...   |
| `d_prev`    | TODO: ... | TODO: ...   |
| `d_gates`   | TODO: ... | TODO: ...   |
| `d_output`  | TODO: ... | TODO: ...   |
| `gates`     | TODO: ... | TODO: ...   |
| `cells`     | TODO: ... | TODO: ...   |
| `prev`      | TODO: ... | TODO: ...   |
| **RETURNS** | TODO: ... | TODO: ...   |

### Ops.softplus {#softplus tag="method"}

| Argument    | Type                     | Description |
| ----------- | ------------------------ | ----------- |
| `X`         | <tt>Array</tt>           | TODO: ...   |
| `threshold` | <tt>float</tt>           | TODO: ...   |
| `out`       | <tt>Optional[Array]</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt>           | TODO: ...   |

### Ops.backprop_softplus {#backprop_softplus tag="method"}

| Argument    | Type                     | Description |
| ----------- | ------------------------ | ----------- |
| `dY`        | <tt>Array</tt>           | TODO: ...   |
| `X`         | <tt>Array</tt>           | TODO: ...   |
| `threshold` | <tt>float</tt>           | TODO: ...   |
| `out`       | <tt>Optional[Array]</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt>           | TODO: ...   |

### Ops.mish {#mish tag="method"}

| Argument    | Type                     | Description |
| ----------- | ------------------------ | ----------- |
| `X`         | <tt>Array</tt>           | TODO: ...   |
| `threshold` | <tt>float</tt>           | TODO: ...   |
| `out`       | <tt>Optional[Array]</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt>           | TODO: ...   |

### Ops.backprop_mish {#backprop_softplus tag="method"}

| Argument    | Type                     | Description |
| ----------- | ------------------------ | ----------- |
| `dY`        | <tt>Array</tt>           | TODO: ...   |
| `X`         | <tt>Array</tt>           | TODO: ...   |
| `threshold` | <tt>float</tt>           | TODO: ...   |
| `out`       | <tt>Optional[Array]</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt>           | TODO: ...   |

### Ops.update_averages {#update_averages tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `ema`       | <tt>Array</tt> | TODO: ...   |
| `weights`   | <tt>Array</tt> | TODO: ...   |
| `t`         | <tt>int</tt>   | TODO: ...   |
| `max_decay` | <tt>float</tt> | TODO: ...   |

### Ops.adam {#adam tag="method"}

| Argument     | Type           | Description |
| ------------ | -------------- | ----------- |
| `weights`    | <tt>Array</tt> | TODO: ...   |
| `gradient`   | <tt>Array</tt> | TODO: ...   |
| `mom1`       | TODO: ...      | TODO: ...   |
| `mom2`       | TODO: ...      | TODO: ...   |
| `beta1`      | <tt>float</tt> | TODO: ...   |
| `beta2`      | <tt>float</tt> | TODO: ...   |
| `eps`        | <tt>float</tt> | TODO: ...   |
| `learn_rate` | <tt>float</tt> | TODO: ...   |
| `mod_rate`   | <tt>float</tt> | TODO: ...   |

### Ops.clip_gradient {#clip_gradient tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `gradient`  | <tt>Array</tt> | TODO: ...   |
| `threshold` | <tt>float</tt> | TODO: ...   |

### Ops.logloss {#logloss tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `y_true`    | <tt>Array</tt> | TODO: ...   |
| `y_true`    | <tt>Array</tt> | TODO: ...   |
| **RETURNS** | <tt>float</tt> | TODO: ...   |

### Ops.sum_pool {#sum_pool tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `X`         | <tt>Array</tt> | TODO: ...   |
| `lengths`   | <tt>Array</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.backprop_sum_pool {#backprop_sum_pool tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `d_sums`    | <tt>Array</tt> | TODO: ...   |
| `lengths`   | TODO: ...      | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.mean_pool {#mean_pool tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `X`         | <tt>Array</tt> | TODO: ...   |
| `lengths`   | <tt>Array</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.backprop_mean_pool {#backprop_mean_pool tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `d_means`   | <tt>Array</tt> | TODO: ...   |
| `lengths`   | TODO: ...      | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.max_pool {#max_pool tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `X`         | <tt>Array</tt> | TODO: ...   |
| `lengths`   | <tt>Array</tt> | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

### Ops.backprop_max_pool {#backprop_max_pool tag="method"}

| Argument    | Type           | Description |
| ----------- | -------------- | ----------- |
| `d_maxes`   | <tt>Array</tt> | TODO: ...   |
| `which`     | <tt>Array</tt> | TODO: ...   |
| `lengths`   | TODO: ...      | TODO: ...   |
| **RETURNS** | <tt>Array</tt> | TODO: ...   |

---

## Utilities {#util}

### get_ops {#get_ops tag="function"}

Get a backend object using a string name or device index.

```python
### Example
from thinc.api import get_ops

cpu_ops = get_ops("cpu")
gpu_ops = get_ops("gpu")
```

| Argument    | Type                              | Description                                              |
| ----------- | --------------------------------- | -------------------------------------------------------- |
| `ops`       | <tt>Union[int, str]</tt>          | `"cpu"`, `"gpu"`, `"numpy"`, `"cupy"` or a device index. |
| **RETURNS** | <tt>Union[NumpyOps, CupyOps]</tt> | The backend object.‚                                     |

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

### use_device {#use_device tag="contextmanager"}

Change the device to execute on for the scope of the block.

```python
### Example
from thinc.api import use_device, get_current_ops

with use_device("cpu"):
    current_ops = get_current_ops()
    assert current_ops.device == "cpu"
```

| Argument | Type                     | Description |
| -------- | ------------------------ | ----------- |
| `device` | <tt>Union[int, str]</tt> | TODO: ...   |
