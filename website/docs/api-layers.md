---
title: Layers
teaser: Weights layers, transforms, combinators and wrappers
next: /docs/api-optimizers
---

This page describes functions for defining your model. Each layer is implemented
in its own module in
[`thinc.layers`](https://github.com/explosion/thinc/blob/master/thinc/layers)
and can be imported from `thinc.api`. Most layer files define two public
functions: a **creation function** that returns a [`Model`](/docs/api-model)
instance, and a **forward function** that performs the computation.

|                                            |                                                                                  |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| [**Weights layers**](#weights-layers)      | Layers that use an internal weights matrix for their computations.               |
| [**Reduction operations**](#reduction-ops) | Layers that perform rank reductions, e.g. pooling from word to sentence vectors. |
| [**Combinators**](#combinators)            | Layers that combine two or more existing layers.                                 |
| [**Data type transfers**](#transfers)      | Layers that transform data to different types.                                   |
| [**Wrappers**](#wrappers)                  | Wrapper layers for other libraries like PyTorch and TensorFlow.                  |

## Weights layers {#weights-layers}

### CauchySimilarity {#cauchysimilarity tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Tuple[Floats2d, Floats2d]</ndarray>
- **Output:** <ndarray shape="batch_size">Floats1d</ndarray>
- **Parameters:** <ndarray shape="1, nI">W</ndarray>

</inline-list>

Compare input vectors according to the Cauchy similarity function proposed by
[Chen (2013)](https://tspace.library.utoronto.ca/bitstream/1807/43097/3/Liu_Chen_201311_MASc_thesis.pdf).
Primarily used within [`siamese`](#siamese) neural networks.

| Argument    | Type                                                | Description                    |
| ----------- | --------------------------------------------------- | ------------------------------ |
| `nI`        | <tt>Optional[int]</tt>                              | The size of the input vectors. |
| **RETURNS** | <tt>Model[Tuple[Floats2d, Floats2d], Floats1d]</tt> | The created similarity layer.  |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/cauchysimilarity.py
```

### Dish {#dish tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with the Dish activation function. Dish or "Daniël's Swish-like
activation" is an activation function with a non-monotinic shape similar to
[GELU](#gelu), [Swish](#swish) and [Mish](#mish). However, Dish does not rely on
elementary functions like `exp` or `erf`, making it much
[faster to compute](https://twitter.com/danieldekok/status/1484898130441166853)
in most cases.

| Argument       | Type                               | Description                                                                                                        |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                    |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                     |
| _keyword-only_ |                                    |                                                                                                                    |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`he_normal_init`](/docs/api-initializers#he_normal_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).             |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                 |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                    |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                           |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/dish.py
```

### Dropout {#dropout tag="function"}

<inline-list>

- **Input:** <ndarray>ArrayXd</ndarray> / <ndarray>Sequence[ArrayXd]</ndarray> /
  <ndarray>Ragged</ndarray> / <ndarray>Padded</ndarray>
- **Output:** <ndarray>ArrayXd</ndarray> / <ndarray>Sequence[ArrayXd]</ndarray>
  / <ndarray>Ragged</ndarray> / <ndarray>Padded</ndarray>
- **Attrs:** `dropout_rate` <tt>float</tt>

</inline-list>

Helps prevent overfitting by adding a random distortion to the input data during
training. Specifically, cells of the input are zeroed with probability
determined by the `dropout_rate` argument. Cells which are not zeroed are
rescaled by `1-rate`. When not in training mode, the distortion is disabled (see
[Hinton et al., 2012](https://arxiv.org/abs/1207.0580)).

```python
### Example
from thinc.api import chain, Linear, Dropout
model = chain(Linear(10, 2), Dropout(0.2))
Y, backprop = model(X, is_train=True)
# Configure dropout rate via the dropout_rate attribute.
for node in model.walk():
    if node.name == "dropout":
        node.attrs["dropout_rate"] = 0.5
```

| Argument       | Type                 | Description                                                                                                                             |
| -------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `dropout_rate` | <tt>float</tt>       | The probability of zeroing the activations (default: 0). Higher dropout rates mean more distortion. Values around `0.2` are often good. |
| **RETURNS**    | <tt>Model[T, T]</tt> | The created dropout layer.                                                                                                              |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/dropout.py
```

### Embed {#embed tag="function"}

<inline-list>

- **Input:** <ndarray shape="n,">Union[Ints1d, Ints2d]</ndarray>
- **Output:** <ndarray shape="n, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nV, nO">E</ndarray>
- **Attrs:** `column` <tt>int</tt>, `dropout_rate` <tt>float</tt>

</inline-list>

Map integers to vectors, using a fixed-size lookup table. The input to the layer
should be a two-dimensional array of integers, one column of which the
embeddings table will slice as the indices.

| Argument       | Type                                            | Description                                                                                                          |
| -------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>                          | The size of the output vectors.                                                                                      |
| `nV`           | <tt>int</tt>                                    | Number of input vectors. Defaults to `1`.                                                                            |
| _keyword-only_ |                                                 |                                                                                                                      |
| `column`       | <tt>int</tt>                                    | The column to slice from the input, to get the indices.                                                              |
| `initializer`  | <tt>Callable</tt>                               | A function to initialize the internal parameters. Defaults to [`uniform_init`](/docs/api-initializers#uniform_init). |
| `dropout`      | <tt>Optional[float]</tt>                        | Dropout rate to avoid overfitting (default `None`).                                                                  |
| **RETURNS**    | <tt>Model[Union[Ints1d, Ints2d], Floats2d]</tt> | The created embedding layer.                                                                                         |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/embed.py
```

### HashEmbed {#hashembed tag="function"}

<inline-list>

- **Input:** <ndarray shape="n,">Union[Ints1d, Ints2d]</ndarray> /
- **Output:** <ndarray shape="n, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nV, nO">E</ndarray>
- **Attrs:** `seed` <tt>Optional[int]</tt>, `column` <tt>int</tt>,
  `dropout_rate` <tt>float</tt>

</inline-list>

An embedding layer that uses the "hashing trick" to map keys to distinct values.
The hashing trick involves hashing each key four times with distinct seeds, to
produce four likely differing values. Those values are modded into the table,
and the resulting vectors summed to produce a single result. Because it's
unlikely that two different keys will collide on all four "buckets", most
distinct keys will receive a distinct vector under this scheme, even when the
number of vectors in the table is very low.

| Argument       | Type                                            | Description                                                                                                          |
| -------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>int</tt>                                    | The size of the output vectors.                                                                                      |
| `nV`           | <tt>int</tt>                                    | Number of input vectors.                                                                                             |
| _keyword-only_ |                                                 |                                                                                                                      |
| `seed`         | <tt>Optional[int]</tt>                          | A seed to use for the hashing.                                                                                       |
| `column`       | <tt>int</tt>                                    | The column to select features from.                                                                                  |
| `initializer`  | <tt>Callable</tt>                               | A function to initialize the internal parameters. Defaults to [`uniform_init`](/docs/api-initializers#uniform_init). |
| `dropout`      | <tt>Optional[float]</tt>                        | Dropout rate to avoid overfitting (default `None`).                                                                  |
| **RETURNS**    | <tt>Model[Union[Ints1d, Ints2d], Floats2d]</tt> | The created embedding layer.                                                                                         |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/hashembed.py
```

### LayerNorm {#layernorm tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nI,">b</ndarray>,
  <ndarray shape="nI,">G</ndarray>

</inline-list>

Perform layer normalization on the inputs
([Ba et al., 2016](https://arxiv.org/abs/1607.06450)). This layer does not
change the dimensionality of the vectors.

| Argument    | Type                               | Description                      |
| ----------- | ---------------------------------- | -------------------------------- |
| `nI`        | <tt>Optional[int]</tt>             | The size of the input vectors.   |
| **RETURNS** | <tt>Model[Floats2d, Floats2d]</tt> | The created normalization layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/layernorm.py
```

### Linear {#linear tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

The `Linear` layer multiplies inputs by a weights matrix `W` and adds a bias
vector `b`. In PyTorch this is called a `Linear` layer, while Keras calls it a
`Dense` layer.

```python
### Example
from thinc.api import Linear

model = Linear(10, 5)
model.initialize()
Y = model.predict(model.ops.alloc2f(2, 5))
assert Y.shape == (2, 10)
```

| Argument       | Type                               | Description                                                                                                                   |
| -------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                               |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                                |
| _keyword-only_ |                                    |                                                                                                                               |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init). |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                        |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created `Linear` layer.                                                                                                   |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/linear.py
```

### Sigmoid {#sigmoid tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A linear (aka dense) layer, followed by a sigmoid activation. This is usually
used as an output layer for multi-label classification (in contrast to the
`Softmax` layer, which is used for problems where exactly one class is correct
per example.

| Argument    | Type                               | Description                      |
| ----------- | ---------------------------------- | -------------------------------- |
| `nOs`       | <tt>Tuple[int, ...]</tt>           | The sizes of the output vectors. |
| `nI`        | <tt>Optional[int]</tt>             | The size of the input vectors.   |
| **RETURNS** | <tt>Model[Floats2d, Floats2d]</tt> | The created sigmoid layer.       |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/sigmoid.py
```

### sigmoid_activation {#sigmoid_activation tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">FloatsXd</ndarray>
- **Output:** <ndarray shape="batch_size, nO">FloatsXd</ndarray>

</inline-list>

Apply the sigmoid logistic function as an activation to the inputs. This is
often used as an output activation for multi-label classification, because each
element of the output vectors will be between `0` and `1`.

| Argument    | Type                               | Description                             |
| ----------- | ---------------------------------- | --------------------------------------- |
| **RETURNS** | <tt>Model[Floats2d, Floats2d]</tt> | The created `sigmoid_activation` layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/sigmoid_activation.py
```

### LSTM and BiLSTM {#lstm tag="function"}

<inline-list>

- **Input:** <ndarray>Padded</ndarray>
- **Output:** <ndarray>Padded</ndarray>
- **Parameters:** `depth` <tt>int</tt>, `dropout` <tt>float</tt>

</inline-list>

An LSTM recurrent neural network. The BiLSTM is bidirectional: that is, each
layer concatenated a forward LSTM with an LSTM running in the reverse direction.
If you are able to install PyTorch, you should usually prefer to use the
`PyTorchLSTM` layer instead of Thinc's implementations, as PyTorch's LSTM
implementation is significantly faster.

| Argument       | Type                           | Description                                      |
| -------------- | ------------------------------ | ------------------------------------------------ |
| `nO`           | <tt>Optional[int]</tt>         | The size of the output vectors.                  |
| `nI`           | <tt>Optional[int]</tt>         | The size of the input vectors.                   |
| _keyword-only_ |                                |                                                  |
| `bi`           | <tt>bool</tt>                  | Use BiLSTM.                                      |
| `depth`        | <tt>int</tt>                   | Number of layers (default `1`).                  |
| `dropout`      | <tt>float</tt>                 | Dropout rate to avoid overfitting (default `0`). |
| **RETURNS**    | <tt>Model[Padded, Padded]</tt> | The created LSTM layer(s).                       |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/lstm.py
```

### Maxout {#maxout tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO*nP">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO*nP, nI">W</ndarray>,
  <ndarray shape="nO*nP,">b</ndarray>

</inline-list>

A dense layer with a "maxout" activation
([Goodfellow et al, 2013](https://arxiv.org/abs/1302.4389)). Maxout layers
require a weights array of shape `(nO, nP, nI)` in order to compute outputs of
width `nO` given inputs of width `nI`. The extra multiple, `nP`, determines the
number of "pieces" that the piecewise-linear activation will consider.

| Argument       | Type                               | Description                                                                                                                   |
| -------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                               |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                                |
| `nP`           | <tt>int</tt>                       | Number of maxout pieces (default: 3).                                                                                         |
| _keyword-only_ |                                    |                                                                                                                               |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init). |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                        |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                            |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm), (default: False).                                                  |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created maxout layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/maxout.py
```

### Mish {#mish tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with Mish activation
([Misra, 2019](https://arxiv.org/pdf/1908.08681.pdf)).

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm), (default: False).                                                 |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/mish.py
```

### Swish {#swish tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with the Swish activation function
[(Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941v2). Swish is a
self-gating non-monotonic activation function similar to [`GELU`](#gelu):
whereas GELU uses the CDF of the Gaussian distribution Φ for self-gating
`x * Φ(x)` Swish uses the logistic CDF `x * σ(x)`. Sometimes referred to as
"SiLU" for "Sigmoid Linear Unit".

| Argument       | Type                               | Description                                                                                                        |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                    |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                     |
| _keyword-only_ |                                    |                                                                                                                    |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`he_normal_init`](/docs/api-initializers#he_normal_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).             |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                 |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                    |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                           |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/swish.py
```

### Gelu {#gelu tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with the GELU activation function
[(Hendrycks and Gimpel, 2016)](https://arxiv.org/abs/1606.08415). The GELU or
"Gaussian Error Linear Unit" is a self-gating non-monotonic activation function
similar to [Swish](#swish): whereas GELU uses the CDF of the Gaussian
distribution Φ for self-gating `x * Φ(x)` the Swish activation uses the logistic
CDF σ and computes `x * σ(x)`. Various approximations exist, but `thinc`
implements the exact GELU. The use of GELU is popular within transformer
feed-forward blocks.

| Argument       | Type                               | Description                                                                                                        |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                    |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                     |
| _keyword-only_ |                                    |                                                                                                                    |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`he_normal_init`](/docs/api-initializers#he_normal_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).             |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                 |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                    |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                           |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/gelu.py
```

### ReluK {#reluk tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with the ReLU activation function where the maximum value is
clipped at `k`. A common choice is `k=6` introduced for convolutional deep
belief networks
[(Krizhevsky, 2010)](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf).
The resulting function `relu6` is commonly used in low-precision scenarios.

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                              |
| `k`            | <tt>float</tt>                     | Maximum value. Defaults to `6.0`..                                                                                           |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/clipped_linear.py#L132
```

### HardSigmoid {#hardsigmoid tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with hard sigmoid activation function, which is a fast linear
approximation of sigmoid, defined as `max(0, min(1, x * 0.2 + 0.5))`.

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                              |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/clipped_linear.py#L90
```

### HardTanh {#hardtanh tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with hard tanh activation function, which is a fast linear
approximation of tanh, defined as `max(-1, min(1, x))`.

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                              |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/clipped_linear.py#L111
```

### ClippedLinear {#clippedlinear tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer implementing a flexible clipped linear activation function of the
form `max(min_value, min(max_value, x * slope + offset))`. It is used to
implement the [`ReluK`](#reluk), [`HardSigmoid`](#hardsigmoid), and
[`HardTanh`](#hardtanh) layers.

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                              |
| `slope`        | <tt>float</tt>                     | The slope of the linear function: `input * slope`.                                                                           |
| `offset`       | <tt>float</tt>                     | The offset or intercept of the linear function: `input * slope + offset`.                                                    |
| `min_val`      | <tt>float</tt>                     | Minimum value to clip to.                                                                                                    |
| `max_val`      | <tt>float</tt>                     | Maximum value to clip to.                                                                                                    |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/clipped_linear.py
```

### HardSwish {#hardswish tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer implementing the hard Swish activation function, which is a fast
linear approximation of Swish: `x * hard_sigmoid(x)`.

| Argument       | Type                               | Description                                                                                                        |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                    |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                     |
| _keyword-only_ |                                    |                                                                                                                    |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`he_normal_init`](/docs/api-initializers#he_normal_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).             |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                 |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                    |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                           |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/hard_swish.py
```

### HardSwishMobileNet {#hardswishmobilenet tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer implementing the a variant of the fast linear hard Swish
activation function used in `MobileNetV3`
[(Howard et al., 2019)](https://arxiv.org/abs/1905.02244), defined as
`x * (relu6(x + 3) / 6)`.

| Argument       | Type                               | Description                                                                                                        |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                    |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                     |
| _keyword-only_ |                                    |                                                                                                                    |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`he_normal_init`](/docs/api-initializers#he_normal_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).             |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                 |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm). Defaults to `False`.                                    |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created dense layer.                                                                                           |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/hard_swish_mobilenet.py
```

### MultiSoftmax {#multisoftmax tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

Neural network layer that predicts several multi-class attributes at once. For
instance, we might predict one class with six variables, and another with five.
We predict the 11 neurons required for this, and then softmax them such that
columns 0-6 make a probability distribution and columns 6-11 make another.

| Argument    | Type                               | Description                      |
| ----------- | ---------------------------------- | -------------------------------- |
| `nOs`       | <tt>Tuple[int, ...]</tt>           | The sizes of the output vectors. |
| `nI`        | <tt>Optional[int]</tt>             | The size of the input vectors.   |
| **RETURNS** | <tt>Model[Floats2d, Floats2d]</tt> | The created multi softmax layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/multisoftmax.py
```

### ParametricAttention {#parametricattention tag="function"}

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray>Ragged</ndarray>
- **Parameters:** <ndarray shape="nO,">Q</ndarray>

</inline-list>

A layer that uses the parametric attention scheme described by
[Yang et al. (2016)](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf).
The layer learns a parameter vector that is used as the keys in a single-headed
attention mechanism.

| Argument    | Type                           | Description                     |
| ----------- | ------------------------------ | ------------------------------- |
| `nO`        | <tt>Optional[int]</tt>         | The size of the output vectors. |
| **RETURNS** | <tt>Model[Ragged, Ragged]</tt> | The created attention layer.    |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/parametricattention.py
```

### Relu {#relu tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with Relu activation.

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`glorot_uniform_init`](/docs/api-initializers#glorot_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm), (default: False).                                                 |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created Relu layer.                                                                                                      |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/relu.py
```

### Softmax {#softmax tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with a softmax activation. This is usually used as a prediction
layer. Vectors produced by the softmax function sum to 1, and have values
between 0 and 1, so each vector can be interpreted as a probability
distribution.

| Argument       | Type                               | Description                                                                                              |
| -------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                          |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                           |
| _keyword-only_ |                                    |                                                                                                          |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`zero_init`](/docs/api-initializers#zero_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).   |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created softmax layer.                                                                               |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/softmax.py
```

### Softmax_v2 {#softmax_v2 tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with a softmax activation. This is usually used as a prediction
layer. Vectors produced by the softmax function sum to 1, and have values
between 0 and 1, so each vector can be interpreted as a probability
distribution.

`Softmax_v2` supports outputting unnormalized probabilities during inference by
using `normalize_outputs=False` as an argument. This is useful when we are only
interested in finding the top-k classes, but not their probabilities. Computing
unnormalized probabilities is faster, because it skips the expensive
normalization step.

The `temperature` argument of `Softmax_v2` provides control of the softmax
distribution. Values larger than 1 increase entropy and values between 0 and 1
(exclusive) decrease entropy of the distribution. The default temperature of 1
will calculate the unmodified softmax distribution. `temperature` is not used
during inference when `normalize_outputs=False`.

| Argument            | Type                               | Description                                                                                              |
| ------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `nO`                | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                          |
| `nI`                | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                           |
| _keyword-only_      |                                    |                                                                                                          |
| `init_W`            | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`zero_init`](/docs/api-initializers#zero_init) |
| `init_b`            | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).   |
| `normalize_outputs` | <tt>bool</tt>                      | Return normalized probabilities during inference. Defaults to `True`.                                    |
| `temperature`       | <tt>float</tt>                     | Temperature to divide logits by. Defaults to `1.0`.                                                      |
| **RETURNS**         | <tt>Model[Floats2d, Floats2d]</tt> | The created softmax layer.                                                                               |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/softmax.py
```

### SparseLinear {#sparselinear tag="function"}

<inline-list>

- **Input:** <ndarray>Tuple[ArrayXd, ArrayXd, ArrayXd]</ndarray>
- **Output:** <ndarray>ArrayXd</ndarray>
- **Parameters:** <ndarray shape="nO*length,">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>, `length` <tt>int</tt>

</inline-list>

A sparse linear layer using the "hashing trick". Useful for tasks such as text
classification. Inputs to the layer should be a tuple of arrays
`(keys, values, lengths)`, where the `keys` and `values` are arrays of the same
length, describing the concatenated batch of input features and their values.
The `lengths` array should have one entry per sequence in the batch, and the sum
of the lengths should equal the length of the keys and values array.

<infobox variant="warning">

`SparseLinear` should not be used for new models because it contains an indexing
bug. As a result, only a subset of the weights is used. Use
[`SparseLinear_v2`](#sparselinear_v2) instead.

</infobox>

| Argument    | Type                                                      | Description                                              |
| ----------- | --------------------------------------------------------- | -------------------------------------------------------- |
| `nO`        | <tt>Optional[int]</tt>                                    | The size of the output vectors.                          |
| `length`    | <tt>int</tt>                                              | The size of the weights vector, to be tuned empirically. |
| **RETURNS** | <tt>Model[Tuple[ArrayXd, ArrayXd, ArrayXd], ArrayXd]</tt> | The created layer.                                       |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/sparselinear.pyx
```

### SparseLinear_v2 {#sparselinear_v2 tag="function" new="8.1.6"}

<inline-list>

- **Input:** <ndarray>Tuple[ArrayXd, ArrayXd, ArrayXd]</ndarray>
- **Output:** <ndarray>ArrayXd</ndarray>
- **Parameters:** <ndarray shape="nO*length,">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>, `length` <tt>int</tt>

</inline-list>

A sparse linear layer using the "hashing trick". Useful for tasks such as text
classification. Inputs to the layer should be a tuple of arrays
`(keys, values, lengths)`, where the `keys` and `values` are arrays of the same
length, describing the concatenated batch of input features and their values.
The `lengths` array should have one entry per sequence in the batch, and the sum
of the lengths should equal the length of the keys and values array.

| Argument    | Type                                                      | Description                                              |
| ----------- | --------------------------------------------------------- | -------------------------------------------------------- |
| `nO`        | <tt>Optional[int]</tt>                                    | The size of the output vectors.                          |
| `length`    | <tt>int</tt>                                              | The size of the weights vector, to be tuned empirically. |
| **RETURNS** | <tt>Model[Tuple[ArrayXd, ArrayXd, ArrayXd], ArrayXd]</tt> | The created layer.                                       |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/sparselinear.pyx
```

## Reduction operations {#reduction-ops}

### reduce_first {#reduce_first tag="function"}

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">ArrayXd</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by selecting the first
item of each sequence. This is most useful after multi-head attention layers,
which can learn to assign a good feature representation for the sequence to one
of its elements.

| Argument    | Type                            | Description                |
| ----------- | ------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, ArrayXd]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/reduce_first.py
```

### reduce_last {#reduce_last tag="function"}

Pooling layer that reduces the dimensions of the data by selecting the last item
of each sequence. This is typically used after multi-head attention or recurrent
neural network layers such as LSTMs, which can learn to assign a good feature
representation for the sequence to its final element.

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">ArrayXd</ndarray>

</inline-list>

| Argument    | Type                            | Description                |
| ----------- | ------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, ArrayXd]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/reduce_last.py
```

### reduce_max {#reduce_max tag="function"}

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by selecting the maximum
value for each feature. A `ValueError` is raised if any element in `lengths` is
zero.

| Argument    | Type                             | Description                |
| ----------- | -------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, Floats2d]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/reduce_max.py
```

### reduce_mean {#reduce_mean tag="function"}

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by computing the average
value of each feature. Zero-length sequences are reduced to the zero vector.

| Argument    | Type                             | Description                |
| ----------- | -------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, Floats2d]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/reduce_mean.py
```

### reduce_sum {#reduce_sum tag="function"}

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by computing the sum for
each feature. Zero-length sequences are reduced to the zero vector.

| Argument    | Type                             | Description                |
| ----------- | -------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, Floats2d]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/reduce_sum.py
```

---

## Combinators {#combinators}

Combinators are layers that express **higher-order functions**: they take one or
more layers as arguments and express some relationship or perform some
additional logic around the child layers. Combinators can also be used to
[overload operators](/docs/usage-models#operators). For example, binding `chain`
to `>>` allows you to write `Relu(512) >> Softmax()` instead of
`chain(Relu(512), Softmax())`.

### add {#add tag="function"}

Compose two or more models `f`, `g`, etc, such that their outputs are added,
i.e. `add(f, g)(x)` computes `f(x) + g(x)`.

| Argument    | Type                         | Description            |
| ----------- | ---------------------------- | ---------------------- |
| `*layers`   | <tt>Model[Any, ArrayXd]</tt> | The models to compose. |
| **RETURNS** | <tt>Model[Any, ArrayXd]</tt> | The composed model.    |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/add.py
```

### bidirectional {#bidirectional tag="function"}

Stitch two RNN models into a bidirectional layer. Expects squared sequences.

| Argument    | Type                                     | Description                       |
| ----------- | ---------------------------------------- | --------------------------------- |
| `l2r`       | <tt>Model[Padded, Padded]</tt>           | The first model.                  |
| `r2l`       | <tt>Optional[Model[Padded, Padded]]</tt> | The second model.                 |
| **RETURNS** | <tt>Model[Padded, Padded]</tt>           | The composed bidirectional layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/bidirectional.py
```

### chain {#chain tag="function"}

Compose two or more models such that they become layers of a single feed-forward
model, e.g. `chain(f, g)` computes `g(f(x))`.

| Argument    | Type           | Description                       |
| ----------- | -------------- | --------------------------------- |
| `layer1 `   | <tt>Model</tt> | The first model to compose.       |
| `layer2`    | <tt>Model</tt> | The second model to compose.      |
| `*layers`   | <tt>Model</tt> | Any additional models to compose. |
| **RETURNS** | <tt>Model</tt> | The composed feed-forward model.  |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/chain.py
```

### clone {#clone tag="function"}

Construct `n` copies of a layer, with distinct weights. For example,
`clone(f, 3)(x)` computes `f(f'(f''(x)))`.

| Argument    | Type           | Description                                      |
| ----------- | -------------- | ------------------------------------------------ |
| `orig`      | <tt>Model</tt> | The layer to copy.                               |
| `n`         | <tt>int</tt>   | The number of copies to construct.               |
| **RETURNS** | <tt>Model</tt> | A composite model containing two or more copies. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/clone.py
```

### concatenate {#concatenate tag="function"}

Compose two or more models `f`, `g`, etc, such that their outputs are
concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`.

| Argument    | Type                | Description            |
| ----------- | ------------------- | ---------------------- |
| `*layers`   | <tt>Model</tt>, ... | The models to compose. |
| **RETURNS** | <tt>Model</tt>      | The composed model.    |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/concatenate.py
```

### map_list {#map_list tag="function"}

Map a child layer across list inputs.

| Argument    | Type                                  | Description             |
| ----------- | ------------------------------------- | ----------------------- |
| `layer`     | <tt>Model[InT, OutT]</tt>             | The child layer to map. |
| **RETURNS** | <tt>Model[List[InT], List[OutT]]</tt> | The composed model.     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/map_list.py
```

### expand_window {#expand_window tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d, Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d, Ragged</ndarray>
- **Attrs:** `window_size` <tt>int</tt>

</inline-list>

For each vector in an input, construct an output vector that contains the input
and a window of surrounding vectors. This is one step in a convolution. If the
`window_size` is three, the output size `nO` will be `nI * 7` after
concatenating three contextual vectors from the left, and three from the right,
to each input vector. In general, `nO` equals `nI * (2 * window_size + 1)`.

| Argument      | Type                 | Description                                                                    |
| ------------- | -------------------- | ------------------------------------------------------------------------------ |
| `window_size` | <tt>int</tt>         | The window size (default 1) that determines the number of surrounding vectors. |
| **RETURNS**   | <tt>Model[T, T]</tt> | The created layer for adding context to vectors.                               |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/expand_window.py
```

### noop {#noop tag="function"}

Transform a sequences of layers into a null operation.

| Argument    | Type           | Description            |
| ----------- | -------------- | ---------------------- |
| `*layers`   | <tt>Model</tt> | The models to compose. |
| **RETURNS** | <tt>Model</tt> | The composed model.    |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/noop.py
```

### residual {#residual tag="function"}

<inline-list>

- **Input:** <ndarray>List[FloatsXd]</ndarray> / <ndarray>Ragged</ndarray> /
  <ndarray>Padded</ndarray> / <ndarray>FloatsXd</ndarray>
  <ndarray>Floats1d</ndarray> <ndarray>Floats2d</ndarray>
  <ndarray>Floats3d</ndarray> <ndarray>Floats4d</ndarray>
- **Output:** <ndarray>List[FloatsXd]</ndarray> / <ndarray>Ragged</ndarray> /
  <ndarray>Padded</ndarray> / <ndarray>FloatsXd</ndarray>
  <ndarray>Floats1d</ndarray> <ndarray>Floats2d</ndarray>
  <ndarray>Floats3d</ndarray> <ndarray>Floats4d</ndarray>

</inline-list>

A unary combinator creating a residual connection. This converts a layer
computing `f(x)` into one that computes `f(x)+x`. Gradients flow through
residual connections directly, helping the network to learn more smoothly.

| Argument    | Type                 | Description                                        |
| ----------- | -------------------- | -------------------------------------------------- |
| `layer`     | <tt>Model[T, T]</tt> | A model with the same input and output types.      |
| **RETURNS** | <tt>Model[T, T]</tt> | A model with the unchanged input and output types. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/residual.py
```

### tuplify {#tuplify tag="function"}

Give each child layer a separate copy of the input, and the combine the output
of the child layers into a tuple. Useful for providing original and modified
input to a downstream layer.

On the backward pass the loss from each child is added together, so when using
custom datatypes they should define an addition operator.

| Argument    | Type                          | Description                      |
| ----------- | ----------------------------- | -------------------------------- |
| `*layers`   | <tt>Model[Any, T] ...</tt>    | The models to compose.           |
| **RETURNS** | <tt>Model[Any, Tuple[T]]</tt> | The composed feed-forward model. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/tuplify.py
```

### siamese {#siamese tag="function"}

Combine and encode a layer and a similarity function to form a
[siamese architecture](https://en.wikipedia.org/wiki/Siamese_neural_network).
Typically used to learn symmetric relationships, such as redundancy detection.

| Argument     | Type                           | Description                               |
| ------------ | ------------------------------ | ----------------------------------------- |
| `layer`      | <tt>Model</tt>                 | The layer to run over the pair of inputs. |
| `similarity` | <tt>Model</tt>                 | The similarity layer.                     |
| **RETURNS**  | <tt>Model[Tuple, ArrayXd]</tt> | The created siamese layer.                |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/siamese.py
```

### uniqued {#uniqued tag="function"}

Group inputs to a layer, so that the layer only has to compute for the unique
values. The data is transformed back before output, and the same transformation
is applied for the gradient. Effectively, this is a cache local to each
minibatch. The `uniqued` wrapper is useful for word inputs, because common words
are seen often, but we may want to compute complicated features for the words,
using e.g. character LSTM.

| Argument       | Type                             | Description                  |
| -------------- | -------------------------------- | ---------------------------- |
| `layer`        | <tt>Model</tt>                   | The layer.                   |
| _keyword-only_ |                                  |                              |
| `column`       | <tt>int</tt>                     | The column. Defaults to `0`. |
| **RETURNS**    | <tt>Model[Ints2d, Floats2d]</tt> | The composed model.          |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/uniqued.py
```

---

## Data type transfers {#transfers}

### array_getitem, ints_getitem, floats_getitem {#array_getitem tag="function"}

<inline-list>

- **Input:** <ndarray>ArrayXd</ndarray>
- **Output:** <ndarray>ArrayXd</ndarray>

</inline-list>

Index into input arrays, and return the subarrays. Multi-dimensional indexing
can be performed by passing in a tuple, and slicing can be performed using the
slice object. For instance, `X[:, :-1]` would be
`(slice(None, None), slice(None, -1))`.

| Argument | Type                                                                                          | Description                |
| -------- | --------------------------------------------------------------------------------------------- | -------------------------- |
| `index`  | <tt>Union[Union[int, slice, Sequence[int]], Tuple[Union[int, slice, Sequence[int]], ...]</tt> | A valid numpy-style index. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/array_getitem.py
```

### list2array {#list2array tag="function"}

<inline-list>

- **Input:** <ndarray>List2d</ndarray>
- **Output:** <ndarray>Array2d</ndarray>

</inline-list>

Transform sequences to ragged arrays if necessary. If sequences are already
ragged, do nothing. A ragged array is a tuple `(data, lengths)`, where `data` is
the concatenated data.

| Argument    | Type                            | Description                              |
| ----------- | ------------------------------- | ---------------------------------------- |
| **RETURNS** | <tt>Model[List2d, Array2d]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/list2array.py
```

### list2ragged {#list2ragged tag="function"}

<inline-list>

- **Input:** <ndarray>ListXd</ndarray>
- **Output:** <ndarray>Ragged</ndarray>

</inline-list>

Transform sequences to ragged arrays if necessary and return the ragged array.
If sequences are already ragged, do nothing. A ragged array is a tuple
`(data, lengths)`, where `data` is the concatenated data.

| Argument    | Type                           | Description                              |
| ----------- | ------------------------------ | ---------------------------------------- |
| **RETURNS** | <tt>Model[ListXd, Ragged]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/list2ragged.py
```

### list2padded {#list2padded tag="function"}

<inline-list>

- **Input:** <ndarray>List2d</ndarray>
- **Output:** <ndarray>Padded</ndarray>

</inline-list>

Create a layer to convert a list of array inputs into
[`Padded`](/docs/api-types#padded).

| Argument    | Type                           | Description                              |
| ----------- | ------------------------------ | ---------------------------------------- |
| **RETURNS** | <tt>Model[List2d, Padded]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/list2padded.py
```

### ragged2list {#ragged2list tag="function"}

<inline-list>

- **Input:** <ndarray>Ragged</ndarray>
- **Output:** <ndarray>ListXd</ndarray>

</inline-list>

Transform sequences from a ragged format into lists.

| Argument    | Type                           | Description                              |
| ----------- | ------------------------------ | ---------------------------------------- |
| **RETURNS** | <tt>Model[Ragged, ListXd]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/ragged2list.py
```

### padded2list {#padded2list tag="function"}

<inline-list>

- **Input:** <ndarray>Padded</ndarray>
- **Output:** <ndarray>List2d</ndarray>

</inline-list>

Create a layer to convert a [`Padded`](/docs/api-types#padded) input into a list
of arrays.

| Argument    | Type                           | Description                              |
| ----------- | ------------------------------ | ---------------------------------------- |
| **RETURNS** | <tt>Model[Padded, List2d]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/padded2list.py
```

### remap_ids {#remap_ids tag="function"}

<inline-list>

- **Input:** <tt>Union[Sequence[Hashable], Ints1d, Ints2d]</tt>
- **Output:** <ndarray>Ints2d</ndarray>

</inline-list>

Remap a sequence of strings, integers or other hashable inputs using a mapping
table, usually as a preprocessing step before embeddings. The input can also be
a two dimensional integer array in which case the `column` attribute tells the
`remap_ids` layer which column of the array to map with the `mapping_table`.
Both attributes can be passed on initialization, but since the layer is designed
to retrieve them from `model.attrs` during `forward`, they can be set any time
before calling `forward`. This means that they can also be changed between
calls. Before calling `forward` the `mapping_table` has to be set and for 2D
inputs the `column` is also required.

| Argument        | Type                                                              | Description                                                                                                  |
| --------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `mapping_table` | <tt>Dict[Any, int]</tt>                                           | The mapping table to use. Can also be set after initialization by writing to `model.attrs["mapping_table"]`. |
| `default`       | <tt>int</tt>                                                      | The default value if the input does not have an entry in the mapping table.                                  |
| `column`        | <tt>int</tt>                                                      | The column to apply the mapper to in case of 2D input.                                                       |
| **RETURNS**     | <tt>Model[Union[Sequence[Hashable], Ints1d, Ints2d], Ints2d]</tt> | The layer to compute the transformation.                                                                     |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/remap_ids.py
```

### strings2arrays {#strings2arrays tag="function"}

<inline-list>

- **Input:** <tt>Sequence[Sequence[str]]</tt>
- **Output:** <ndarray>List[Ints2d]</ndarray>

</inline-list>

Transform a sequence of string sequences to a list of arrays.

| Argument    | Type                                                  | Description                              |
| ----------- | ----------------------------------------------------- | ---------------------------------------- |
| **RETURNS** | <tt>Model[Sequence[Sequence[str]], List[Ints2d]]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/strings2arrays.py
```

### with_array {#with_array tag="function"}

<inline-list>

- **Input / output:** <tt>Union[Padded, Ragged, ListXd, ArrayXd]</tt>

</inline-list>

Transform sequence data into a contiguous array on the way into and out of a
model. Handles a variety of sequence types: lists, padded and ragged. If the
input is an array, it is passed through unchanged.

| Argument       | Type                             | Description                   |
| -------------- | -------------------------------- | ----------------------------- |
| `layer`        | <tt>Model[ArrayXd, ArrayXd]</tt> | The layer to wrap.            |
| _keyword-only_ |                                  |                               |
| `pad`          | <tt>int</tt>                     | The padding. Defaults to `0`. |
| **RETURNS**    | <tt>Model</tt>                   | The wrapped layer.            |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_array2d.py
```

### with_array2d {#with_array2d tag="function"}

<inline-list>

- **Input / output:** <tt>Union[Padded, Ragged, List2d, Array2d]</tt>

</inline-list>

Transform sequence data into a contiguous two-dimensional array on the way into
and out of a model. In comparison to the `with_array` layer, the behavior of
this layer mostly differs on `Padded` inputs, as this layer merges the batch and
length axes to form a two-dimensional array. Handles a variety of sequence
types: lists, padded and ragged. If the input is a two-dimensional array, it is
passed through unchanged.

| Argument       | Type                             | Description                   |
| -------------- | -------------------------------- | ----------------------------- |
| `layer`        | <tt>Model[Array2d, Array2d]</tt> | The layer to wrap.            |
| _keyword-only_ |                                  |                               |
| `pad`          | <tt>int</tt>                     | The padding. Defaults to `0`. |
| **RETURNS**    | <tt>Model</tt>                   | The wrapped layer.            |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_array.py
```

### with_flatten {#with_flatten tag="function"}

<inline-list>

- **Input:** <tt>Sequence[Sequence[Any]]</tt>
- **Output:** <tt>ListXd</tt>

</inline-list>

Flatten nested inputs on the way into a layer and reverse the transformation
over the outputs.

| Argument    | Type                                                             | Description        |
| ----------- | ---------------------------------------------------------------- | ------------------ |
| `layer`     | <tt>Model[Sequence[Sequence[Any]], Sequence[Sequence[Any]]]</tt> | The layer to wrap. |
| **RETURNS** | <tt>Model[ListXd, ListXd]</tt>                                   | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_flatten.py
```

### with_padded {#with_padded tag="function"}

<inline-list>

- **Input / output:** <tt>Union[Padded, Ragged, List2d, Floats3d,
  Tuple[Floats3d, Ints1d, Ints1d, Ints1d]]</tt>

</inline-list>

Convert sequence input into the [`Padded`](/docs/api-types#padded) data type on
the way into a layer and reverse the transformation on the output.

| Argument    | Type                           | Description        |
| ----------- | ------------------------------ | ------------------ |
| `layer`     | <tt>Model[Padded, Padded]</tt> | The layer to wrap. |
| **RETURNS** | <tt>Model</tt>                 | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_padded.py
```

### with_ragged {#with_ragged tag="function"}

<inline-list>

- **Input / output:** <tt>Union[Padded, Ragged, ListXd, Floats3d,
  Tuple[Floats2d, Ints1d]]</tt>

</inline-list>

Convert sequence input into the [`Ragged`](/docs/api-types#ragged) data type on
the way into a layer and reverse the transformation on the output.

| Argument    | Type                           | Description        |
| ----------- | ------------------------------ | ------------------ |
| `layer`     | <tt>Model[Ragged, Ragged]</tt> | The layer to wrap. |
| **RETURNS** | <tt>Model</tt>                 | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_ragged.py
```

### with_list {#with_list tag="function"}

<inline-list>

- **Input / output:** <tt>Union[Padded, Ragged, List2d]</tt>

</inline-list>

Convert sequence input into lists on the way into a layer and reverse the
transformation on the outputs.

| Argument    | Type                           | Description        |
| ----------- | ------------------------------ | ------------------ |
| `layer`     | <tt>Model[List2d, List2d]</tt> | The layer to wrap. |
| **RETURNS** | <tt>Model</tt>                 | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_list.py
```

### with_getitem {#with_getitem tag="function"}

<inline-list>

- **Input:** <tt>Tuple</tt>
- **Output:** <tt>Tuple</tt>

</inline-list>

Transform data on the way into and out of a layer by plucking an item from a
tuple.

| Argument    | Type                             | Description                        |
| ----------- | -------------------------------- | ---------------------------------- |
| `idx`       | <tt>int</tt>                     | The index to pluck from the tuple. |
| `layer`     | <tt>Model[ArrayXd, ArrayXd]</tt> | The layer to wrap.                 |
| **RETURNS** | <tt>Model[Tuple, Tuple]</tt>     | The wrapped layer.                 |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_getitem.py
```

### with_reshape {#with_reshape tag="function"}

<inline-list>

- **Input:** <ndarray>Array3d</tt>
- **Output:** <ndarray>Array3d</tt>

</inline-list>

Reshape data on the way into and out from a layer.

| Argument    | Type                             | Description        |
| ----------- | -------------------------------- | ------------------ |
| `layer`     | <tt>Model[Array2d, Array2d]</tt> | The layer to wrap. |
| **RETURNS** | <tt>Model[Array3d, Array3d]</tt> | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_reshape.py
```

### with_debug {#with_debug tag="function"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Debugging layer that wraps any layer and allows executing callbacks during the
forward pass, backward pass and initialization. The callbacks will receive the
same arguments as the functions they're called in and are executed before the
function runs.

<infobox variant="warning">

This layer should only be used for **debugging, logging, benchmarking etc.**,
not to modify data or perform any other side-effects that are relevant to the
network outside of debugging and testing it. If you need hooks that run in
specific places of the model lifecycle, you should write your own
[custom layer](/docs/usage-models#new-layers). You can use the implementation of
`with_debug` as a template.

</infobox>

```python
### Example
from thinc.api import Linear, with_debug

def on_init(model, X, Y):
    print(f"X: {type(Y)}, Y ({type(Y)})")

model = with_debug(Linear(2, 5), on_init=on_init)
model.initialize()
```

| Argument       | Type                                        | Description                                                                                                                                         |
| -------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layer`        | <tt>Model</tt>                              | The layer to wrap.                                                                                                                                  |
| `name`         | <tt>Optional[str]</tt>                      | Optional name for the wrapped layer, will be prefixed by `debug:`. Defaults to name of the wrapped layer.                                           |
| _keyword-only_ |                                             |                                                                                                                                                     |
| `on_init`      | <tt>Callable[[Model, Any, Any], None]</tt>  | Function called on initialization. Receives the model and the `X` and `Y` passed to [`Model.initialize`](/docs/api-model#initialize), if available. |
| `on_forward`   | <tt>Callable[[Model, Any, bool], None]</tt> | Function called at the start of the forward pass. Receives the model, the inputs and the value of `is_train`.                                       |
| `on_backprop`  | <tt>Callable[[Any], None] = do_nothing</tt> | Function called at the start of the backward pass. Receives the gradient.                                                                           |
| **RETURNS**    | <tt>Model</tt>                              | The wrapped layer.                                                                                                                                  |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_debug.py
```

### with_nvtx_range {#with_nvtx_range tag="function"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Layer that wraps any layer and marks the forward and backprop passes as an NVTX
range. This can be helpful when profiling GPU performance of a layer.

```python
### Example
from thinc.api import Linear, with_nvtx_range

model = with_nvtx_range(Linear(2, 5))
model.initialize()
```

| Argument         | Type                   | Description                                                                     |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------- |
| `layer`          | <tt>Model</tt>         | The layer to wrap.                                                              |
| `name`           | <tt>Optional[str]</tt> | Optional name for the wrapped layer. Defaults to the name of the wrapped layer. |
| _keyword-only_   |                        |                                                                                 |
| `forward_color`  | <tt>int</tt>           | Identifier of the color to use for the forward pass                             |
| `backprop_color` | <tt>int</tt>           | Identifier of the color to use for the backward pass                            |
| **RETURNS**      | <tt>Model</tt>         | The wrapped layer.                                                              |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_nvtx_range.py
```

### with_signpost_interval {#with_signpost_interval tag="function" new="8.1.1"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Layer that wraps any layer and marks the init, forward and backprop passes as a
(macOS) signpost interval. This can be helpful when profiling the performance of
a layer using macOS
[Instruments.app](https://help.apple.com/instruments/mac/current/). Use of this
layer requires that the
[`os-signpost`](https://github.com/explosion/os-signpost) package is installed.

```python
### Example
from os_signpost import Signposter
from thinc.api import Linear, with_signpost_interval

signposter = Signposter("com.example.my_subsystem",
    Signposter.Category.DynamicTracing)

model = with_signpost_interval(Linear(2, 5), signposter)
model.initialize()
```

| Argument     | Type                              | Description                                                                     |
| ------------ | --------------------------------- | ------------------------------------------------------------------------------- |
| `layer`      | <tt>Model</tt>                    | The layer to wrap.                                                              |
| `signposter` | <tt>os_signposter.Signposter</tt> | `Signposter` object to log the interval with.                                   |
| `name`       | <tt>Optional[str]</tt>            | Optional name for the wrapped layer. Defaults to the name of the wrapped layer. |
| **RETURNS**  | <tt>Model</tt>                    | The wrapped layer.                                                              |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/with_signpost_interval.py
```

---

## Wrappers {#wrappers}

### PyTorchWrapper, PyTorchRNNWrapper {#pytorchwrapper tag="function"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Wrap a [PyTorch](https://pytorch.org) model so that it has the same API as Thinc
models. To optimize the model, you'll need to create a PyTorch optimizer and
call `optimizer.step` after each batch. The `PyTorchRNNWrapper` has the same
signature as the `PyTorchWrapper` and lets you to pass in a custom sequence
model that has the same inputs and output behavior as a
[`torch.nn.RNN`](https://pytorch.org/docs/stable/nn.html#torch.nn.RNN) object.

Your PyTorch model's forward method can take arbitrary positional arguments and
keyword arguments, but must return either a **single tensor** as output or a
**tuple**. You may find
[PyTorch's `register_forward_hook`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_hook)
helpful if you need to adapt the output. The convert functions are used to map
inputs and outputs to and from your PyTorch model. Each function should return
the converted output, and a callback to use during the backward pass:

```python
Xtorch, get_dX = convert_inputs(X)
Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
Y, get_dYtorch = convert_outputs(Ytorch)
```

To allow maximum flexibility, the [`PyTorchShim`](/docs/api-model#shims) expects
[`ArgsKwargs`](/docs/api-types#argskwargs) objects on the way into the forward
and backward passes. The `ArgsKwargs` objects will be passed straight into the
model in the forward pass, and straight into `torch.autograd.backward` during
the backward pass.

| Argument          | Type                     | Description                                                                              |
| ----------------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `pytorch_model`   | <tt>Any</tt>             | The PyTorch model.                                                                       |
| `convert_inputs`  | <tt>Callable</tt>        | Function to convert inputs to PyTorch tensors (same signature as `forward` function).    |
| `convert_outputs` | <tt>Callable</tt>        | Function to convert outputs from PyTorch tensors (same signature as `forward` function). |
| **RETURNS**       | <tt>Model[Any, Any]</tt> | The Thinc model.                                                                         |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/pytorchwrapper.py
```

### TorchScriptWrapper_v1 {#torchscriptwrapper tag="function" new="8.1.6"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Wrap a [TorchScript](https://pytorch.org/docs/stable/jit.html) model so that it
has the same API as Thinc models. To optimize the model, you'll need to create a
PyTorch optimizer and call `optimizer.step` after each batch.

Your TorchScript model's forward method can take arbitrary positional arguments
and keyword arguments, but must return either a **single tensor** as output or a
**tuple**. The convert functions are used to map inputs and outputs to and from
your TorchScript model. Each function should return the converted output, and a
callback to use during the backward pass:

```python
Xtorch, get_dX = convert_inputs(X)
Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
Y, get_dYtorch = convert_outputs(Ytorch)
```

To allow maximum flexibility, the [`TorchScriptShim`](/docs/api-model#shims)
expects [`ArgsKwargs`](/docs/api-types#argskwargs) objects on the way into the
forward and backward passes. The `ArgsKwargs` objects will be passed straight
into the model in the forward pass, and straight into `torch.autograd.backward`
during the backward pass.

Note that the `torchscript_model` argument can be `None`. This is useful for
deserialization since serialized TorchScript contains both the model and its
weights.

A PyTorch wrapper can be converted to a TorchScript wrapper using the
`pytorch_to_torchscript_wrapper` function:

```python
from thinc.api import PyTorchWrapper_v2, pytorch_to_torchscript_wrapper
import torch

model = PyTorchWrapper_v2(torch.nn.Linear(nI, nO)).initialize()
script_model = pytorch_to_torchscript_wrapper(model)
```

| Argument            | Type                                      | Description                                                                              |
| ------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| `torchscript_model` | <tt>Optional[torch.jit.ScriptModule]</tt> | The TorchScript model.                                                                   |
| `convert_inputs`    | <tt>Callable</tt>                         | Function to convert inputs to PyTorch tensors (same signature as `forward` function).    |
| `convert_outputs`   | <tt>Callable</tt>                         | Function to convert outputs from PyTorch tensors (same signature as `forward` function). |
| `mixed_precision`   | <tt>bool</tt>                             | Enable mixed-precision training.                                                         |
| `grad_scaler`       | <tt>Optional[PyTorchGradScaler]</tt>      | Gradient scaler to use during mixed-precision training.                                  |
| `device`            | <tt>Optional[torch.Device]</tt>         | The Torch device to execute the model on.                                                |
| **RETURNS**         | <tt>Model[Any, Any]</tt>                  | The Thinc model.                                                                         |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/torchscriptwrapper.py
```

### TensorFlowWrapper {#tensorflowwrapper tag="function"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Wrap a [TensorFlow](https://tensorflow.org) model, so that it has the same API
as Thinc models. To optimize the model, you'll need to create a TensorFlow
optimizer and call `optimizer.apply_gradients` after each batch. To allow
maximum flexibility, the [`TensorFlowShim`](/docs/api-model#shims) expects
[`ArgsKwargs`](/docs/api-types#argskwargs) objects on the way into the forward
and backward passes.

| Argument           | Type                     | Description           |
| ------------------ | ------------------------ | --------------------- |
| `tensorflow_model` | <tt>Any</tt>             | The TensorFlow model. |
| **RETURNS**        | <tt>Model[Any, Any]</tt> | The Thinc model.      |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/tensorflowwrapper.py
```

### MXNetWrapper {#mxnetwrapper tag="function"}

<inline-list>

- **Input:** <tt>Any</tt>
- **Output:** <tt>Any</tt>

</inline-list>

Wrap a [MXNet](https://mxnet.apache.org/) model, so that it has the same API as
Thinc models. To optimize the model, you'll need to create a MXNet optimizer and
call `optimizer.step()` after each batch. To allow maximum flexibility, the
[`MXNetShim`](/docs/api-model#shims) expects
[`ArgsKwargs`](/docs/api-types#argskwargs) objects on the way into the forward
and backward passes.

| Argument           | Type                     | Description           |
| ------------------ | ------------------------ | --------------------- |
| `tensorflow_model` | <tt>Any</tt>             | The TensorFlow model. |
| **RETURNS**        | <tt>Model[Any, Any]</tt> | The Thinc model.      |

```python
https://github.com/explosion/thinc/blob/master/thinc/layers/mxnetwrapper.py
```
