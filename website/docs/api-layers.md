---
title: Layers
teaser: Weights layers, transforms and combinators
next: /docs/api-optimizers
---

This page describes functions for defining your model. Each layer is implemented
in its own module in `thinc.layers`. Most layer files define two public
functions: a **creation function** that returns a [`Model`](/docs/api-model)
instance, and a **forward function** that performs the computation.

|                                       |                                                                  |
| ------------------------------------- | ---------------------------------------------------------------- |
| [**Weights layers**](#weights-layers) | Layer that uses an internal weights matrix for its computations. |
| [**Pooling layers**](#pooling-layers) | Pooling layers.                                                  |
| [**Combinators**](#combinators)       | Layer that combines two or more existing layers.                 |
| [**Data type transfers**](#transfers) | Layers that transform data to different types.                   |

## Weights layers {#weights-layers}

### CauchySimilarity {#cauchysimilarity tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Tuple[Floats2d]</ndarray>
- **Output:** <ndarray shape="batch_size">Floats1d</ndarray>
- **Parameters:** <ndarray shape="1, nI">W</ndarray>

</inline-list>

Compare input vectors according to the Cauchy similarity function proposed by
[Chen (2013)](https://tspace.library.utoronto.ca/bitstream/1807/43097/3/Liu_Chen_201311_MASc_thesis.pdf).
Primarily used within [`Siamese`](#siamese) neural networks.

| Argument    | Type                                      | Description                    |
| ----------- | ----------------------------------------- | ------------------------------ |
| `nI`        | <tt>Optional[int]</tt>                    | The size of the input vectors. |
| **RETURNS** | <tt>Model[Tuple[Floats2d], Floats1d]</tt> | The created similarity layer.  |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/cauchysimilarity.py
```

### Dropout {#dropout tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nO">Array</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Array</ndarray>
- **Attrs:** `rate` <tt>float</tt>

</inline-list>

Help prevent overfitting by adding a random distortion to the input data during
training. Specifically, cells of the input are zeroed with probability
determined by the `rate` argument. Cells which are not zeroed are rescaled by
`1-rate`. When not in training mode, the distortion is disabled (see
[Hinton et al., 2012](https://arxiv.org/abs/1207.0580)).

```python
### Example
from thinc.api import chain, Linear, Dropout
model = chain(Linear(10, 2), Dropout(0.2))
Y, backprop = model(X, is_train=True)
# Configure dropout rate via the `rate` attribute.
for node in model.walk():
    if node.name == "dropout":
        node.set_attr("rate", 0.5)
```

| Argument    | Type                         | Description                                                                                                                             |
| ----------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `rate`      | <tt>float</tt>               | The probability of zeroing the activations (default: 0). Higher dropout rates mean more distortion. Values around `0.2` are often good. |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The created dropout layer.                                                                                                              |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/dropout.py
```

### Embed {#embed tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nV">Ints2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nV, nO">vectors</ndarray>
- **Attrs:** `column` <tt>int</tt>

</inline-list>

Map integers to vectors, using a fixed-size lookup table. The input to the layer
should be a 2-dimensional array of integers, one column of which the embeddings
table will slice as the indices.

| Argument       | Type                             | Description                                                                                                          |
| -------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>           | The size of the output vectors.                                                                                      |
| `nV`           | <tt>Optional[int]</tt>           | Number of input vectors.                                                                                             |
| _keyword-only_ |                                  |                                                                                                                      |
| `column`       | <tt>int</tt>                     | The column to slice from the input, to get the indices.                                                              |
| `initializer`  | <tt>Callable</tt>                | A function to initialize the internal parameters. Defaults to [`uniform_init`](/docs/api-initializers#uniform_init). |
| **RETURNS**    | <tt>Model[Ints2d, Floats2d]</tt> | The created embedding layer.                                                                                         |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/embed.py
```

### ExtractWindow {#extractwindow tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Array</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Array</ndarray>
- **Attrs:** `window_size` <tt>int</tt>

</inline-list>

For each vector in an input, construct an output vector that contains the input
and a window of surrounding vectors. This is one step in a convolution. If the
`window_size` is 3, the output size `nO` will be `nI * 7` after concatenating 3
contextual vectors from the left, and 3 from the right, to each input vector. In
general, `nO` equals `nI * (2 * window_size + 1)`.

| Argument      | Type                         | Description                                                                    |
| ------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| `window_size` | <tt>int</tt>                 | The window size (default 1) that determines the number of surrounding vectors. |
| **RETURNS**   | <tt>Model[Array, Array]</tt> | The created layer for adding context to vectors.                               |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/extractwindow.py
```

### FeatureExtractor {#featureextractor tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">List[spacy.tokens.Doc]</ndarray>
- **Output:** <ndarray shape="batch_size, nO">List[Ints2d]</ndarray>
- **Attrs:** `columns` <tt>int</tt>

</inline-list>

spaCy-specific layer to extract arrays of input features from `Doc` objects.
Expects a list of feature names to extract, which should refer to spaCy token
attributes.

| Argument    | Type                                                 | Description                            |
| ----------- | ---------------------------------------------------- | -------------------------------------- |
| `columns`   | <tt>List[Union[int, str]]</tt>                       | The spaCy token attributes to extract. |
| **RETURNS** | <tt>Model[List[spacy.tokens.Doc], List[Ints2d]]</tt> | The created feature extraction layer   |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/featureextractor.py
```

### HashEmbed {#hashembed tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nV">Ints2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Attrs:** `seed` <tt>Optional[int]</tt>, `column` <tt>int</tt>
- **Parameters:** `vectors` <tt>Optional[Array]</tt>

</inline-list>

An embedding layer that uses the "hashing trick" to map keys to distinct values.
The hashing trick involves hashing each key 4 times with distinct seeds, to
produce 4 likely differing values. Those values are modded into the table, and
the resulting vectors summed to produce a single result. Because it's unlikely
that two different keys will collide on all four "buckets", most distinct keys
will receive a distinct vector under this scheme, even when the number of
vectors in the table is very low, even when the number of vectors in the table
is very low, even when the number of vectors in the table is very low, even when
the number of vectors in the table is very low.

| Argument       | Type                             | Description                                                                                                          |
| -------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>int</tt>                     | The size of the output vectors.                                                                                      |
| `nV`           | <tt>int</tt>                     | Number of input vectors.                                                                                             |
| _keyword-only_ |                                  |                                                                                                                      |
| `seed`         | <tt>Optional[int]</tt>           | A seed to use for the hashing.                                                                                       |
| `column`       | <tt>int</tt>                     | The column to select features from.                                                                                  |
| `initializer`  | <tt>Callable</tt>                | A function to initialize the internal parameters. Defaults to [`uniform_init`](/docs/api-initializers#uniform_init). |
| **RETURNS**    | <tt>Model[Ints2d, Floats2d]</tt> | The created embedding layer.                                                                                         |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/hashembed.py
```

### LayerNorm {#layernorm tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO,">b</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

Perform layer normalization on the inputs
([Ba et al., 2016](https://arxiv.org/abs/1607.06450)). This layer does not
change the dimensionality of the vectors.

| Argument    | Type                               | Description                      |
| ----------- | ---------------------------------- | -------------------------------- |
| `nO`        | <tt>Optional[int]</tt>             | The size of the output vectors.  |
| **RETURNS** | <tt>Model[Floats2d, Floats2d]</tt> | The created normalization layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/layernorm.py
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
Y = model.predict(model.ops.allocate(2, 5))
assert Y.shape == (2, 10)
```

| Argument       | Type                               | Description                                                                                                                   |
| -------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                               |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                                |
| _keyword-only_ |                                    |                                                                                                                               |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`xavier_uniform_init`](/docs/api-initializers#xavier_uniform_init). |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                        |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created `Linear` layer.                                                                                                   |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/linear.py
```

### LSTM and BiLSTM {#lstm tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">List[Floats2d]</ndarray>
- **Output:** <ndarray shape="batch_size, nO">List[Floats2d]</ndarray>
- **Parameters:** `depth` <tt>int</tt>, `dropout` <tt>float</tt>

</inline-list>

An LSTM recurrent neural network. The BiLSTM is bidirectional: that is, each
layer concatenated a forward LSTM with an LSTM running in the reverse direction.
If you are able to install PyTorch, you should usually prefer to use the
`PyTorchBiLSTM` layer instead of Thinc's implementations, as PyTorch's LSTM
implementation is significantly faster.

| Argument       | Type                                           | Description                                   |
| -------------- | ---------------------------------------------- | --------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>                         | The size of the output vectors.               |
| `nI`           | <tt>Optional[int]</tt>                         | The size of the input vectors.                |
| _keyword-only_ |                                                |                                               |
| `depth`        | <tt>int</tt>                                   | Number of layers (default 1)                  |
| `dropout`      | <tt>float</tt>                                 | Dropout rate to avoid overfitting (default 0) |
| **RETURNS**    | <tt>Model[List[Floats2d], List[Floats2d]]</tt> | The created LSTM layer(s).                    |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/lstm.py
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
| `nP`           | <tt>int</tt>                       | Number of Maxout pieces (default: 3).                                                                                         |
| _keyword-only_ |                                    |                                                                                                                               |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`xavier_uniform_init`](/docs/api-initializers#xavier_uniform_init). |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                        |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                            |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm), (default: False).                                                  |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created Maxout layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/maxout.py
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
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`xavier_uniform_init`](/docs/api-initializers#xavier_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm), (default: False).                                                 |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created Dense layer.                                                                                                     |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/mish.py
```

### MultiSoftmax {#multisoftmax tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

Neural network layer that predicts several multi-class attributes at once. For
instance, we might predict one class with 6 variables, and another with 5. We
predict the 11 neurons required for this, and then softmax them such that
columns 0-6 make a probability distribution and columns 6-11 make another.

| Argument    | Type                               | Description                      |
| ----------- | ---------------------------------- | -------------------------------- |
| `nOs`       | <tt>Tuple[int, ...]</tt>           | The sizes of the output vectors. |
| `nI`        | <tt>Optional[int]</tt>             | The size of the input vectors.   |
| **RETURNS** | <tt>Model[Floats2d, Floats2d]</tt> | The created multi softmax layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/multisoftmax.py
```

### ParametricAttention {#parametricattention tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Ragged</ndarray>
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
https://github.com/explosion/thinc/blob/develop/thinc/layers/parametricattention.py
```

### PyTorchWrapper {#pytorchwrapper tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Array</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Array</ndarray>

</inline-list>

Wrap a [PyTorch](https://pytorch.org) model, so that it has the same API as
Thinc models. To optimize the model, you'll need to create a PyTorch optimizer
and call `optimizer.step` after each batch.

| Argument        | Type                         | Description        |
| --------------- | ---------------------------- | ------------------ |
| `pytorch_model` | <tt>Any</tt>                 | The PyTorch model. |
| **RETURNS**     | <tt>Model[Array, Array]</tt> | The Thinc model.   |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/pytorchwrapper.py
```

### ReLu {#relu tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Floats2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>

</inline-list>

A dense layer with ReLu activation.

| Argument       | Type                               | Description                                                                                                                  |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nO`           | <tt>Optional[int]</tt>             | The size of the output vectors.                                                                                              |
| `nI`           | <tt>Optional[int]</tt>             | The size of the input vectors.                                                                                               |
| _keyword-only_ |                                    |                                                                                                                              |
| `init_W`       | <tt>Callable</tt>                  | A function to initialize the weights matrix. Defaults to [`xavier_uniform_init`](/docs/api-initializers#xavier_uniform_init) |
| `init_b`       | <tt>Callable</tt>                  | A function to initialize the bias vector. Defaults to [`zero_init`](/docs/api-initializers#zero_init).                       |
| `dropout`      | <tt>Optional[float]</tt>           | Dropout rate to avoid overfitting.                                                                                           |
| `normalize`    | <tt>bool</tt>                      | Whether or not to apply [layer normalization](#layernorm), (default: False).                                                 |
| **RETURNS**    | <tt>Model[Floats2d, Floats2d]</tt> | The created ReLu layer.                                                                                                      |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/relu.py
```

### residual {#residual tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">List[Array]</ndarray>
- **Output:** <ndarray shape="batch_size, nO">List[Array]</ndarray>

</inline-list>

A unary combinator creating a residual connection. This converts a layer
computing `f(x)` into one that computes `f(x)+x`. Gradients flow through
residual connections directly, helping the network to learn more smoothly.

| Argument    | Type                 | Description                                        |
| ----------- | -------------------- | -------------------------------------------------- |
| `layer`     | <tt>Model[T, T]</tt> | A model with the same input and output types.      |
| **RETURNS** | <tt>Model[T, T]</tt> | A model with the unchanged input and output types. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/residual.py
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
https://github.com/explosion/thinc/blob/develop/thinc/layers/softmax.py
```

### SparseLinear {#sparselinear tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Array</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Array</ndarray>
- **Parameters:** <ndarray shape="nO, nI">W</ndarray>,
  <ndarray shape="nO,">b</ndarray>, `length` <tt>int</tt>

</inline-list>

A sparse linear layer using the "hashing trick". Useful for tasks such as text
classification. Inputs to the layer should be a tuple of arrays
`(keys, values, lengths)`, where the `keys` and `values` are arrays of the same
length, describing the concatenated batch of input features and their values.
The `lengths` array should have one entry per sequence in the batch, and the sum
of the lengths should equal the length of the keys and values array.

| Argument    | Type                         | Description                                              |
| ----------- | ---------------------------- | -------------------------------------------------------- |
| `nO`        | <tt>Optional[int]</tt>       | The size of the output vectors.                          |
| `length`    | <tt>int</tt>                 | The size of the weights vector, to be tuned empirically. |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The created layer.                                       |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/sparselinear.pyx
```

### StaticVectors {#staticvectors tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nV">Ints2d</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>
- **Attrs:** `lang` <tt>str</tt>, `column` <tt>int</tt>

</inline-list>

TODO: ...

| Argument       | Type                             | Description                                    |
| -------------- | -------------------------------- | ---------------------------------------------- |
| `lang`         | <tt>str</tt>                     | Language code corresponding to the input data. |
| `nO`           | <tt>int</tt>                     | The size of the output vectors.                |
| _keyword-only_ |                                  |                                                |
| `column`       | <tt>int</tt>                     | The column of values to slice for the indices. |
| **RETURNS**    | <tt>Model[Ints2d, Floats2d]</tt> | The created embedding layer.                   |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/staticvectors.py
```

---

## Pooling layers {#pooling-layers}

### MaxPool {#maxpool tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by selecting the maximum
value for each feature.

| Argument    | Type                             | Description                |
| ----------- | -------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, Floats2d]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/maxpool.py
```

### MeanPool {#meanpool tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by computing the average
value of each feature.

| Argument    | Type                             | Description                |
| ----------- | -------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, Floats2d]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/meanpool.py
```

### SumPool {#sumpool tag="function"}

<inline-list>

- **Input:** <ndarray shape="batch_size, nI">Ragged</ndarray>
- **Output:** <ndarray shape="batch_size, nO">Floats2d</ndarray>

</inline-list>

Pooling layer that reduces the dimensions of the data by computing the sum for
each feature.

| Argument    | Type                             | Description                |
| ----------- | -------------------------------- | -------------------------- |
| **RETURNS** | <tt>Model[Ragged, Floats2d]</tt> | The created pooling layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/sumpool.py
```

---

## Combinators {#combinators}

TODO: ...

### add {#add tag="function"}

Compose two or more models `f`, `g`, etc, such that their outputs are added,
i.e. `add(f, g)(x)` computes `f(x) + g(x)`.

| Argument    | Type                         | Description            |
| ----------- | ---------------------------- | ---------------------- |
| `*layers`   | <tt>Model[Array, Array]</tt> | The models to compose. |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed model.    |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/add.py
```

### bidirectional {#bidirectional tag="function"}

Stitch two RNN models into a bidirectional layer. Expects squared sequences.

| Argument    | Type                         | Description                       |
| ----------- | ---------------------------- | --------------------------------- |
| `l2r`       | <tt>Model[Array, Array]</tt> | The first model.                  |
| `r2l`       | <tt>Optional[Model]</tt>     | The second model.                 |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed bidirectional layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/bidirectional.py
```

### chain {#chain tag="function"}

Compose two models `f` and `g` such that they become layers of a single
feed-forward model that computes `g(f(x))`.

| Argument    | Type                         | Description                      |
| ----------- | ---------------------------- | -------------------------------- |
| `*layers`   | <tt>Model[Array, Array]</tt> | The models to compose.           |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed feed-forward model. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/chain.py
```

### clone {#clone tag="function"}

Construct `n` copies of a layer, with distinct weights. For example,
`clone(f, 3)(x)` computes `f(f'(f''(x)))`.

| Argument    | Type                         | Description                        |
| ----------- | ---------------------------- | ---------------------------------- |
| `orig`      | <tt>Model[Array, Array]</tt> | The layer to copy.                 |
| `n`         | <tt>int</tt>                 | The number of copies to construct. |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed model.                |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/clone.py
```

### concatenate {#concatenate tag="function"}

Compose two or more models `f`, `g`, etc, such that their outputs are
concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`.

| Argument    | Type                         | Description            |
| ----------- | ---------------------------- | ---------------------- |
| `*layers`   | <tt>Model[Array, Array]</tt> | The models to compose. |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed model.    |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/concatenate.py
```

### foreach {#foreach tag="function"}

Map a layer across list items.

| Argument    | Type                         | Description         |
| ----------- | ---------------------------- | ------------------- |
| `layer`     | <tt>Model[Array, Array]</tt> | The layer.          |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed model. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/foreach.py
```

### list2ragged {#list2ragged tag="function"}

Transform sequences to ragged arrays if necessary. If sequences are already
ragged, do nothing. A ragged array is a tuple `(data, lengths)`, where `data` is
the concatenated data.

| Argument    | Type                         | Description         |
| ----------- | ---------------------------- | ------------------- |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed model. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/list2ragged.py
```

### noop {#noop tag="function"}

Transform a sequences of layers into a null operation.

| Argument    | Type                         | Description            |
| ----------- | ---------------------------- | ---------------------- |
| `*layers`   | <tt>Model[Array, Array]</tt> | The models to compose. |
| **RETURNS** | <tt>Model[Array, Array]</tt> | The composed model.    |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/noop.py
```

### recurrent {#recurrent tag="function"}

TODO: ...

| Argument     | Type                         | Description |
| ------------ | ---------------------------- | ----------- |
| `step_model` | <tt>Model[Array, Array]</tt> | TODO: ...   |
| **RETURNS**  | <tt>Model[Array, Array]</tt> | TODO: ...   |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/recurrent.py
```

### siamese {#siamese tag="function"}

TODO: ...

| Argument     | Type                         | Description                |
| ------------ | ---------------------------- | -------------------------- |
| `layer`      | <tt>Model[Array, Array]</tt> | TODO: ...                  |
| `similarity` | <tt>Model[Array, Array]</tt> | TODO: ...                  |
| **RETURNS**  | <tt>Model[Array, Array]</tt> | The created siamese layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/siamese.py
```

### uniqued {#uniqued tag="function"}

Group inputs to a layer, so that the layer only has to compute for the unique
values. The data is transformed back before output, and the same transformation
is applied for the gradient. Effectively, this is a cache local to each
minibatch. The `uniqued` wrapper is useful for word inputs, because common words
are seen often, but we may want to compute complicated features for the words,
using e.g. character LSTM.

| Argument       | Type                         | Description                        |
| -------------- | ---------------------------- | ---------------------------------- |
| `layer`        | <tt>Model[Array, Array]</tt> | TODO: The layer.                   |
| _keyword-only_ |                              |                                    |
| `column`       | <tt>int</tt>                 | TODO: The column. Defaults to `0`. |
| **RETURNS**    | <tt>Model[Array, Array]</tt> | TODO: The model.                   |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/uniqued.py
```

---

## Data type transfers {#transfers}

### list2array {#list2array tag="function"}

Transform sequences to ragged arrays if necessary. If sequences are already
ragged, do nothing. A ragged array is a tuple (data, lengths), where data is the
concatenated data.

| Argument    | Type                                     | Description                              |
| ----------- | ---------------------------------------- | ---------------------------------------- |
| **RETURNS** | <tt>Model[List[Floats2d], Floats2d]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/list2array.py
```

### list2ragged {#list2ragged tag="function"}

Transform sequences to ragged arrays if necessary and return the ragged array.
If sequences are already ragged, do nothing. A ragged array is a tuple
`(data, lengths)`, where `data` is the concatenated data.

| Argument    | Type                                   | Description                              |
| ----------- | -------------------------------------- | ---------------------------------------- |
| **RETURNS** | <tt>Model[List[Floats2d], Ragged]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/list2ragged.py
```

### ragged2list {#ragged2list tag="function"}

Transform sequences from a ragged format into lists.

| Argument    | Type                                | Description                              |
| ----------- | ----------------------------------- | ---------------------------------------- |
| **RETURNS** | <tt>Model[Ragged, List[Array]]</tt> | The layer to compute the transformation. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/ragged2list.py
```

### with_list2array {#with_list2array tag="function"}

TODO: ...

| Argument       | Type                                     | Description                   |
| -------------- | ---------------------------------------- | ----------------------------- |
| `layer`        | <tt>Model[Array, Array]</tt>             | The layer to wrap.            |
| _keyword-only_ |                                          |                               |
| `pad`          | <tt>int</tt>                             | The padding. Defaults to `0`. |
| **RETURNS**    | <tt>Model[List[Array], List[Array]]</tt> | The wrapped layer.            |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/with_list2array.py
```

### with_list2padded {#with_list2padded tag="function"}

TODO: ...

| Argument    | Type                                     | Description        |
| ----------- | ---------------------------------------- | ------------------ |
| `layer`     | <tt>Model[Padded, Padded]</tt>           | The layer to wrap. |
| **RETURNS** | <tt>Model[List[Array], List[Array]]</tt> | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/with_list2padded.py
```

### with_getitem {#with_getitem tag="function"}

Transform data on the way into and out of a layer, by plucking an item from a
tuple.

| Argument    | Type                         | Description                        |
| ----------- | ---------------------------- | ---------------------------------- |
| `idx`       | <tt>int</tt>                 | The index to pluck from the tuple. |
| `layer`     | <tt>Model[Array, Array]</tt> | The layer to wrap.                 |
| **RETURNS** | <tt>Model[Tuple, Tuple]</tt> | The wrapped layer.                 |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/with_getitem.py
```

### with_reshape {#with_reshape tag="function"}

Reshape data on the way into and out from a layer.

| Argument    | Type                               | Description        |
| ----------- | ---------------------------------- | ------------------ |
| `layer`     | <tt>Model[Floats2d, Floats2d]</tt> | The layer to wrap. |
| **RETURNS** | <tt>Model[Floats3d, Floats3d]</tt> | The wrapped layer. |

```python
https://github.com/explosion/thinc/blob/develop/thinc/layers/with_reshape.py
```
