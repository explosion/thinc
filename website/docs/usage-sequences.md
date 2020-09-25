---
title: Variable-length sequences
teaser: Dataclasses for ragged, padded, paired and list-based sequences
next: /docs/usage-type-checking
---

Thinc's built-in layers support several ways to **encode variable-length
sequence data**. The encodings are designed to avoid losing information, so you
can compose operations smoothly and easily build hierarchical models over
structured inputs. This page provides a summary of the different formats, their
advantages and disadvantages, and a summary of which built-in layers accept and
return them. There's no restrictions on what objects your own models can accept
and return, so you're free to invent your own data types.

## Background and motivation {#background}

The
[`numpy.ndarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
object represents a multi-dimensional table of data. In the simplest case,
there's only one dimension, or "axis", so the size of the array is equal to its
length:

```python
array1d = numpy.ndarray((10,))
assert array1d.size == 10
assert array1d.shape == (10,)
```

To make a two-dimensional array, we can instead write
`array2d = numpy.ndarray((10, 16))`. This will be a table with 10 rows and 16
columns, and a total size of 160 items. However, the `ndarray` object does not
have a native way to represent data with a **variable number of columns per
row**. If the last row of your data only has 15 items rather than 16, you cannot
create a two-dimensional array with only 159 items, where rows one to nine have
16 columns and the last row has 15 columns. The limitation makes a lot of sense:
the rest of the `numpy` API presents operations defined in terms of
regularly-shaped arrays, and there's often no obvious generalization to
irregularly shaped data.

While you could not represent the 159 items in a two-dimensional array, there's
no reason why you couldn't keep the data together in a flat format, all in one
dimension. You could keep track of the intended number of columns separately,
and reshape the data to do various operations according to your intended
definitions. This is essentially the approach that Thinc takes for
variable-length sequences.

Inputs very often are irregularly shaped. For instance, texts vary in length,
often significantly. You might also want to represent your texts hierarchically:
each word can be seen as a variable-length sequence of characters, each sentence
a variable-length sequence of words, each paragraph a variable-length sequence
of sentences, and each text a variable-length sequence of paragraphs. A single,
padded `ndarray` is a poor choice for this type of hierarchical representation,
as each dimension would need to be padded to its longest item. If the longest
word in your batch is 10 characters, you will need to use 10 characters for
every word. If the longest sentence in your batch has 40 words, every sentence
will need to be 40 words. The inefficiency will be multiplied along each
dimension, so that the vast majority of the final structure is empty space.

Unfortunately, there is no single best solution that is most efficient for every
situation. It depends on the shapes of the data, and the hardware being used. On
GPU devices, it is often better to use **padded representations**, so long as
there is only padding along one dimension. However, on CPU, **denser
representations** are often more efficient, as maintaining parallelism is less
important. Different libraries also introduce different considerations. For
[JAX](https://github.com/google/jax), operations over irregularly-sized arrays
are extremely expensive, as a new kernel will need to be compiled for every
combination of shapes you provide.

Thinc therefore provides a number of **different sequence formats**, with
utility layers that convert between them. Thinc also provides layers that
represent reversible transformations. The
[`with_*` layers](/docs/api-layers#with_array) accept a layer as an argument,
and transform inputs on the way into the layer, and then perform the opposite
transformation on the way out. For instance, the
[`with_padded`](/docs/api-layers#with_padded) wrapper will allow you to
temporarily convert to a [`Padded`](/docs/api-types#padded) representation, for
the scope of the layer being wrapped.

---

## Padded {#padded}

The [`Padded`](/docs/api-types#padded) dataclass represents a **padded batch of
sequences** sorted by descending length. The data is formatted in "sequence
major" format, i.e. the first dimension represents the sequence position, and
the second dimension represents the batch index. The `Padded` type uses three
auxiliary integer arrays, to keep track of the actual sequence lengths and the
original positions, so that the original structure can be restored. The third
auxiliary array, `size_at_t`, allows the padded batch to be sliced to currently
active sequences at different time steps.

Although the underlying data is sequence-major, the `Padded` dataclass supports
**getting items** or **slices along the batch dimension**: you can write
`padded[1:3]` to retrieve a `Padded` object with sequence items one and two. The
`Padded` format is well-suited for LSTM and other RNN models.

```python
### Example
from thinc.api import get_current_ops, Padded

ops = get_current_ops()
sequences = [
    ops.alloc2f(7, 5) + 1,
    ops.alloc2f(2, 5) + 2,
    ops.alloc2f(4, 5) + 3,
]
padded = ops.list2padded(sequences)
assert padded.data.shape == (7, 3, 5)
# Data from sequence 0 is first, as it was the longest
assert padded.data[:, 0] == 1
# Data from sequence 2 is second, and it's padded on dimension 0
assert padded.data[:4, 1] == 3
# Data from sequence 1 is third, also padded on dimension 0
assert padded.data[:2, 2] == 2
# Original positions
assert list(padded.indices) == [0, 2, 1]
# Slices refer to batch index.
assert isinstance(padded[0], Padded)
assert padded[0].data.shape == (7, 1, 5)
```

|                |                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Operations** | [`Ops.list2padded`](/docs/api-backends#list2padded), [`Ops.padded2list`](/docs/api-backends#padded2list)                                    |
| **Transforms** | [`padded2list`](/docs/api-layers#padded2list), [`list2padded`](/docs/api-layers#list2padded), [`with_padded`](/docs/api-layers#with_padded) |
| **Layers**     | [`LSTM`](/docs/api-layers#lstm), [`PyTorchLSTM`](/docs/api-layers#lstm)                                                                     |

## Ragged {#ragged}

The [`Ragged`](/docs/api-types#ragged) dataclass represents a **concatenated
batch of sequences**. An auxiliary array is used to keep track of the lengths.
The `Ragged` format is memory efficient, and is efficient for some operations.
However, it is not supported directly by most underlying operations.
Per-sequence operations such as sequence transposition and matrix multiplication
are relatively expensive, but Thinc does support custom CPU and CUDA kernels for
more efficient reduction (aka. pooling) operation on ragged arrays.

The `Ragged` format makes it easy to ignore the sequence structure of your data
for some operations, such as word embeddings or feed-forward layers. These
layers do not accept the `Ragged` object directly, but you can wrap the layer
using the [`with_array`](/docs/api-layers#with_array) transform to make them
compatible without requiring copy operations. The `with_array` transform will
pass the underlying array data into the layer, and return the outputs as a
`Ragged` object so that the sequence information remains available to the rest
of your network.

```python
### Example
from thinc.api import Ragged

from thinc.api import get_current_ops, Ragged, Linear

ops = get_current_ops()
sequences = [
    ops.alloc2f(7, 5) + 1,
    ops.alloc2f(2, 5) + 2,
    ops.alloc2f(4, 5) + 3,
]
ragged = ops.list2ragged(sequences)
assert ragged.data.shape == (13, 5)
# This will always be true:
assert ragged.data.shape[0] == ragged.lengths.sum()
# Data from sequence 0 is in the first 7 rows, followed by seqs 1 and 2
assert ragged.data[:7] == 1
assert ragged.data[7:2] == 2
assert ragged.data[9:] == 3
# Indexing gets the batch item, and returns a Ragged object
ragged[0].data.shape == (7, 5)
# You can pass the data straight into dense layers
model = Linear(6, 5).initialize()
output = model.predict(ragged.data)
ragged_out = Ragged(output, ragged.lengths)
# Internally, data is reshaped to 2d. The original shape is accessible at the
# the dataXd property.
sequences3d = [ops.alloc3f(5, 6, 7), ops.alloc3f(10, 6, 7)]
ragged3d = ops.list2ragged(sequences3d)
ragged3d.data.shape == (15, 13)
ragged3d.dataXd.shape == (15, 6, 7)
```

|                |                                                                                                                                                                                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Operations** | [`Ops.ragged2list`](/docs/api-backends#ragged2list), [`Ops.list2ragged`](/docs/api-backends#list2ragged), [`Ops.reduce_sum`](/docs/api-backends#reduce_sum), [`Ops.reduce_mean`](/docs/api-backends#reduce_sum), [`Ops.reduce_max`](/docs/api-backends#reduce_sum) |
| **Transforms** | [`with_ragged`](/docs/api-layers#with_ragged), [`ragged2list`](/docs/api-layers#ragged2list), [`list2ragged`](/docs/api-layers#list2ragged)                                                                                                                        |
| **Layers**     | [`reduce_sum`](/docs/api-layers#reduce_sum), [`reduce_mean`](/docs/api-layers#reduce_mean), [`reduce_max`](/docs/api-layers#reduce_max)                                                                                                                            |

## List[ArrayXd] {#list-array}

A list of arrays is often the most convenient input and output format for
sequence data, especially for runtime usage of the model. However, most
mathematical operations require the data to be passed in **as a single array**,
so you will usually need to transform the array list into another format to pass
it into various layers. A common pattern is to use `list2padded` or
`list2ragged` as the first layer of your network, and `ragged2list` or
`padded2list` as the final layer. You could then opt to strip these from the
network during training, so that you can make the transformation just once at
the beginning of training. However, this does mean that you'll be training on
the same batches of data in each epoch, which may lead to lower accuracies.

<!-- TODO: example of network with transforms? -->

|                |                                                                                                                                                                                                                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Operations** | [`Ops.ragged2list`](/docs/api-backends#ragged2list), [`Ops.list2ragged`](/docs/api-backends#list2ragged) , [`Ops.padded2list`](/docs/api-backends#padded2list), [`Ops.list2padded`](/docs/api-backends#list2padded)                                                                      |
| **Transforms** | [`ragged2list`](/docs/api-layers#ragged2list), [`list2ragged`](/docs/api-layers#list2ragged), [`padded2list`](/docs/api-layers#padded2list), [`list2padded`](/docs/api-layers#list2padded), [`with_ragged`](/docs/api-layers#with_ragged), [`with_padded`](/docs/api-layers#with_padded) |
| **Layers**     | [`reduce_sum`](/docs/api-layers#reduce_sum), [`reduce_mean`](/docs/api-layers#reduce_mean), [`reduce_max`](/docs/api-layers#reduce_max)                                                                                                                                                  |

## List[List[Any]] {#nested-list}

Nested lists are a useful format for many types of **hierarchically structured
data**. Often you'll need to write your own layers for these situations, but
Thinc does have a helpful utility tool, the `with_flatten` transform. This
transform can be applied around a layer in your network, and the layer will be
called with a flattened representation of your list data. The outputs are then
repackaged into lists, with arrays divided as needed.

<!-- TODO: example of network with transforms? -->

|                |                                                 |
| -------------- | ----------------------------------------------- |
| **Transforms** | [`with_flatten`](/docs/api-layers#with_flatten) |

<!-- TODO:

## List[Any] {#object-list}

-->

## Array {#array}

Most Thinc layers that work on sequences do **not** expect plain arrays, because
the array does not include any representation of where the sequences begin and
end, which makes the semantics of some operations unclear. For instance, there's
no way to accurately do mean pooling on an array of padded sequences without
knowing where the sequences actually end. Max pooling is often also difficult,
depending on the padding value. If the array represents sequences, you should
**maintain the metadata** to treat it as an intelligible sequence.
