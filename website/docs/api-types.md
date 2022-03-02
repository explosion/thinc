---
title: Types & Dataclasses
teaser: Type annotations, data structures and more
next: /docs/api-backends
---

|                                 |                                                                         |
| ------------------------------- | ----------------------------------------------------------------------- |
| [**Types**](#types)             | Custom type annotations for input/output types available in Thinc.      |
| [**Dataclasses**](#dataclasses) | Data structures for efficient processing, especially for sequence data. |

## Types {#types}

<!-- TODO: write description -->

|                                                            |                                                                             |
| ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| `Floats1d`, `Floats2d`, `Floats3d`, `Floats4d`, `FloatsXd` | 1d, 2d, 3d, 4d and any-d arrays of floats (`DTypesFloat`).                  |
| `Ints1d`, `Ints2d`, `Ints3d`, `Ints4d`, `IntsXd`           | 1d, 2d, 3d, 4d and any-d arrays of ints (`DTypesInt`).                      |
| `Array1d`, `Array2d`, `Array3d`, `Array4d`, `ArrayXd`      | 1d, 2d, 3d, 4d and any-d arrays of floats or ints.                          |
| `List1d`, `List2d`, `List3d`, `List4d`, `ListXd`           | Lists of 1d, 2d, 3d, 4d and any-d arrays (with same-type elements).         |
| `DTypesFloat`                                              | Float data types: `"f"` or `"float32"`.                                     |
| `DTypesInt`                                                | Integer data types: `"i"`, `"int32"`, `"int64"`, `"uint32"`, `"uint64"`.    |
| `DTypes`                                                   | Union of <tt>DTypesFloat</tt> and <tt>DTypesInt</tt>.                       |
| `Shape`                                                    | An array shape. Equivalent to <tt>Tuple[int, ...]</tt>.                     |
| `Xp`                                                       | `numpy` on CPU or `cupy` on GPU. Equivalent to <tt>Union[numpy, cupy]</tt>. |
| `Generator`                                                | Custom type for generators / iterators for better config validation.        |
| `Batchable`                                                | <tt>Union[Pairs, Ragged, Padded, ArrayXd, List, Tuple]</tt>.                |

---

## Dataclasses {#dataclasses}

A dataclass is a **lightweight data structure**, similar in spirit to a named
tuple, defined using the
[`@dataclass`](https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions)
decorator introduced in Python 3.7 (and backported to 3.6). Thinc uses
dataclasses for many situations that would otherwise be written with nested
Python containers. Dataclasses work better with the type system, and often
result in code that's **easier to read and test**.

### Ragged {#ragged tag="dataclass"}

A batch of concatenated sequences, that vary in the size of their first
dimension. `Ragged` allows variable-length sequence data to be contiguous in
memory, without padding. Indexing into `Ragged` is just like indexing into the
`lengths` array, except it returns a `Ragged` object with the accompanying
sequence data. For instance, you can write `ragged[1:4]` to get a `Ragged`
object with sequences `1`, `2` and `3`. Internally, the input data is reshaped
into a two-dimensional array, to allow routines to operate on it consistently.
The original data shape is stored, and the reshaped data is accessible via the
`dataXd` property.

| Member       | Type             | Description                                               |
| ------------ | ---------------- | --------------------------------------------------------- |
| `data`       | <tt>Array2d</tt> | The data array.                                           |
| `dataXd`     | <tt>ArrayXd</tt> | The data array with the original shape.                   |
| `data_shape` | <tt>Shape</tt>   | The original data shape, with -1 for the first dimension. |
| `lengths`    | <tt>Ints1d</tt>  | The sequence lengths.                                     |

### Padded {#padded tag="dataclass"}

A batch of padded sequences, sorted by decreasing length. The auxiliary array
`size_at_t` indicates the length of the batch at each timestep, so you can do
`data[:, :size_at_t[t]]` to shrink the batch. For instance, let's say you have a
batch of four documents, of lengths `[6, 5, 2, 1]`. The `size_at_t` will be
`[4, 3, 3, 3, 2, 1]`. The lengths array indicates the length of each row, and
the indices indicates the original ordering.

<infobox variant="warning">

The `Padded` container is currently limited to two-dimensional array content
(that is, a batch of sequences, where each timestep of each batch is a 2d
array). This restriction will be relaxed in a future release.

</infobox>

| Member      | Type              | Description                                                                                                                                     |
| ----------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`      | <tt>Floats3d</tt> | A three-dimensional array, sorted by decreasing sequence length. The dimensions are timestep, batch item, row data.                             |
| `site_at_t` | <tt>Ints1d</tt>   | An array indicating how the batch can be truncated at different sequence lengths. You can do `data[:, :size_at_t[t]]` to get an unpadded batch. |
| `lengths`   | <tt>Ints1d</tt>   | The sequence lengths. Applies to the reordered sequences, not the original ordering. So it'll be decreasing length.                             |
| `indices`   | <tt>Ints1d</tt>   | Lists of indices indicating how to put the items back into original order.                                                                      |

### Pairs {#pairs}

A batch of paired data, for instance images and their captions, or pairs of
texts to compare. Indexing operations are performed as though the data were
transposed to make the batch the outer dimension. For instance, `pairs[:3]` will
return `Pairs(pairs.one[:3], pairs.two[:3])`, i.e. a slice of the batch with the
first three items, as a new `Pairs` object.

```python
### Example
from thinc.types import Pairs

pairs = Pairs([1, 2, 3, 4], [5, 6, 7, 8])
assert pairs.one == [1, 2, 3, 4]
assert pairs[2] == Pairs(3, 7)
assert pairs[2:4] == Pairs([3, 4], [7, 8])
```

| Member | Type              | Description          |
| ------ | ----------------- | -------------------- |
| `one`  | <tt>Sequence</tt> | The first sequence.  |
| `two`  | <tt>Sequence</tt> | The second sequence. |

### SizedGenerator {#sizedgenerator tag="dataclass"}

A custom dataclass for a generator that has a `__len__` and can repeatedly call
the generator function. This is especially useful for batching (see
[`Ops.minibatch`](/docs/api-backends#minibatch)) where you know the length of
the data upfront, but still want to batch it as a stream and return a generator.
Exposing a `__len__` attribute also makes it work seamlessly with progress bars
like [`tqdm`](https://github.com/tqdm/tqdm) and similar tools.

<infobox variant="warning">

The underlying generator function is called _every time_ the sized generator is
executed and won't be consumed. This allows defining the batching outside of the
training loop. On each iteration, the data will be reshuffled and rebatched.

</infobox>

```python
### Example
train_data = model.ops.multibatch(128, train_X, train_Y, shuffle=True)
assert isinstance(train_data, SizedGenerator)
for i in range(10):
    for X, Y in tqdm(train_data, leave=False):
        Yh, backprop = model.begin_update(X)
```

| Member      | Type                             | Description                                                    |
| ----------- | -------------------------------- | -------------------------------------------------------------- |
| `get_items` | <tt>Callable[[], Generator]</tt> | The generator function. Available via the `__iter__` method.   |
| `length`    | <tt>int</tt>                     | The length of the data. Available via the `__len__` attribute. |

### ArgsKwargs {#argskwargs tag="dataclass"}

A tuple of `(args, kwargs)` that can be spread into some function f:
`f(*args, **kwargs)`. Makes it easier to handle positional and keyword arguments
that get passed around, especially for integrating custom models via a
[`Shim`](/docs/api-model#shims).

| Member   | Type                     | Description                                                                    |
| -------- | ------------------------ | ------------------------------------------------------------------------------ |
| `args`   | <tt>Tuple[Any, ...]</tt> | The positional arguments. Can be passed into a function as `*ArgsKwargs.args`. |
| `kwargs` | <tt>Dict[str, Any]</tt>  | The keyword arguments. Can be passed into a function as `**ArgsKwargs.kwargs`. |

#### ArgsKwargs.from_items {#argskwargs-from_items tag="classmethod"}

Create an `ArgsKwargs` object from a sequence of `(key, value)` tuples, such as
produced by `ArgsKwargs.items`. Each key should be either a string or an
integer. Items with integer keys are added to the `args`, and items with string
keys are added to the `kwargs`. The `args` are determined by sequence order, not
the value of the integer.

```python
### Example
from thinc.api import ArgsKwargs

items = [(0, "value"), ("key", "other value"), (1, 15), ("foo", True)]
ak = ArgsKwargs.from_items(items)
assert ak.args == ("value", 15)
assert ak.kwargs == {"key": "other value", "foo": True}
```

| Argument    | Type                                           | Description                 |
| ----------- | ---------------------------------------------- | --------------------------- |
| `items`     | <tt>Sequence[Tuple[Union[int, str], Any]]</tt> | The items.                  |
| **RETURNS** | <tt>ArgsKwargs</tt>                            | The `ArgsKwargs` dataclass. |

#### ArgsKwargs.keys {#argskwargs-keys tag="method"}

Yield indices from `ArgsKwargs.args`, followed by keys from `ArgsKwargs.kwargs`.

| Argument   | Type                     | Description                            |
| ---------- | ------------------------ | -------------------------------------- |
| **YIELDS** | <tt>Union[int, str]</tt> | The keys, `args` followed by `kwargs`. |

#### ArgsKwargs.values {#argskwargs-values tag="method"}

Yield values from `ArgsKwargs.args`, followed by keys from `ArgsKwargs.kwargs`.

| Argument   | Type         | Description                              |
| ---------- | ------------ | ---------------------------------------- |
| **YIELDS** | <tt>Any</tt> | The values, `args` followed by `kwargs`. |

#### ArgsKwargs.items {#argskwargs-items tag="method"}

Yield `enumerate(ArgsKwargs.args)`, followed by `ArgsKwargs.kwargs.items()`.

| Argument   | Type                                 | Description                              |
| ---------- | ------------------------------------ | ---------------------------------------- |
| **YIELDS** | <tt>Tuple[Union[int, str], Any]</tt> | The values, `args` followed by `kwargs`. |
