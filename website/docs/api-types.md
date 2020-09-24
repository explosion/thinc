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

TODO: ...

|             |                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------- |
| `Xp`        | `numpy` on CPU or `cupy` on GPU. Equivalent to <tt>Union[numpy, cupy]</tt>.                        |
| `Array`     | Any `numpy.ndarray` or `cupy.ndarray`.                                                             |
| `Floats1d`  | 1-dimensional array of floats.                                                                     |
| `Floats2d`  | 2-dimensional array of floats.                                                                     |
| `Floats3d`  | 3-dimensional array of floats.                                                                     |
| `Floats4d`  | 4-dimensional array of floats.                                                                     |
| `FloatsNd`  | N-dimensional array of floats.                                                                     |
| `Ints1d`    | 1-dimensional array of ints.                                                                       |
| `Ints2d`    | 2-dimensional array of ints.                                                                       |
| `Ints3d`    | 3-dimensional array of ints.                                                                       |
| `Ints4d`    | 4-dimensional array of ints.                                                                       |
| `IntsNd`    | N-dimensional array of ints.                                                                       |
| `RNNState`  | TODO: ...                                                                                          |
| `Shape`     | An array shape. Equivalent to <tt>Tuple[int, ...]</tt>.                                            |
| `Device`    | [Backend](/docs/api-backends) identifier: `"numpy"`, `"cupy"`, `"cpu"`, `"gpu"` or a device index. |
| `Generator` | Custom type for generators / iterators for better config validation.                               |

<infobox variant="warning">

The types for arrays of different dimensions and data types, e.g. `Floats1d` or
`Ints4d`, are currently only used for consistency and to make it easier to tell
the shapes of arrays that are being passed around. If you pass a type `Floats2d`
as an argument annotated as `Floats1d`, you'll see a type error. However, it
doesn't statically check whether the array is _actually_ a two-dimensional array
of floats. This level of validation is currently only available **at runtime**,
if you use the types as part of a
[`pydantic` model](/docs/usage-config#advanced-types).

</infobox>

---

## Dataclasses {#dataclasses}

TODO: ...

### Ragged {#ragged tag="dataclass"}

TODO: ...

```python
### Example
# TODO: write
```

| Member    | Type           | Description     |
| --------- | -------------- | --------------- |
| `data`    | <tt>Array</tt> | The data array. |
| `lengths` | <tt>Array</tt> | TODO: ...       |

### Padded {#padded tag="dataclass"}

A batch of padded sequences, sorted by decreasing length. The `data` array is of
shape `(step, batch, ...)`. The auxiliary array `size_at_t` indicates the length
of the batch at each timestep, so you can do `data[:, :size_at_t[t]]` to shrink
the batch.

```python
### Example
# TODO: write
```

| Member      | Type           | Description                                   |
| ----------- | -------------- | --------------------------------------------- |
| `data`      | <tt>Array</tt> | The data array of shape `(step, batch, ...)`. |
| `site_at_t` | <tt>Array</tt> | Length of the batch at each timestep.         |
