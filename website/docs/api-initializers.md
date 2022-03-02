---
title: Initializers
next: /docs/api-schedules
---

A collection of initialization functions. Parameter initialization schemes can
be very important for deep neural networks, because the initial distribution of
the weights helps determine whether activations change in mean and variance as
the signal moves through the network. If the activations are not stable, the
network will not learn effectively. The "best" initialization scheme changes
depending on the activation functions being used, which is why a variety of
initializations are necessary. You can reduce the importance of the
initialization by using normalization after your hidden layers.

### normal_init {#normal_init tag="function"}

Initialize from a normal distribution, with `scale = sqrt(1 / fan_in)`.

| Argument       | Type              | Description                                                                       |
| -------------- | ----------------- | --------------------------------------------------------------------------------- |
| `ops`          | <tt>Ops</tt>      | The backend object, e.g. `model.ops`.                                             |
| `shape`        | <tt>Shape</tt>    | The data shape.                                                                   |
| _keyword-only_ |                   |                                                                                   |
| `fan_in`       | <tt>int</tt>      | Usually the number of inputs to the layer. If `-1`, the second dimension is used. |
| **RETURNS**    | <tt>FloatsXd</tt> | The initialized array.                                                            |

### glorot_uniform_init {#glorot_uniform_init tag="function"}

Initialize with the randomization introduced by Xavier Glorot
([Glorot and Bengio, 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf),
which is a uniform distribution centered on zero, with
`scale = sqrt(6.0 / (data.shape[0] + data.shape[1]))`. Usually used in
[`Relu`](/docs/api-layers#relu) layers.

| Argument    | Type              | Description                           |
| ----------- | ----------------- | ------------------------------------- |
| `ops`       | <tt>Ops</tt>      | The backend object, e.g. `model.ops`. |
| `shape`     | <tt>Shape</tt>    | The data shape.                       |
| **RETURNS** | <tt>FloatsXd</tt> | The initialized array.                |

### zero_init {#zero_init tag="function"}

Initialize a parameter with zero weights. This is usually used for output layers
and for bias vectors.

| Argument    | Type              | Description                           |
| ----------- | ----------------- | ------------------------------------- |
| `ops`       | <tt>Ops</tt>      | The backend object, e.g. `model.ops`. |
| `shape`     | <tt>Shape</tt>    | The data shape.                       |
| **RETURNS** | <tt>FloatsXd</tt> | The initialized array.                |

### uniform_init {#uniform_init tag="function"}

Initialize values from a uniform distribution. This is usually used for word
embedding tables.

| Argument       | Type              | Description                              |
| -------------- | ----------------- | ---------------------------------------- |
| `ops`          | <tt>Ops</tt>      | The backend object, e.g. `model.ops`.    |
| `shape`        | <tt>Shape</tt>    | The data shape.                          |
| _keyword-only_ |                   |                                          |
| `lo`           | <tt>float</tt>    | The minimum of the uniform distribution. |
| `hi`           | <tt>float</tt>    | The maximum of the uniform distribution. |
| **RETURNS**    | <tt>FloatsXd</tt> | The initialized array.                   |

---

## Usage via config and function registry {#registry}

Since the initializers need to be called with data, defining them in the
[config](/docs/usage-config) will return a **configured function**: a partial
with only the settings (the keyword arguments) applied. Within your script, you
can then pass in the data, and the configured function will be called using the
settings defined in the config.

Most commonly, the initializer is passed as an argument to a
[layer](/docs/api-layers), so it can be defined as its own config block nested
under the layer settings:

<grid>

```ini
### config.cfg {small="true"}
[model]
@layers = "linear.v1"
nO = 10

[model.init_W]
@initializers = "normal_init.v1"
fan_in = -1
```

```python
### Usage {small="true"}
from thinc.api import registry, Config

config = Config().from_disk("./config.cfg")
resolved = registry.resolve(config)
model = resolved["model"]
```

</grid>

You can also define it as a regular config setting and then call the configured
function in your script:

<grid>

```ini
### config.cfg {small="true"}
[initializer]
@initializers = "uniform_init.v1"
lo = -0.1
hi = 0.1
```

```python
### Usage {small="true"}
from thinc.api import registry, Config, NumpyOps

config = Config().from_disk("./config.cfg")
resolved = registry.resolve(config)
initializer = resolved["initializer"]
weights = initializer(NumpyOps(), (3, 2))
```

</grid>
