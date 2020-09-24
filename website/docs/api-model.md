---
title: Model
next: /docs/api-layers
---

Thinc uses just one model class for almost all layer types. Instead of creating
subclasses, you'll usually instantiate the `Model` class directly, passing in a
`forward` function that actually performs the computation. You can find examples
of this in the library itself, in `thinc.layers` (also see the
[layers documentation](/docs/api-layers)).

|                                 |                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| [**Model**](#model)             | Class for implementing Thinc models and layers.                                                  |
| [**Utilities**](#util)          | Helper functions for implementing models.                                                        |
| [**Shim**](#shim)               | Interface for external models. Users can create subclasses of `Shim` to wrap external libraries. |
| [**PyTorchShim**](#pytorchshim) | Interface between a [PyTorch](https://pytorch.org) model and a Thinc `Model`.                    |

## Model {#model tag="class"}

Class for implementing Thinc models and layers.

<infobox variant="warning">

There's only one `Model` class in Thinc and layers are built using
**composition**, not inheritance. This means that a [layer](/docs/api-layers) or
composed model will return an **instance** of `Model` – it doesn't subclass it.
To read more about this concept, see the pages on
[Thinc's philosophy](/docs/philosophy) and
[defining models](/docs/usage-models).

</infobox>

### Attributes {#attributes}

| Name     | Type                              | Description                       |
| -------- | --------------------------------- | --------------------------------- |
| `name`   | <tt>str</tt>                      | The name of the layer type.       |
| `ops`    | <tt>Union[NumpyOps, CupyOps]</tt> | Perform array operations.         |
| `id`     | <tt>int</tt>                      | ID number for the model instance. |
| `layers` | <tt>List[Model]</tt>              | List of child models.             |

### Properties {#properties}

| Name          | Type                     | Description                                                                                                                                                                                                                                                                                         |
| ------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layers`      | <tt>List[Model]</tt>     | A list of child layers of the model. You can use the list directly, including modifying it in-place: the standard way to add a layer to a model is simply `model.layers.append(layer)`. However, you cannot reassign the `model.layers` attribute to a new variable: `model.layers = []` will fail. |
| `shims`       | <tt>List[Shim]</tt>      | A list of child shims added to the model.                                                                                                                                                                                                                                                           |
| `param_names` | <tt>Tuple[str, ...]</tt> | Get the names of registered parameter (including unset).                                                                                                                                                                                                                                            |
| `grad_names`  | <tt>Tuple[str, ...]</tt> | Get the names of parameters with registered gradients (including unset).                                                                                                                                                                                                                            |
| `dim_names`   | <tt>Tuple[str, ...]</tt> | Get the names of registered dimensions (including unset).                                                                                                                                                                                                                                           |
| `attr_names`  | <tt>Tuple[str, ...]</tt> | Get the names of attributes.                                                                                                                                                                                                                                                                        |
| `ref_names`   | <tt>Tuple[str, ...]</tt> | Get the names of registered node references (including unset).                                                                                                                                                                                                                                      |

### Model.\_\_init\_\_ {#init tag="method"}

Initialize a new model.

```python
### Example
model = Model(
    "linear",
    linear_forward,
    init=linear_init,
    dims={"nO": nO, "nI": nI},
    params={"W": None, "b": None},
)
```

| Argument       | Type                                        | Description                                                                             |
| -------------- | ------------------------------------------- | --------------------------------------------------------------------------------------- |
| `name`         | <tt>str</tt>                                | The name of the layer type.                                                             |
| `forward`      | <tt>Callable</tt>                           | Function to compute the forward result and the backpropagation callback.                |
| _keyword-only_ |                                             |                                                                                         |
| `init`         | <tt>Callable</tt>                           | Function to define the initialization logic.                                            |
| `dims`         | <tt>Dict[str, Optional[int]]</tt>           | Dictionary describing the model's dimensions. Map unknown dimensions to `None`.         |
| `params`       | <tt>Dict[str, Optional[bool]]</tt>          | Dictionary with the model's parameters. Set currently unavailable parameters to `None`. |
| `grads`        | <tt>Dict[str, Optional[Array]]</tt>         | Dictionary with initial gradients, if any. Keys should be the parameter names.          |
| `layers`       | <tt>List[Model]</tt>                        | List of child layers.                                                                   |
| `shims`        | <tt>List[Shim]</tt>                         | List of interfaces for external models.                                                 |
| `attrs`        | <tt>Dict[str, Any]</tt>                     | Dictionary of non-parameter attributes.                                                 |
| `ops`          | <tt>Optional[Union[NumpyOps, CupyOps]]</tt> | An `Ops` instance, which provides mathematical and memory operations.                   |

### Model.define_operators {#define_operators tag="classmethod,contextmanager"}

Bind arbitrary binary functions to Python operators, for use in any `Model`
instance. The method can (and should) be used as a contextmanager, so that the
overloading is limited to the immediate block. This allows concise and
expressive model definition.

```python
### Example
from thinc.api import Model, Relu, Softmax, chain

with Model.define_operators({">>": chain}):
    model = ReLu(512) >> ReLu(512) >> Softmax()
```

| Argument    | Type                         | Description                    |
| ----------- | ---------------------------- | ------------------------------ |
| `operators` | <tt>Dict[str, Callable]</tt> | Functions mapped to operators. |

### Model.initialize {#initialize tag="method"}

Finish initialization of the model, optionally providing a batch of example
input and output data to perform shape inference. Until `Model.initialize` is
called, the model may not be in a ready state to operate on data: parameters or
dimensions may be unset. The `Model.initialize` method will usually delegate to
the `init` function given to [`Model.__init__`](#init).

```python
### Example
from thinc.api import Linear, zero_init
import numpy

X = numpy.zeros((128, 16), dtype="f")
Y = numpy.zeros((128, 10), dtype="f")
model = Linear(init_W=zero_init)
model.initialize(X=X, Y=Y)
```

| Argument    | Type                   | Description                      |
| ----------- | ---------------------- | -------------------------------- |
| `X`         | <tt>Optional[Any]</tt> | An example batch of input data.  |
| `Y`         | <tt>Optional[Any]</tt> | An example batch of output data. |
| **RETURNS** | <tt>Model</tt>         | The model instance.              |

### Model.\_\_call\_\_ {#call tag="method"}

Call the model's `forward` function, returning the output and a callback to
compute the gradients via backpropagation.

```python
### Example
from thinc.api import Linear
import numpy

X = numpy.zeros((128, 10), dtype="f")
model = Linear(10)
model.initialize(X=X)
Y, backprop = model(X)
```

| Argument    | Type                          | Description                                                                                         |
| ----------- | ----------------------------- | --------------------------------------------------------------------------------------------------- |
| `X`         | <tt>Any</tt>                  | A batch of input data.                                                                              |
| `is_train`  | <tt>bool</tt>                 | A boolean indicating whether the model is running in a training (as opposed to prediction) context. |
| **RETURNS** | <tt>Tuple[Any, Callable]</tt> | A batch of output data and the `backprop` callback.                                                 |

### Model.begin_update {#begin_update tag="method"}

Call the model's `forward` function with `is_train=True`, and return the output
and the backpropagation callback, which is a function that takes a gradient of
outputs and returns the corresponding gradient of inputs. The backpropagation
callback may also increment the gradients of the model parameters, via calls to
[`Model.inc_grad`](#inc_grad).

```python
### Example
from thinc.api import Linear
import numpy

X = numpy.zeros((128, 10), dtype="f")
model = Linear(10)
model.initialize(X=X)
Y, backprop = model.begin_update(X)
```

| Argument    | Type                          | Description                                         |
| ----------- | ----------------------------- | --------------------------------------------------- |
| `X`         | <tt>Any</tt>                  | A batch of input data.                              |
| **RETURNS** | <tt>Tuple[Any, Callable]</tt> | A batch of output data and the `backprop` callback. |

### Model.predict {#predict tag="method"}

Call the model's `forward` function with `is_train=False`, and return only the
output, instead of the `(output, callback)` tuple.

```python
### Example
from thinc.api import Linear
import numpy

X = numpy.zeros((128, 10), dtype="f")
model = Linear(10)
model.initialize(X=X)
Y = model.predict(X)
```

| Argument    | Type         | Description             |
| ----------- | ------------ | ----------------------- |
| `X`         | <tt>Any</tt> | A batch of input data.  |
| **RETURNS** | <tt>Any</tt> | A batch of output data. |

### Model.finish_update {#finish_update tag="method"}

Update parameters using the current parameter gradients. The
[`Optimizer` instance](/docs/api-optimizers) contains the functionality to
perform the stochastic gradient descent.

```python
### Example
from thinc.api import Adam

optimizer = Adam()
model = Linear(10)
model.finish_update(optimizer)
```

| Argument    | Type               | Description                                                                   |
| ----------- | ------------------ | ----------------------------------------------------------------------------- |
| `optimizer` | <tt>Optimizer</tt> | The optimizer, which is called with each parameter and gradient of the model. |

### Model.use_params {#use_params tag="contextmanager"}

Contextmanager to temporarily set the model's parameters to specified values.

```python
### Example
TODO: write
```

| Argument | Type                      | Description                                                                |
| -------- | ------------------------- | -------------------------------------------------------------------------- |
| `params` | <tt>Dict[int, Array]</tt> | A dictionary keyed by model IDs, whose values are arrays of weight values. |

### Model.walk {#walk tag="method"}

Iterate out layers of the model, breadth-first.

```python
### Example
from thinc.api import LSTM

model = LSTM(1, 2)
for node in model.walk():
    print(node.name)
```

| Argument    | Type                     | Description              |
| ----------- | ------------------------ | ------------------------ |
| **RETURNS** | <tt>Iterable[Model]</tt> | The layers of the model. |

### Model.remove_node {#remove_node tag="method"}

Remove a node from all layers lists, and then update references. References that
no longer point to a node within the tree will be set to `None`. For instance,
if a node has its grandchild as a reference and the child is removed, the
grandchild reference will be left dangling, so will be set to `None`.

```python
### Example
TODO: write
```

| Argument | Type           | Description         |
| -------- | -------------- | ------------------- |
| `node`   | <tt>Model</tt> | The node to remove. |

### Model.has_dim {#has_dim tag="method"}

Check whether the model has a dimension of a given name. If the dimension is
registered but the value is unset, returns `None`.

```python
### Example
from thinc.api import Linear
import numpy

model = Linear(10)
assert model.has_dim("nI") is None
model.initialize(X=numpy.zeros((128, 16), dtype="f"))
assert model.has_dim("nI") is True
```

| Argument    | Type                    | Description                                   |
| ----------- | ----------------------- | --------------------------------------------- |
| `name`      | <tt>str</tt>            | The name of the dimension, e.g. `"nO"`.       |
| **RETURNS** | <tt>Optional[bool]</tt> | A ternary value (`True`, `False`, or `None`). |

### Model.get_dim {#get_dim tag="method"}

Retrieve the value of a dimension of the given name. Raises a `KeyError` if the
dimension is either unregistered or the value is currently unset.

```python
### Example
from thinc.api import Linear
import numpy

model = Linear(10)
model.initialize(X=numpy.zeros((128, 16), dtype="f"))
assert model.get_dim("nI") == 16
```

| Argument    | Type         | Description                             |
| ----------- | ------------ | --------------------------------------- |
| `name`      | <tt>str</tt> | The name of the dimension, e.g. `"nO"`. |
| **RETURNS** | <tt>int</tt> | The size of the dimension.              |

### Model.set_dim {#set_dim tag="method"}

Set a value for a dimension.

```python
### Example
from thinc.api import Linear
import numpy

model = Linear(10)
model.set_dim("nI", 16)
assert model.get_dim("nI") == 16
```

| Argument | Type         | Description                       |
| -------- | ------------ | --------------------------------- |
| `name`   | <tt>str</tt> | The name of the dimension to set. |
| `value`  | <tt>int</tt> | The new value for the dimension.  |

### Model.has_param {#has_param tag="method"}

Check whether the model has a weights parameter of the given name. Returns
`None` if the parameter is registered but currently unset.

```python
### Example
from thinc.api import Linear, zero_init
import numpy

model = Linear(10, init_W=zero_init)
assert model.has_param("W") is None
model.initialize(X=numpy.zeros((128, 16), dtype="f"))
assert model.has_param("W") is True
```

| Argument    | Type                    | Description                                   |
| ----------- | ----------------------- | --------------------------------------------- |
| `name`      | <tt>str</tt>            | The name of the parameter.                    |
| **RETURNS** | <tt>Optional[bool]</tt> | A ternary value (`True`, `False`, or `None`). |

### Model.get_param {#get_param tag="method"}

Retrieve a weights parameter by name. Raises a `KeyError` if the parameter is
unregistered or its value is undefined.

```python
### Example
from thinc.api import Linear, zero_init
import numpy

model = Linear(10, init_W=zero_init)
assert model.has_param("W") is None
model.initialize(X=numpy.zeros((128, 16), dtype="f"))
W = model.get_param("W")
assert W.shape == (10, 16)
```

| Argument    | Type           | Description                       |
| ----------- | -------------- | --------------------------------- |
| `name`      | <tt>str</tt>   | The name of the parameter to get. |
| **RETURNS** | <tt>Array</tt> | The current parameter.            |

### Model.set_param {#set_param tag="method"}

Set a weights parameter's value.

```python
### Example
from thinc.api import Linear, zero_init
import numpy

model = Linear(10, init_W=zero_init)
assert model.has_param("W") is None
model.set_param("W", numpy.zeros((10, 16), dtype="f"))
assert model.has_param("W") is True
```

| Argument | Type                     | Description                                   |
| -------- | ------------------------ | --------------------------------------------- |
| `name`   | <tt>str</tt>             | The name of the parameter to set a value for. |
| `value`  | <tt>Optional[Array]</tt> | The new value of the parameter.               |

### Model.has_attr {#has_attr tag="method"}

Check whether the model has the given attribute.

```python
### Example
from thinc.api import ExtractWindow

model = ExtractWindow()
model.has_attr("window_size") is True
model.has_attr("blah") is False
```

| Argument    | Type                    | Description                                   |
| ----------- | ----------------------- | --------------------------------------------- |
| `name`      | <tt>str</tt>            | The name of the attribute.                    |
| **RETURNS** | <tt>Optional[bool]</tt> | A ternary value (`True`, `False`, or `None`). |

### Model.get_attr {#get_attr tag="method"}

Get the attribute. Raises a `KeyError` if the attribute is not present.

```python
### Example
from thinc.api import ExtractWindow

model = ExtractWindow()
model.get_attr("window_size") == 3
```

| Argument    | Type         | Description                |
| ----------- | ------------ | -------------------------- |
| `name`      | <tt>str</tt> | The name of the attribute. |
| **RETURNS** | <tt>Any</tt> | The attribute's value.     |

### Model.set_attr {#set_attr tag="method"}

Set the attribute to the given value.

```python
### Example
from thinc.api import ExtractWindow

model = ExtractWindow()
model.set_attr("window_size", 5)
assert model.get_attr("window_size") == 5
```

| Argument | Type         | Description                      |
| -------- | ------------ | -------------------------------- |
| `name`   | <tt>str</tt> | Set an attribute to a new value. |
| `value`  | <tt>Any</tt> | The new value for the attribute. |

### Model.has_ref {#has_ref tag="method"}

Check whether the model has a reference of a given name. If the reference is
registered but the value is unset, returns `None`.

```python
### Example
TODO: write
```

| Argument    | Type                    | Description                                   |
| ----------- | ----------------------- | --------------------------------------------- |
| `name`      | <tt>str</tt>            | The name of the reference.                    |
| **RETURNS** | <tt>Optional[bool]</tt> | A ternary value (`True`, `False`, or `None`). |

### Model.get_ref {#get_ref tag="method"}

Retrieve the value of a reference of the given name, or None if unset. Raises a
`KeyError` if unset.

```python
### Example
TODO: write
```

| Argument    | Type           | Description                |
| ----------- | -------------- | -------------------------- |
| `name`      | <tt>str</tt>   | The name of the reference. |
| **RETURNS** | <tt>Model</tt> | The reference.             |

### Model.set_ref {#set_ref tag="method"}

Set a value for a reference.

```python
### Example
TODO: write
```

| Argument | Type                     | Description                      |
| -------- | ------------------------ | -------------------------------- |
| `name`   | <tt>str</tt>             | The name of the reference.       |
| `value`  | <tt>Optional[Model]</tt> | The new value for the attribute. |

### Model.has_grad {#has_grad tag="method"}

Check whether the model has a non-zero gradient for the given parameter. If the
gradient is allocated but is zeroed, returns `None`.

```python
### Example
TODO: write
```

| Argument    | Type                    | Description                                   |
| ----------- | ----------------------- | --------------------------------------------- |
| `name`      | <tt>str</tt>            | The parameter to check the gradient for.      |
| **RETURNS** | <tt>Optional[bool]</tt> | A ternary value (`True`, `False`, or `None`). |

### Model.get_grad {#get_grad tag="method"}

Get the gradient for a parameter, if one is available. If the parameter is
undefined or no gradient has been allocated, raises a `KeyError`.

```python
### Example
TODO: write
```

| Argument    | Type           | Description                                        |
| ----------- | -------------- | -------------------------------------------------- |
| `name`      | <tt>str</tt>   | The name of the parameter to get the gradient for. |
| **RETURNS** | <tt>Array</tt> | The current gradient of the parameter.             |

### Model.set_grad {#set_grad tag="method"}

Set a parameter gradient to a new value.

```python
### Example
TODO: write
```

| Argument | Type           | Description                                           |
| -------- | -------------- | ----------------------------------------------------- |
| `name`   | <tt>str</tt>   | The name of the parameter to assign the gradient for. |
| `value`  | <tt>Array</tt> | The new gradient.                                     |

### Model.inc_grad {#inc_grad tag="method"}

Add a value to the gradient of a parameter.

```python
### Example
TODO: write
```

| Argument | Type           | Description                       |
| -------- | -------------- | --------------------------------- |
| `name`   | <tt>str</tt>   | The name of the parameter.        |
| `value`  | <tt>Array</tt> | The value to add to its gradient. |

### Model.get_gradients {#get_gradients tag="method"}

Get non-zero gradients of the model's parameters, as a dictionary keyed by the
parameter ID. The values are `(weights, gradients)` tuples.

```python
### Example
TODO: write
```

| Argument    | Type                      | Description                          |
| ----------- | ------------------------- | ------------------------------------ |
| **RETURNS** | <tt>Dict[int, Array]</tt> | The gradients keyed by parameter ID. |

### Model.copy {copy# tag="method"}

Create a copy of the model, its attributes, and its parameters. Any child layers
will also be deep-copied. The copy will receive a distinct `model.id` value.

```python
### Example
from thinc.api import Linear

model = Linear()
model_copy = model.copy()
```

| Argument    | Type           | Description              |
| ----------- | -------------- | ------------------------ |
| **RETURNS** | <tt>Model</tt> | A new copy of the model. |

### Model.to_gpu {#to_gpu tag="method"}

Transfer the model to a given GPU device.

```python
### Example
device = model.to_gpu(0)
```

| Argument    | Type                      | Description             |
| ----------- | ------------------------- | ----------------------- |
| `gpu_id`    | <tt>int</tt>              | Device index to select. |
| **RETURNS** | <tt>cupy.cuda.Device</tt> | The device.             |

### Model.to_cpu {#to_cpu tag="method"}

Copy the model to CPU.

```python
### Example
model.to_cpu()
```

### Model.to_bytes {#to_bytes tag="method"}

Serialize the model to a bytes representation.

<infobox>

Models are usually serialized using `msgpack`, so you should be able to call
`msgpack.loads()` on the data and get back a dictionary with the contents.
Serialization should round-trip identically, i.e. the same bytes should result
from loading and serializing a model.

</infobox>

```python
### Example
bytes_data = model.to_bytes()
```

| Argument    | Type           | Description           |
| ----------- | -------------- | --------------------- |
| **RETURNS** | <tt>bytes</tt> | The serialized model. |

### Model.from_bytes {#from_bytes tag="method"}

Deserialize the model from a bytes representation.

<infobox>

Models are usually serialized using `msgpack`, so you should be able to call
`msgpack.loads()` on the data and get back a dictionary with the contents.
Serialization should round-trip identically, i.e. the same bytes should result
from loading and serializing a model.

</infobox>

```python
### Example
bytes_data = model.to_bytes()
model = Model("model_name", forward).from_bytes(bytes_data)
```

| Argument     | Type           | Description             |
| ------------ | -------------- | ----------------------- |
| `bytes_data` | <tt>bytes</tt> | The bytestring to load. |
| **RETURNS**  | <tt>Model</tt> | The loaded model.       |

### Model.to_disk {#to_disk tag="method"}

Serialize the model to disk. Most models will serialize to a single file, which
should just be the bytes contents of [`Model.to_bytes`](#to_bytes).

```python
### Example
model.to_disk("/path/to/model")
```

| Argument | Type                      | Description                     |
| -------- | ------------------------- | ------------------------------- |
|  `path`  | <tt>Union[Path, str]</tt> | Directory to save the model to. |

### Model.from_disk {#from_disk tag="method"}

Deserialize the model from disk. Most models will serialize to a single file,
which should just be the bytes contents of [`Model.to_bytes`](#to_bytes).

```python
### Example
model = Model().from_disk("/path/to/model")
```

| Argument    | Type                      | Description                       |
| ----------- | ------------------------- | --------------------------------- |
|  `path`     | <tt>Union[Path, str]</tt> | Directory to load the model from. |
| **RETURNS** | <tt>Model</tt>            | The loaded model.                 |

---

## Utilities {#util}

### create_init {#create_init tag="function"}

Create an init function, given a dictionary of parameter initializers.

```python
### Example
from thinc.api import create_init, zero_init

init = create_init({"W": zero_init, "b": zero_init})
```

| Argument       | Type                         | Description                      |
| -------------- | ---------------------------- | -------------------------------- |
| `initializers` | <tt>Dict[str, Callable]</tt> | The parameter initializers.      |
| **RETURNS**    | <tt>Callable</tt>            | The init function for the model. |

---

## Shim {#shim tag="class"}

Define a basic interface for external models. Users can create subclasses of
`Shim` to wrap external libraries. The Thinc `Model` class treats `Shim` objects
as a sort of special type of sublayer: it knows they're not actual Thinc `Model`
instances, but it also knows to talk to the shim instances when doing things
like using transferring between devices, loading in parameters, optimization. It
also knows `Shim` objects need to be serialized and deserialized with to/from
bytes/disk, rather than expecting that they'll be `msgpack`-serializable. A
`Shim` can implement the following methods:

| Method           | Description                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
|  `__init__`      | Initialize the model.                                                                                  |
|  `__call__`      | Call the model and return the output and a callback to compute the gradients via backpropagation.      |
|  `predict`       | Call the model and return only the output, instead of the `(output, callback)` tuple.                  |
|  `begin_update`  | Run the model over a batch of data, returning the output and a callback to complete the backward pass. |
|  `finish_update` | Update parameters with current gradients.                                                              |
|  `use_params`    | Context manager to temporarily set the model's parameters to specified values.                         |
|  `to_gpu`        | Transfer the model to a given GPU device.                                                              |
|  `to_cpu`        | Copy the model to CPU.                                                                                 |
|  `to_bytes`      | Serialize the model to bytes.                                                                          |
|  `from_bytes`    | Load the model from bytes.                                                                             |
|  `to_disk`       | Serialize the model to disk. Defaults to writing the bytes representation to a file.                   |
|  `from_disk`     | Load the model from disk. Defaults to loading the byte representation from a file.                     |

### PyTorchShim {#pytorchshim tag="class"}

Interface between a [PyTorch](https://pytorch.org) model and a Thinc `Model`.
For more details and examples, see the docs on
[integrating PyTorch models](/docs/usage-framework).

<infobox variant="warning">

This container is **not** a Thinc `Model` subclass itself, it's a subclass of
`Shim`.

</infobox>

```python
https://github.com/explosion/thinc/blob/develop/thinc/shims/pytorch.py
```
