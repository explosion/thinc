---
title: PyTorch, TensorFlow & MXNet
teaser: Interoperability with machine learning frameworks
next: /docs/usage-sequences
---

Wrapping models from other frameworks is a core use case for Thinc: we want to
make it easy for people to write [spaCy](https://spacy.io) components using
their preferred machine learning solution. We expect a lot of code-bases will
have similar requirements. As well as **wrapping whole models**, Thinc lets you
call into an external framework for just **part of your model**: you can have a
model where you use PyTorch just for the transformer layers, using "native"
Thinc layers to do fiddly input and output transformations and add on
task-specific "heads", as efficiency is less of a consideration for those parts
of the network.

## How it works {#how-it-works}

Thinc uses a special class, [`Shim`](/docs/api-model#shim), to hold references
to external objects. This allows each wrapper space to define a custom type,
with whatever attributes and methods are helpful, to assist in managing the
communication between Thinc and the external library. The
[`Model`](/docs/api-model#model) class holds `shim` instances in a separate
list, and communicates with the shims about updates, serialization, changes of
device, etc.

![](images/wrapper_pytorch.svg)

![](images/wrapper_tensorflow.svg)

The wrapper will receive each batch of inputs, convert them into a suitable form
for the underlying model instance, and pass them over to the shim, which will
**manage the actual communication** with the model. The output is then passed
back into the wrapper, and converted for use in the rest of the network. The
equivalent procedure happens during backpropagation. Array conversion is handled
via the [DLPack](https://github.com/dmlc/dlpack) standard wherever possible, so
that data can be passed between the frameworks **without copying the data back**
to the host device unnecessarily.

| Framework      | Wrapper layer                                                                                                                                        | Shim                                                                                                                         | DLPack                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **PyTorch**    | [`PyTorchWrapper`](/docs/api-layers#pytorchwrapper) ([code](https://github.com/explosion/thinc/blob/master/thinc/layers/pytorchwrapper.py))          | [`PyTorchShim`](/docs/api-model#shims) ([code](https://github.com/explosion/thinc/blob/master/thinc/shims/pytorch.py))       | <i name="yes"></i>                                                                      |
| **TensorFlow** | [`TensorFlowWrapper`](/docs/api-layers#tensorflowwrapper) ([code](https://github.com/explosion/thinc/blob/master/thinc/layers/tensorflowwrapper.py)) | [`TensorFlowShim`](/docs/api-model#shims) ([code](https://github.com/explosion/thinc/blob/master/thinc/shims/tensorflow.py)) | <i name="no"></i> [<sup>1</sup>](https://github.com/tensorflow/tensorflow/issues/24453) |
| **MXNet**      | [`MXNetWrapper`](/docs/api-layers#mxnetwrapper) ([code](https://github.com/explosion/thinc/blob/master/thinc/layers/mxnetwrapper.py))                | [`MXNetShim`](/docs/api-model#shims) ([code](https://github.com/explosion/thinc/blob/master/thinc/shims/mxnet.py))           | <i name="yes"></i>                                                                      |

To see wrapped models in action, check out the following examples:

<!-- TODO: more examples -->

<tutorials header="false">

- intro
- transformers_tagger

</tutorials>

---

## Integrating models {#integrating-models}

The [`PyTorchWrapper`](/docs/api-layers#pytorchwrapper) and
[`TensorFlowWrapper`](/docs/api-layers#tensorflowwrapper) layers allow you to
easily use your predefined models in Thinc, as part or all of your network. For
simple models that accept one array as input and return one array as output, all
you need to do is create the PyTorch/TensorFlow layer and pass it into the
wrapper. The wrapper model will behave like any other Thinc layer.

```python
### PyTorch Example {highlight="5"}
from thinc.api import PyTorchWrapper, chain, Linear
import torch.nn

model = chain(
    PyTorchWrapper(torch.nn.Linear(16, 8)), # 🚨 PyTorch goes (nI, nO)!
    Linear(4, 8)
)
X = model.ops.alloc2f(1, 16)  # make a dummy batch
model.initialize(X=X)
Y, backprop = model(X, is_train=True)
dX = backprop(Y)
```

```python
### TensorFlow Example {highlight="6"}
from thinc.api import TensorFlowWrapper, chain, Linear
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = chain(
    TensorFlowWrapper(Sequential([Dense(8, input_shape=(16,))])),
    Linear(4, 8)
)
X = model.ops.alloc2f(1, 16)  # make a dummy batch
model.initialize(X=X)
Y, backprop = model(X, is_train=True)
dX = backprop(Y)
```

**In theory**, you can also chain together layers and models written in PyTorch
_and_ TensorFlow. However, this is likely a bad idea for actual production use,
especially since TensorFlow tends to hog the GPU. It could come in handy during
development, though, for instance if you need to port your models from one
framework to another.

```python
### Frankenmodel {highlight="7-8"}
from thinc.api import PyTorchWrapper, TensorFlowWrapper, chain, Linear
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch.nn

model = chain(  # 🚨 probably don't do this in production
    PyTorchWrapper(torch.nn.Linear(16, 8)),
    TensorFlowWrapper(Sequential([Dense(4, input_shape=(8,))])),
    Linear(2, 4)
)
model.initialize(X=model.ops.alloc2f(1, 16))
```

For more complex cases, you can control the way data is passed into the wrapper
using the `convert_inputs` and `convert_outputs` callbacks. Both callbacks have
input signatures like normal `forward` functions and return a tuple with their
output and a callback to handle the backward pass. However, the converters will
send and receive different data during the backward pass.

| ‎          | Forward                                       | Backward                                      |
| ---------- | --------------------------------------------- | --------------------------------------------- |
| **Input**  | Thinc <i name="right" alt="to"></i> Framework | Framework <i name="right" alt="to"></i> Thinc |
| **Output** | Framework <i name="right" alt="to"></i> Thinc | Thinc <i name="right" alt="to"></i> Framework |

To convert arrays, you can use the `xp2` and `2xp` utility functions which
translate to and from `numpy` and `cupy` arrays.

| Framework      | To numpy / cupy                                 | From numpy / cupy                               |
| -------------- | ----------------------------------------------- | ----------------------------------------------- |
| **PyTorch**    | [`xp2torch`](/docs/api-util#xp2torch)           | [`torch2xp`](/docs/api-util#torch2xp)           |
| **TensorFlow** | [`xp2tensorflow`](/docs/api-util#xp2tensorflow) | [`tensorflow2xp`](/docs/api-util#tensorflow2xp) |
| **MXNet**      | [`xp2mxnet`](/docs/api-util#xp2mxnet)           | [`mxnet2xp`](/docs/api-util#mxnet2xp)           |

### convert_inputs {#convert_inputs tag="function"}

The [`ArgsKwargs`](/docs/api-types#argskwargs) object is a little dataclass that
represents the tuple `(args, kwargs)`. Whatever you put in the `ArgsKwargs` you
return from your `convert_inputs` function will be passed directly into
PyTorch/TensorFlow/etc. In the backward pass, the shim will return a callback
with the gradients from PyTorch, in matching positions on another `ArgsKwargs`
object, and you'll then return an object that matches the original input, to
pass the gradient back down the Thinc model.

| Argument    | Type                                                   | Description                                                                                                                     |
| ----------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `model`     | <tt>Model</tt>                                         | The wrapper layer.                                                                                                              |
| `inputs`    | <tt>Any</tt>                                           | The input to the layer.                                                                                                         |
| `is_train`  | <tt>bool</tt>                                          | A flag indicating training context.                                                                                             |
| **RETURNS** | <tt>Tuple[ArgsKwargs, Callable[[ArgsKwargs], Any]</tt> | Inputs to the `PyTorchShim`, and a callback that receives the input gradients from PyTorch and returns the converted gradients. |

### convert_outputs {#convert_outputs tag="function"}

For the output, the converter will receive a tuple that contains the original
input (i.e. whatever was passed into `convert_inputs`, and the output from the
PyTorch/TensorFlow/etc. layer. The input is provided because the output may not
contain all the information you need to manage the conversion. The
`convert_output` function should return the output object for the rest of the
network, converting via the `2xp` helpers as necessary, along with the
un-convert callback.

The un-convert callback will receive a gradient object from the Thinc layer
above it in the network, and will return an `ArgsKwargs` object that will be
passed into
[`torch.autograd.backward`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward)
and the TensorFlow model and
[`tensorflow.GradientTape.watch`](https://www.tensorflow.org/api_docs/python/tf/GradientTape#watch)
respectively.

| Argument        | Type                                             | Description                                                                                                                                                                    |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| convert_outputs | <tt>Model</tt>                                   | The wrapper layer.                                                                                                                                                             |
| `outputs`       | <tt>Tuple[Any, Any]                              | A tuple of the original inputs and the PyTorch model's outputs.                                                                                                                |
| **RETURNS**     | <tt>Tuple[Any, Callable[[Any], ArgsKwargs]]</tt> | A tuple of the PyTorch outputs, and a callback to un-convert the gradient for PyTorch that takes the output gradients from Thinc and returns the output gradients for PyTorch. |

<infobox variant="warning">

The converter functions are designed to allow the `PyTorchWrapper` and
`TensorFlowWrapper` to **handle the vast majority of cases** you might face, but
if you do face a situation they don't cover, you can always just write a
distinct wrapper layer to handle your custom logic.

</infobox>

### More specific PyTorch layers {#pytorch-layers}

Thinc also includes some more specific PyTorch layers for common use-cases. The
[`PyTorchLSTM`](/docs/api-layers#lstm) layer creates and wraps the
[`torch.nn.LSTM`](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM) class,
making creation particularly easy. The
[`PyTorchRNNWrapper`](/docs/api-layers#pytorchwrapper) provides a little more
flexibility, allowing you to pass in a custom sequence model that has the same
inputs and output behavior as a
[`torch.nn.RNN`](https://pytorch.org/docs/stable/nn.html#torch.nn.RNN) object.

### Avoiding memory contention (experimental) {#memory-contention}

If you use the `PyTorchWrapper` for part of your network while using Thinc's
layers for other parts, you may find yourself running out of GPU memory
unexpectedly. This can occur because both PyTorch and `cupy` reserve their own
**internal memory pools**, and the two libraries do not communicate with each
other. When PyTorch needs more memory, it can only ask the device – so you may
get an out-of-memory error even though `cupy`'s pool has plenty of spare memory
available.

The best solution to this problem is to **reroute the memory requests** so that
only one library is in charge. Specifically, `cupy` offers a
[`cupy.cuda.set_allocator`](https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.cuda.set_allocator.html)
function, which should allow a custom allocator to be created that requests its
memory via PyTorch. Thinc provides a handy shortcut for this via the
[`use_pytorch_for_gpu_memory`](/docs/api-util#use_pytorch_for_gpu_memory) helper
function. We're hoping to add a helper for TensorFlow in the future once
[DLPack is supported in TensorFlow](https://github.com/tensorflow/tensorflow/issues/24453).

<tutorials header="false">

- gpu_memory

</tutorials>
