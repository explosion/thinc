---
title: Defining and Using Models
next: /docs/usage-training
---

Thinc's model-definition API is based on functional programming. The library
provides a compact set of useful
[combinator functions](/docs/api-layers#combinators), which combine layers
together in different ways. It's also very easy to write your own combinators,
to implement custom logic. Thinc also provides
[shim classes](/docs/usage-frameworks) that let you wrap models from other
libraries, allowing you to use them within Thinc.

There are a few great advantages to Thinc's approach: there's **less syntax to
remember**, complex models can be defined **very concisely**, we can perform
**shape inference** to save you from passing in redundant values, and we're able
to perform sophisticated **network validation**, making it easier to raise
errors early if there's a problem with your network.

Thinc's approach does come with some disadvantages, however. If you write custom
combinators, you'll have to take care to **pass your gradients through**
correctly during the backward pass. Thinc also doesn't try to perform
sophisticated graph optimizations, so "native" Thinc models may be slower than
PyTorch or TensorFlow. That's where the [shim layers](/docs/usage-frameworks)
come in: you can use Thinc to make all the **data manipulation and
preprocessing** operations easy and readable, and then call into TensorFlow or
PyTorch for the expensive part of your model, such as your transformer layers or
BiLSTM.

## Basic usage {#basics}

Let's start with the "hello world" of neural network models: recognizing
handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
We've prepared a [separate package](https://github.com/explosion/ml-datasets)
with loaders for some common datasets, including MNIST. So we can set up the
data with the following function:

```python
### 1. Setting up MNIST
import ml_datasets
(train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
```

Now let's define a model with **two ReLu-activated hidden layers**, followed by
a **softmax-activated output layer**. We'll also add dropout after the two
hidden layers, to help the model generalize better:

```python
### 2. Defining the model
from thinc.api import chain, ReLu, Softmax

model = chain(
    ReLu(nO=128, dropout=0.2),
    ReLu(nO=128, dropout=0.2),
    Softmax()
)
model.initialize(X=train_X[:5], Y=train_Y[:5])
```

The [`chain` combinator](/docs/api-layers#chain) is like `Sequential` in PyTorch
or Keras: it combines a list of layers together with a feed-forward
relationship. After creating the model, we also call the
[`Model.initialize`](/docs/api-model#initialize) method, passing in a small
batch of input data `X` and a small batch of output data `Y`. This allows Thinc
to infer the missing dimensions: when we defined the model, we didn't tell it
the input size or the output size. Those values are generally not free
hyperparameters under your control: they're defined by the problem you're
working on. You can still provide these values explicitly, but we recommend
using shape inference where possible, as it means there will be fewer ways to
define your model incorrectly.

The [`Model.initialize`](/docs/api-model#initialize) method also allocates and
initializes the weights parameters of the [`ReLu`](/docs/api-layers#relu) and
[`Softmax`](/docs/api-layers#softmax) layers. Thinc is able to set reasonable
defaults for the weight randomization because it knows the activation functions,
but you can also specify the initialization using keyword arguments.

After creating and initializing the model, we need to train it. Here's how we'll
update the model on each batch of data:

```python
### 3. Training the model
def update_model(model, optimizer, inputs, truths):
    # Predict, and get callback for backprop
    guesses, backprop = model.begin_update(inputs)
    # Calculate gradient of loss with respect to predictions
    d_guesses = (guesses - truths) / guesses.shape[0]
    # Backpropagate, incrementing gradients of weights
    d_inputs = backprop(d_guesses)
    # Update weights and clear current gradients
    model.finish_update(optimizer)
    # Calculate loss statistic, for logging
    loss = ((guesses - truths)**2).sum()
    return loss
```

Next we need to create an [optimizer](/docs/api-optimizers), and make several
passes over the data, randomly selecting paired batches of the inputs and labels
each time. While some machine learning libraries provide a single `.fit()`
method to train a model all at once, Thinc puts you in charge of **shuffling and
batching your data**, with the help of a few handy utility methods.

```python
### 4. The training loop
from thinc.api import Adam, get_shuffled_batches, evaluate_model_on_arrays

optimizer = Adam(0.001)

for epoch in range(n_epochs):
    for X, Y in get_shuffled_batches(train_X, train_Y):
        loss += update_model(model, optimizer, X, Y)
    accuracy = evaluate_model_on_arrays(model, dev_X, dev_Y)
    print(epoch, loss, accuracy)
```

The [`get_shuffled_batches`](/docs/api-util#get_shuffled_batches) and
[`evaluate_model_on_arrays`](/docs/api-util#evaluate_model_on_arrays) helpers
won't always work on your exact dataset, so you'll often need to write your own
code to iterate over your data. However, this task will be completely
independent of Thinc: it's just a normal Python programming task, of producing
an iterator that works with whatever data format you are working with. Thinc's
philosophy is **keep the API surface small,** giving you fewer library-specific
details to remember and program against.

Of course, training a model isn't very useful without a way to save out the
weights and load them back in later. Thinc supports **three ways of saving and
loading your model**:

1. The most flexible is to use the [`Model.to_bytes`](/docs/api-model#to_bytes)
   method, which saves the model state to a byte string, serialized using the
   `msgpack` library. The result can then be loaded back using the
   [`Model.from_bytes`](/docs/api-model#from_bytes) method.

2. The [`Model.to_disk`](/docs/api-model#to_disk) method works similarly, except
   the result is saved to a path you provide instead. The result can be loaded
   back using the [`Model.from_disk`](/docs/api-model#from_disk) method.

3. Pickle the `Model` instance. This should work, but is not our recommendation
   for most use-cases. Pickle is inefficient in both time and space, does not
   work reliably across Python versions or platforms, and is not suitable for
   untrusted inputs, as unpickling an object allows arbitrary code execution by
   design.

The `from_bytes` and `from_disk` methods are intended to be relatively safe:
unlike formats such as Pickle, untrusted inputs are not intended to allow
arbitrary code execution. This means you have to create the `Model` object
yourself first, and then use that object to load in the state. To make this
easier, you'll usually want to put your model creation code inside a function,
and then **register it**, like this:

```python
### 5. Register the model
import thinc
from thinc.api import ReLu, Softmax, chain

@thinc.registry.layers("mnist_mlp.v1")
def create_mnist_mlp_model(nO: int, dropout: float, from_disk: Optional[str] = None):
    model = chain(
        ReLu(nO=128, dropout=0.2),
        ReLu(nO=128, dropout=0.2),
        Softmax()
    )
    if from_disk is not None:
        model.from_disk(from_disk)
    return model
```

The [registry](/docs/api-config#registry) allows you to look up the function by
name later, so you can pass along all the details to recreate your model in one
message. Check out our [guide on the config system](/docs/usage-config) for more
details.

### Full MNIST example {#mnist-example}

TODO: intro

```python
https://github.com/explosion/thinc/blob/develop/examples/scripts/mnist.py
```

---

## Inspecting and updating model state {#model-state}

As you build more complicated models, you'll often need to inspect your model in
various ways. This is especially important when you're writing your own layers.
Here's a quick summary of the different types of information you can attach and
query.

|                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`Model.id`](/docs/api-model#attributes)                                                                                                                                                       | A numeric identifier, to distinguish different model instances. During [`Model.__init__`](/docs/api-model#init), the `Model.global_id` class attribute is incremented and the next value is used for the `model.id` value.                                                                                                           |
| [`Model.name`](/docs/api-model#attributes)                                                                                                                                                     | A string name for the model.                                                                                                                                                                                                                                                                                                         |
| [`Model.layers`](/docs/api-model#properties) [`Model.walk`](/docs/api-model#walk)                                                                                                              | List the immediate sublayers of a model, or iterate over the model's whole subtree (including the model itself).                                                                                                                                                                                                                     |
| [`Model.shims`](/docs/api-model#properties)                                                                                                                                                    | Wrappers for external libraries, such as PyTorch and TensorFlow. [`Shim`](/docs/api-model#shim) objects hold a reference to the external object, and provide a consistent interface for Thinc to work with, while also letting Thinc treat them separately from `Model` instances for the purpose of serialization and optimization. |
| [`Model.has_dim`](/docs/api-model#has_dim) [`Model.get_dim`](/docs/api-model#get_dim) [`Model.set_dim`](/docs/api-model#set_dim) [`Model.dim_names`](/docs/api-model#properties)               | Check, get, set and list the layer's **dimensions**. A dimension is an integer value that affects a model's parameters or the shape of its input data.                                                                                                                                                                               |
| [`Model.has_param`](/docs/api-model#has_param) [`Model.get_param`](/docs/api-model#get_param) [`Model.set_param`](/docs/api-model#set_param) [`Model.param_names`](/docs/api-model#properties) | Check, get, set and list the layer's **weights parameters**. A parameter is an array that can have a gradient and can be optimized.                                                                                                                                                                                                  |
| [`Model.has_grad`](/docs/api-model#has_grad) [`Model.get_grad`](/docs/api-model#get_grad) [`Model.set_grad`](/docs/api-model#set_grad) [`Model.grad_names`](/docs/api-model#properties)        | Check, get, set, increment and list the layer's **weights gradients**. A gradient is an array of the same shape as a weights parameter, that increments values used to update the parameter during training.                                                                                                                         |
| [`Model.has_attr`](/docs/api-model#has_attr) [`Model.get_attr`](/docs/api-model#get_attr) [`Model.set_attr`](/docs/api-model#set_attr) [`Model.attr_names`](/docs/api-model#properties)        | Check, get, set and list the layer's **attributes**. Attributes are other information the layer needs, such as configuration or settings. You should ensure that attribute values you set are either JSON-serializable, or support a `to_bytes` method, or the attribute will prevent model serialization.                           |
| [`Model.has_ref`](/docs/api-model#has_ref) [`Model.get_ref`](/docs/api-model#get_ref) [`Model.set_ref`](/docs/api-model#set_ref) [`Model.ref_names`](/docs/api-model#properties)               | Check, get, set and list the layer's **node references**. A node reference lets you easily refer to particular nodes within your model's subtree. For instance, if you want to expose the embedding table from your model, you can add a reference to it.                                                                            |

### Naming conventions

Thinc names dimensions and parameters with strings, so you can use arbitrary
names on your models. For the built-in
[layers and combinators](/docs/api-layers), we use the following conventions:

|      |                                                                            |
| ---- | -------------------------------------------------------------------------- |
| `nO` | The width of output arrays from the layer.                                 |
| `nI` | The width of input arrays from the layer.                                  |
| `nP` | Number of "pieces". Used in the [`Maxout`](/docs/api-layers#maxput) layer. |
| `W`  | A 2-dimensional weights parameter, for connection weights.                 |
| `b`  | A 1-dimensional weights parameter, for biases.                             |
| `E`  | A 2-dimensional weights parameter, for an embedding table.                 |
