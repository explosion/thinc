---
title: Concept and Design
teaser: Thinc's conceptual model and how it works
next: /docs/install
---

Thinc is built on a fairly simple conceptual model that's a little bit different
from other neural network libraries. On this page, we build up the library from
first principles, so you can see how everything fits together. This page assumes
some conceptual familiarity with [backpropagation](/docs/backprop101), but you
should be able to follow along even if you're hazy on some of the details.

## The model composition problem

The central problem for a neural network implementation is this: during the
**forward pass**, you compute results that will later be useful during the
**backward pass**. How do you keep track of this arbitrary state, while making
sure that layers can be cleanly composed?

Instead of starting with the problem directly, let's start with a simple and
obvious approach, so that we can run into the problem more naturally. The most
obvious idea is that we have some thing called a `model`, and this thing holds
some parameters ("weights") and has a method to predict from some inputs to some
outputs using the current weights. So far so good. But we also need a way to
update the weights. The most obvious API for this is to add an `update` method,
which will take a batch of inputs and a batch of correct labels, and compute the
weight update.

```python
class UncomposableModel:
    def __init__(self, W):
        self.W = W

    def predict(self, inputs):
        return inputs @ self.W.T

    def update(self, inputs, targets, learn_rate=0.001):
        guesses = self.predict(inputs)
        d_guesses = (guesses-targets) / targets.shape[0]  # gradient of loss w.r.t. output
        # The @ is newish Python syntax for matrix multiplication
        d_inputs = d_guesses @ self.W
        dW = d_guesses.T @ inputs  # gradient of parameters
        self.W -= learn_rate * dW  # update weights
        return d_inputs
```

This API design works in itself, but the `update()` method only works as the
outer-level API. You wouldn't be able to put another layer with the same API
after this one and backpropagate through both of them. Let's look at the steps
for backpropagating through two matrix multiplications:

```python
def backprop_two_layers(W1, W2, inputs, targets):
    hiddens = inputs @ W1.T
    guesses = hiddens @ W2.T
    d_guesses = (guesses-targets) / targets.shape[0]  # gradient of loss w.r.t. output
    dW2 = d_guesses @ hiddens.T
    d_hiddens = d_guesses @ W2
    dW1 = d_hiddens @ inputs.T
    d_inputs = d_hiddens @ W1
    return dW1, dW2, d_inputs
```

In order to update the first layer, we need to know the gradient with respect to
its output. We can't calculate that value until we've finished the full forward
pass, calculated the gradient of the loss, and then backpropagated through the
second layer. This is why the `UncomposableModel` is uncomposable: the `update`
method expects the input and the target to both be available. That only works
for the outermost API – the same API can't work for intermediate layers.

Although nobody thinks of it this way, reverse-model auto-differentiation (as
supported by PyTorch, Tensorflow, etc) can be seen as a solution to this API
problem. The solution is to base the API around the `predict` method, which
doesn't have the same composition problem: there's no problem with writing
`model3.predict(model2.predict(model1.predict(X)))`, or
`model3.predict(model2.predict(X) + model1.predict(X))`, etc. We can easily
build a larger model from smaller functions when we're programming the forward
computations, and so that's exactly the API that reverse-mode
autodifferentiation was invented to offer.

The key idea behind Thinc is that it's possible to just fix the API problem
directly, so that models can be composed cleanly both forwards and backwards.
This results in an interestingly different developer experience: the code is far
more explicit and there are very few details of the framework to consider.
There's potentially more flexibility, but potentially lost performance and
sometimes more opportunities to make mistakes.

We don't want to suggest that Thinc's approach is uniformly better than a
high-performance computational graph engine such as PyTorch or Tensorflow. It
isn't. The trick is to use them together: you can use PyTorch, Tensorflow or
some other library to do almost all of the actual computation, while doing
almost all of your programming with a much more transparent, flexible and
simpler system. Here's how it works.

## No (explicit) computational graph – just higher order functions

The API design problem we're facing here is actually pretty basic. We're trying
to compute two values, but before we can compute the second one, we need to pass
control back to the caller, so they can use the first value to give us an extra
input. The general solution to this type of problem is a **callback**, and in
fact a callback is exactly what we need here.

Specifically, we need to make sure our model functions return a result, and then
a callback that takes a gradient of outputs, and computes the corresponding
gradient of inputs.

```python
def forward(X: InT) -> Tuple[OutT, Callable[[OutT], InT]]:
    Y: OutT = _do_whatever_computation(X)

    def backward(dY: OutT) -> InT:
        dX: InputType = _do_whatever_backprop(dY, X)
        return dX

    return Y, backward
```

To make this less abstract, here are two [layers](/docs/api-layers) following
this signature. For now, we'll stick to layers that don't introduce any
trainable weights, to keep things simple.

```python
### reduce_sum layer
def reduce_sum(X: Floats3d) -> Tuple[Floats2d, Callable[[Floats2d], Floats3d]]:
    Y = X.sum(axis=1)
    X_shape = X.shape

    def backprop_reduce_sum(dY: Floats2d) -> Floats3d:
        dX = zeros(X_shape)
        dX += dY.reshape((dY.shape[0], 1, dY.shape[1]))
        return dX

    return Y, backprop_reduce_sum
```

```python
### Relu layer
def relu(inputs: Floats2d) -> Tuple[Floats2d, Callable[[Floats2d], Floats2d]]:
    mask = inputs >= 0
    def backprop_relu(d_outputs: Floats2d) -> Floats2d:
        return d_outputs * mask
    return inputs * mask, backprop_relu

```

Notice that the `reduce_sum` layer's output is a different shape from its input.
The forward pass runs from input to output, while the backward pass runs from
gradient-of-output to gradient-of-input. This means that we'll always have two
matching pairs: `(input_to_forward, output_of_backprop)` and
`(output_of_forward, input_of_backprop)`. These pairs must match in type. If our
functions obey this invariant, we'll be able to write
[combinator functions](/docs/api-layers#combinators) that can wire together
layers in standard ways.

The most basic way we'll want to combine layers is a feed-forward relationship.
We call this combinator `chain`, after the chain rule:

```python
### Chain combinator
def chain(layer1, layer2):
    def forward_chain(X):
        Y, get_dX = layer1(X)
        Z, get_dY = layer2(Y)

        def backprop_chain(dZ):
            dY = get_dY(dZ)
            dX = get_dX(dY)
            return dX

        return Z, backprop_chain

    return forward_chain
```

We can use the `chain` combinator to build a function that runs our `reduce_sum`
and `relu` layers in succession:

```python
chained = chain(reduce_sum, relu)
X = uniform((2, 10, 6)) # (batch_size, sequence_length, width)
dZ = uniform((2, 6))    # (batch_size, width)
Z, get_dX = chained(X)
dX = get_dX(dZ)
assert dX.shape == X.shape
```

Our `chain` combinator works easily because our layers return callbacks. The
callbacks ensure that there is no distinction in API between the outermost layer
and a layer that's part of a larger network. We can see this clearly by
imagining the alternative, where the function expects the gradient with respect
to the output along with its input:

```python
### Problem without callbacks {highlight="15-19"}
def reduce_sum_no_callback(X, dY):
    Y = X.sum(axis=1)
    X_shape = X.shape
    dX = zeros(X_shape)
    dX += dY.reshape((dY.shape[0], 1, dY.shape[1]))
    return Y, dX

def relu_no_callback(inputs, d_outputs):
    mask = inputs >= 0
    outputs = inputs * mask
    d_inputs = d_outputs * mask
    return outputs, d_inputs

def chain_no_callback(layer1, layer2):
    def chain_forward_no_callback(X, dZ):
        # How do we call layer1? We can't, because its signature expects dY
        # as part of its input – but we don't know dY yet! We can only
        # compute dY once we have Y. That's why layers must return callbacks.
        raise CannotBeImplementedError()
```

The `reduce_sum` and `relu` layers are easy to work with, because they don't
introduce any parameters. But networks that don't have any parameters aren't
very useful. So how should we handle them? We can't just say that parameters are
just another type of input variable, because that's not how we want to use the
network. We want the parameters of a layer to be an internal detail – **we don't
want to have to pass in the parameters on each input**.

Parameters need to be handled differently from input variables, because we want
to specify them at different times. We'd like to specify the parameters once
when we create the function, and then have them be an internal detail that
doesn't affect the function's signature. The most direct approach is to
introduce another layer of closures, and make the parameters and their gradients
arguments to the outer layer. The gradients can then be incremented during the
backward pass:

```python
def Linear(W, b, dW, db):
    def forward_linear(X):

        def backward_linear(dY):
            dW += dY.T @ X
            db += dY.sum(axis=0)
            return dY @ W

        return X @ W.T + b, backward_linear
    return forward_linear

n_batch = 128
n_in = 16
n_out = 32
W = uniform((n_out, n_in))
b = uniform((n_out,))
dW = zeros(W.shape)
db = zeros(b.shape)
X = uniform((n_batch, n_in))
Y_true = uniform((n_batch, n_out))

linear = Linear(W, b, dW, db)
Y_out, get_dX = linear(X)

# Now we could calculate a loss and backpropagate
dY = (Y_out - Y_true) / Y_true.shape[0]
dX = get_dX(dY)

# Now we could do an optimization step like
W -= 0.001 * dW
b -= 0.001 * db
dW.fill(0.0)
db.fill(0.0)
```

While the above approach would work, handling the parameters and their gradients
explicitly will quickly get unmanageable. To make things easier, we need to
introduce a `Model` class, so that we can **keep track of the parameters,
gradients, dimensions** and other attributes that each layer might require.

The most obvious thing to do at this point would be to introduce one class per
layer type, with the forward pass implemented as a method on the class. While
this approach would work reasonably well, we've preferred a slightly different
implementation, that relies on composition rather than inheritance. The
implementation of the [`Linear` layer](/docs/api-layers#linear) provides a good
example.

Instead of defining a subclass of `thinc.model.Model`, the layer provides a
function `Linear` that constructs a [`Model` instance](/docs/api-model), passing
in the function `forward` in `thinc.layers.linear`:

```python
def forward(model: Model, X: InputType, is_train: bool):
```

The function receives a `model` instance as its first argument, which provides
you access to the dimensions, parameters, gradients, attributes and layers. The
second argument is the input data, and the third argument is a boolean that lets
layers run differently during training and prediction – an important requirement
for layers like dropout and batch normalization.

As well as the `forward` function, the `Model` also lets you pass in a function
`init`, allowing us to support **shape inference**.

```python
### Linear {highlight="3-4"}
model = Model(
    "linear",
    forward,
    init=init,
    dims={"nO": nO, "nI": nI},
    params={"W": None, "b": None},
)
```

We want to be able to define complex networks concisely, passing in **only
genuine configuration** — we shouldn't have to pass in a lot of variables whose
values are dictated by the rest of the network. The more redundant the
configuration, the more ways the values we pass in can be invalid. In the
example above, there are many different ways for the inputs to `Linear` to be
invalid: the `W` and `dW` variables could be different shapes, the size of `b`
could fail to match the first dimension of `W`, the second dimension of `W`
could fail to match the second dimension of the input, etc. With inputs like
these, there's no way we can expect functions to validate their inputs reliably,
leading to unpredictable logic errors that make the calling code difficult to
debug.

In a network with two `Linear` layers, only one dimension is an actual
hyperparameter. The input size to the first layer and the output size of the
second layer are both **determined by the shape of the data**. The only choice
to make is the number of "hidden units", which will determine the output size of
the first layer and the input size of the second layer. So we want to be able to
write something like this:

```python
model = chain(Linear(nO=n_hidden), Linear())
```

... and have the missing dimensions **inferred later**, based on the input and
output data. In order to make this work, we need to specify initialization logic
for each layer we define. For example, here's the initialization logic for the
`Linear` and `chain` layers:

```python
### Initialization logic
from typing import Optional
from thinc.api import Model, glorot_uniform_init
from thinc.types import Floats2d
from thinc.util import get_width

def init(model: Model, X: Optional[Floats2d] = None, Y: Optional[Floats2d] = None) -> None:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    W = model.ops.alloc2f(model.get_dim("nO"), model.get_dim("nI"))
    b = model.ops.alloc1f(model.get_dim("nO"))
    glorot_uniform_init(model.ops, W.shape)
    model.set_param("W", W)
    model.set_param("b", b)
```
