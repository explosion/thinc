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

<tutorials>

- intro
- intro_model
- basic_cnn_tagger
- transformers_tagger

</tutorials>

---

## Composing models {#composing}

Thinc follows a **functional-programming approach** to model definition. Its
approach is especially effective for complicated network architectures, and
use-cases where different data-types need to be passed through the network to
reach specific subcomponents. However, individual Thinc components are often
less performant than implementations from other libraries, so we suggest the use
of wrapper objects around performance-sensitive components such as LSTM or
transformer encoders. You then use Thinc to wire these building blocks together
in complicated ways, taking advantage of the type checking, configuration system
and concise syntax to make your code easier to write, read and maintain.

For instance, let's say you want a simple three-layer network, where you apply
two fully-connected layers, each with a non-linear activation function, and then
an output layer with a softmax activation. Thinc provides creation functions
that pair weights + activation with optional dropout and layer normalization,
and the feed-forward relationship is expressed using the
[`chain`](/docs/api-layers#chain) combinator. So the three layer network would
be:

```python
### Simple model
model = chain(
    Relu(nO=hidden_width, dropout=0.2),
    Relu(nO=hidden_width, dropout=0.2),
    Softmax(nO=n_classes)
)
```

The [`chain`](/docs/api-layers#chain) function is similar to the `Sequential`
class in PyTorch and Keras: it expresses a **feed-forward relationship** between
the layers you pass in. We refer to wiring functions like `chain` as
[**combinators**](/docs/api-layers#combinators). While most libraries provide a
few combinators, Thinc takes this general approach a bit further. Instead of
expressing operations over data, you'll often create your network by expressing
operations over functions. This is what we mean when we say Thinc favors a
"functional" as opposed to "imperative" approach.

<grid>

```python
### Imperative {small="true"}
def multiply_reshape_sum(linear, X, pieces=4):
    Y = linear.forward(X)
    Y = Y.reshape((Y.shape[0], -1, pieces))
    Y = Y.sum(axis=-1)
    return Y
```

```python
### Functional {small="true"}
multiply_reshape_sum = chain(
    linear,
    reshape(lambda Y, pieces: (X.shape[0], -1, pieces), {"pieces": 4}),
    reduce_sum(axis=-1)
)
```

</grid>

In the **imperative** code above, you write a function that takes a batch of
data, do some things to it, and return the result. The **functional** code is
one step more abstract: you define the function using a relationship (`chain`)
and pass in functions to do each step. This approach is confusing at first, but
we promise it pays off once the network gets more complicated, especially when
combined with the option to define your own infix notation through operator
overloading.

```python
### Tagger with multi-feature embedding and CNN encoding
from thinc.api import HashEmbed, Maxout, Softmax, expand_window
from thinc.api import residual, with_array, clone, chain, concatenate

width = 128
depth = 4
n_tags = 17

def MultiEmbed(width):
    return concatenate(
        HashEmbed(width, 4000, column=0),
        HashEmbed(width // 2, 2000, column=1),
        HashEmbed(width // 2, 2000, column=2),
        HashEmbed(width // 2, 2000, column=3),
    )

def Hidden(nO, dropout=0.2):
    return Maxout(nO, pieces=3, normalize=True, dropout=dropout)

def CNN(width):
    return residual(chain(expand_window(1), Hidden(width)))

model = with_array(
    chain(
        MultiEmbed(width),
        Hidden(width),
        clone(CNN(width), depth),
        Softmax(n_tags)
    )
)
```

The example above shows the definition of a tagger model with a **multi-feature
CNN token-to-vector encoder**, similar to the one we used in
[spaCy](https://spacy.io) v2.x. Multiple numeric ID features are extracted for
each word, and each feature is separately embedded. The separate vectors are
concatenated and passed through a hidden layer, and then several convolutional
layers are applied for contextual encoding. Each CNN layer performs a
"sequence-to-column" transformation, where a window of surrounding words is
concatenated to each vector. A hidden layer then maps the result back to the
original dimensionality. Residual connections and layer normalization are used
to assist convergence.

### Overloading operators {#operators}

The [`Model.define_operators`](/docs/api-model#define_operators) classmethod
allows you to bind arbitrary binary functions to Python operators, for use in
any `Model` instance. The method can (and should) be used as a contextmanager,
so that the overloading is limited to the immediate block. This allows concise
and expressive model definitions, using custom infix notations.

```python
from thinc.api import Model, chain, Relu, Softmax

with Model.define_operators({">>": chain}):
    model = Relu(512) >> Relu(512) >> Softmax()
```

`Model.define_operators` takes a dict of operators mapped to functions,
typically [combinators](/docs/api-layers#combinators). Each function should
expect two arguments, one of which is a [`Model`](/docs/api-model) instance. The
second argument can be any type, but will usually be another model. Within the
block you can now use the defined operators to compose layers – for instance,
`a >> b` is equivalent to `chain(a, b)`. The overloading is cleaned up again at
the end of the block. The following operators are supported: `+`, `-`, `*`, `@`,
`/`, `//`, `%`, `**`, `<<`, `>>`, `&`, `^` and `|`.

If your models are very complicated, operator overloading can make your code
**more concise and readable**, while also making it easier to change things and
experiment with different architectures. Here's the same CNN-based tagger,
written with operator overloading.

```python
### with operator overloading
from thinc.api import Model, HashEmbed, Maxout, Softmax, expand_window
from thinc.api import residual, with_array, clone, chain, concatenate

width = 128
depth = 4
n_tags = 17

def Hidden(nO, dropout=0.2):
    return Maxout(nO, pieces=3, normalize=True, dropout=dropout)

with Model.define_operators({">>": chain, "**": clone, "|": concatenate}):
    model = with_array(
        (
            HashEmbed(width, 4000, column=0)
            | HashEmbed(width // 2, 2000, column=1)
            | HashEmbed(width // 2, 2000, column=2)
            | HashEmbed(width // 2, 2000, column=3)
        )
        >> Hidden(width)
        >> residual(expand_window(1) >> Hidden(width)) ** depth
        >> Softmax(n_tags)
    )
```

You won't always want to use operator overloading, but sometimes it's the best
way to show how information flows through the network. It can also help you
toggle debugging, logging or other configuration over individual components. For
instance, you might set up your operators so that you can write
`LSTM(width, width) @ 4` to set logging level 4 over just that component. Note
that the binding is defined at the beginning of each block, so you're free to
bind operators to your own functions, allowing you to define something like a
local domain-specific language.

### Initialization and data validation {#validation}

After defining your model, you can call
[`Model.initialize`](/docs/api-model#initialize) to initialize the weights,
calculate unset dimensions, set attributes or perform any other setup that's
required. Combinators are in charge of initializing their child layers.
`Model.initialize` takes an optional **sample of input and output data** that's
used to **infer missing shapes** and **validate your network**. If possible, you
should always provide at least _some_ data to ensure all dimensions are set and
to spot potential problems early on.

If a layer receives an unexpected data type, Thinc will raise a
[`DataValidationError`](/docs/api-util#errors) – like in this case where a
[`Linear`](/docs/api-layers#linear) layer that expects a 2d array is initialized
with a 3d array:

<grid>

```python
### Invalid data {small="true"}
from thinc.api import Linear
import numpy

X = numpy.zeros((1, 1, 1), dtype="f")
model = Linear(1, 2)
model.initialize(X=X)
```

```
### Error {small="true"}
Data validation error in 'linear'
X: &lt;class 'numpy.ndarray'&gt;
Y: &lt;class 'NoneType'&gt;

X   wrong array dimensions (expected 2, got 3)
```

</grid>

During initialization, the inputs and outputs that pass through the model are
checked against the **signature** of the layer's `forward` function. If a
layer's forward pass annotates the input as `X: Floats2d` but receives a 3d
array of floats, an error is raised. Similarly, if the forward pass annotates
its return value as `-> Tuple[List[FloatsXd], Callable]` but the model is
initialized with an array as the output data sample, you'll also see an error.

Because each layer is only responsible for itself (and its direct children),
data validation also works out-of-the-box for complex and nested networks.
That's also where it's most powerful, since it lets you detect problems as the
data is transformed. In this example, the [`Relu`](/docs/api-layers#relu) layer
outputs a 2d array, but the
[`ParametricAttention`](/docs/api-layers#parametricattention) layer expects a
[ragged array](/docs/api-types#ragged) of data and lengths.

<grid>

```python
### Invalid network {small="true"}
X = [numpy.zeros((4, 75), dtype="f")]
Y = numpy.zeros((1,), dtype="f")
model = chain(
    list2ragged(),
    reduce_sum(),
    Relu(12, dropout=0.5),  # -> Floats2d
    ParametricAttention(12)
)
model.initialize(X=X, Y=Y)
```

```
### Error {small="true"}
Data validation error in 'para-attn'
X: &lt;class 'numpy.ndarray'&gt;
Y: &lt;class 'NoneType'&gt;

X   instance of Ragged expected
```

</grid>

Note that if a layer accepts multiple types, the data will be validated against
each type and if it doesn't match any of them, you'll see an error describing
**all mismatches**. For instance, `with_array` accepts a ragged array, a padded
array, a 2d array or a list of 2d arrays. If you pass in a 3d array, which is
invalid, the error will look like this:

<grid>

```python
### Invalid data {small="true"}
from thinc.api import with_array, Linear
import numpy

X = numpy.zeros((1, 1, 1), dtype="f")
model = with_array(Linear())
model.initialize(X=X)
```

```
### Error {small="true"}
Data validation error in 'with_array-linear'
X: &lt;class 'numpy.ndarray'&gt;
Y: &lt;class 'numpy.ndarray'&gt;

X   instance of Padded expected
X   instance of Ragged expected
X   value is not a valid list
X   wrong array dimensions (expected 2, got 3)
```

</grid>

To take advantage of runtime validation,
[config validation](/docs/usage-config#registry) and
[static type checking](/docs/usage-type-checking), you should **add type hints**
to any custom layers, wrappers and functions you define. Type hints are
optional, though, and if no types are provided, the data won't be validated and
any inputs will be accepted.

</infobox>

---

## Defining new layers {#new-layers}

Thinc favors a **composition rather than inheritance** approach to creating
custom sublayers: the base [`Model`](/docs/api-model) class should be all you
need. You can define new layers by simply passing in different data, especially
the **forward** function, which is where you'll implement the layer's actual
logic. You'll usually want to make a function to put the pieces together. We
refer to such functions as **constructors**. The constructor is responsible for
defining the layer. Parameter allocation and initialization takes place in an
optional `init` function, which is called by `model.initialize`.

```python
### Layer definition
model = Model(
    "layer-name",                   # string name of layer
    forward,                        # forward function
    init=init,                      # optional initialize function
    dims={"nO": 128, "nI": None},   # optional dimensions
    params={"W": None, "b": None},  # optional parameters
    attrs={"my_attr": True},        # optional non-parameter attributes
    refs={},                        # optional references to other layers
    layers=[],                      # child layers
    shims=[]                        # child shims
)
```

### Constructor functions {#constructor-functions}

Thinc layers are almost always instances of [`Model`](/docs/api-model). You
usually don't need to create a subclass, although you can if you prefer. Because
layers usually reuse the same `Model` class, the constructor takes on some
responsibilities for defining the layer, even if all the data isn't available.
The `refs`, `params` and `dim` dictionaries are all mappings from string names
to optional values (where "optional" means you can make the value `None`). You
should use `None` to indicate the full set of names that should be present once
the layer is fully initialized. However, you cannot pass `None` values for
missing child layers or shims: these lists do not support `None` values.

Dimension, attribute and parameter names are identified using strings. Thinc's
built-in layers use the [convention](#naming-conventions) that `"nI"` refers to
the model's input width, and `"nO"` refers to the model's output width. You
should usually try to provide these, unless they are undefined for your model
(for instance, if your model accepts an arbitrary unsized object like a database
connector, it doesn't make sense to provide an `"nI"` dimension.) Your
constructor should define all dimensions and parameters you want to attach to
the model, mapping them to `None` if the values aren't available yet. You'll
usually map parameters to `None`, and only allocate them in your `init`
function.

```python
from thinc.api import Model

def random_chain(child_layer1: Model, child_layer2: Model, prob: float = 0.2) -> Model:
    """Randomly invert the order of two layers during training."""
    return Model(
        "random_order",
        random_chain_forward,
        init=init,
        attrs={"prob": prob},
        layers=[child_layer1, child_layer2],
    )
```

Many model instances will have one or more **child layers**. For composite
models that have several distinct parts, you should usually write your creation
functions to receive instances of the child layers, rather than settings to
construct it. This will make your layer more modular, and let you take better
advantage of Thinc's config system. You can add or remove child layers after
creation via the [`Model.layers`](/docs/api-model#properties) list property, but
you should usually prefer to set up the child layers on creation if possible.

In complicated networks, sometimes you need to refer back to specific parts of
the model. For instance, you might want to access the embedding layer of a
network directly. The `refs` dict lets you create named references to nodes. You
can have nodes referring to their siblings or parents, so long as you don't try
to serialize only that component: when you call
[`Model.to_bytes`](/docs/api-model#to_bytes) or
[`Model.to_disk`](/docs/api-model#to_disk), all of the reference targets must be
somewhere within the model's tree. Under the hood, references are implemented
using Python's weakref feature, to avoid circular dependencies.

For instance, let's say you needed each child layer in the `random_order`
example above to refer to each other. You could do this by setting node
references for them:

```python
child_layer1.set_ref("sibling", child_layer2)
child_layer2.set_ref("sibling", child_layer1)
```

If you call `model.to_bytes`, both references will be within the tree, so there
will be no problem. But you would not be able to call `child_layer1.to_bytes` or
`child_layer2.to_bytes`, as the link targets aren't reachable from
[`Model.walk`](/docs/api-model#walk).

Thinc's built-in layers follow a naming convention where combinators and
stateless transformations are created from `snake_case` functions, while weights
layers or higher level components are created from `CamelCased` names. This
naming reflects the general usage purpose of the layer, rather than the details
of exactly what top-level container is returned. Constructing models via
functions allows your code to do some things that would be difficult or
impossible with an API that exposes `__init__` methods directly, because it's
difficult to return a different object instance from a class constructor. For
instance, Thinc's `Relu` layer accepts the options `dropout` and `normalize`.
These operations are implemented as separate layers, so the constructor uses the
`chain` combinator to put everything together. You should feel free to take this
type of approach in your own constructors too: you can design the components of
your network to be smaller reusable pieces, while making the user-facing API
refer to larger units.

### The forward function and backprop callback {#weights-layers-forward}

Writing the forward function is the main part of writing a new layer --- it's
where the computation actually takes place. The forward function is passed into
[`Model.__init__`](/docs/api-model#init), and then kept as a reference within
the model instance. You won't normally call your forward function directly. It
will usually be invoked indirectly, via the [`__call__`](/docs/api-model#call),
[`predict`](/docs/api-model#predict) and
[`begin_update`](/docs/api-model#begin_update) methods. The implementation of
the `Model` class is pretty straightforward, so you can have a look at the code
to see how it all fits together.

Because the `forward` function is invoked within the `Model` instance, it needs
to stick to a strict signature. Your forward function needs to accept **exactly
three arguments**: the model instance, the input data and a flag indicating
whether the model is being invoked for training, or for prediction. It needs to
return a tuple with the output, and the backprop callback.

```python
### Forward function for a Linear layer
def linear_forward(model: Model, X, is_train):
    W = model.get_param("W")
    b = model.get_param("b")
    Y = X @ W.T + b

    def backprop(dY):
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", dY.T @ X)
        return dY @ W

    return Y, backprop
```

| Argument    | Type                          | Description                                         |
| ----------- | ----------------------------- | --------------------------------------------------- |
|  `model`    | <tt>Model</tt>                | The model instance.                                 |
| `X`         | <tt>Any</tt>                  | The inputs.                                         |
| `is_train`  | <tt>bool</tt>                 | Whether the model is running in a training context. |
| **RETURNS** | <tt>Tuple[Any, Callable]</tt> | The output and the backprop callback.               |

The model won't invoke your forward function with any extra positional or
keyword arguments, so you'll normally attach what you need to the model, as
`params`, `dims`, `attrs`, `layers`, `shims` or `refs`. At the beginning of your
function, you'll fetch what you need from the model and carry out your
computation. For example, the `random_chain_forward` function retrieves its
child layers and the `prob` attribute, uses them to compute the output, and then
returns it along with the `backprop` callback. The `backprop` callback uses some
results from the `forward`'s scope (specifically the two child callbacks and the
`prob` attribute), and returns the gradient of the input.

```python
def random_chain_forward(model: Model, X, is_train: bool):
    child_layer1 = model.layers[0]
    child_layer2 = model.layers[1]
    prob = model.get_attr("prob")
    is_reversed = is_train and prob >= random.random()
    if is_reversed:
        Y, get_dX = child_layer2(X, is_train)
        Z, get_dY = child_layer1(Y, is_train)
    else:
        Y, get_dX = child_layer1(X, is_train)
        Z, get_dY = child_layer2(Y, is_train)

    def backprop(dZ):
        dY = get_dY(dZ)
        dX = get_dX(dY)
        return dX

    return Z, backprop
```

Instead of defining the `forward` function separately, it's sometimes more
elegant to write it as a closure within the constructor. This is especially
helpful for quick utilities and transforms. If you don't otherwise need a
setting to be accessible from a model or serialized with it, you can simply
reference it from the outer scope, rather than passing it in as an attribute.
You should avoid referencing child layers in this way, however, as you do need
to pass the child layers into the `layers` list – otherwise they will be not
part of the model's tree.

```python
def random_chain(child_layer1, child_layer2, prob=0.2):
    ...
    def random_chain_forward(model: Model, X, is_train: bool):
        # You can define the `forward` function as a closure. If so, it's fine
        # to use attributes from the outer scope, but child layers should be
        # retrieved from the model.
        child_layer1, child_layer2 = model.layers
        is_reversed = is_train and prob >= random.random()
        ...
    ...
```

Another way to pass data from the constructor into the forward function is
partial function application. This is the best approach for static data that
will be reliably available at creation and does not need to be serialized.
Partial application is also the best way to establish temporary buffers for your
forward function to reuse. Reusable buffers can be helpful for performance
tuning to prevent repeat additional memory allocation.

There are a few constraints that well-behaved forward functions should follow in
order to make them interoperate better with other layers, and to help your code
work smoothly in distributed settings. Many of these considerations are less
relevant for quick hacks or experimentation, but even if everything stays
between you and your editor, you're likely to find your code easier to work with
and reason about if you keep these rules in mind, as they all amount to "avoid
unnecessary side-effects".

- **Params can be read-only.** The `model.get_params` method is allowed to
  return read-only arrays, or copies of the internal data. You should not assume
  that in-place changes to the params will have any effect.

- **Inputs can be read-only.** You should avoid writing to input variables, but
  if it's _really_ necessary for efficiency, you should at least check the
  `array.flags["WRITEABLE"]` attribute, and make a copy if necessary. Equally,
  if it's crucial for your layer that variables are not written to, use
  `array.setflags(write=False)` to prevent any shenanigans.

- **Writeable variables might get written to.** You'll often want to reference a
  variable returned by your forward function inside your backprop callback, as
  otherwise you'd have to recompute it. However, after your function has
  returned, the caller might write to the array, changing it out from under you.
  If the data is small, the best solution is to make a copy of variables you
  want to retain. If you're really worried about efficiency, you can set the
  array to read-only using `array.set_flags(write=False)`.

- **Avoid calls to [`Model.set_param`](/docs/api-model#set_param).** If you do
  _have_ to change the params during the forward function or the backprop
  callback, `set_param` is the best way to do it – but you should try to prefer
  other designs, as other code may not expect params to change during forward or
  backprop.

- **Don't call [`Model.set_dim`](/docs/api-model#set_dim).** There's not really
  a need to do this, and you're likely to cause a lot of problems for other
  layers. If the parent layer checks a child dimension and then invokes it, the
  parent should not have to double check that the dimensions of the child have
  changed.

- **It's okay to update the [`Model.attrs`](/docs/api-model#properties).** If
  you do need to change state during the forward pass (for instance, for batch
  normalization), `model.attrs["some_attr"] = new_value` is the best approach.
  Of course, all else being equal, side-effects are worse than no side-effects –
  but sometimes it's just what you need to do. Consequently, your layer should
  expect that child layers are allowed to modify or set attrs during their
  forward or backward functions.

You can avoid any of the read-only stuff by following a policy of never
modifying your inputs, and always making a copy of your outputs if you need to
retain them. This is the most conservative approach, and for most situations
it's what we would recommend – it ensures your layer will work well even when
combined with other layers that aren't written carefully. However, you'll
sometimes be writing code where it's reasonable to worry about unnecessary
copies. In these situations, the read-only flags work sort of like traffic
rules. If everyone cooperates, you can go faster, but if someone forgets to
indicate, there might be a crash.

### Writing the backprop callback {#backprop}

Your `forward` function must return a callback to compute gradients of the
parameters and weights during training. The callback must accept inputs that
match the outputs of the forward pass and return outputs that match the inputs
to the forward pass (the specifics of "matching" might differ between types, but
for arrays, assume it means the same shape and type). If the forward function is
`Y = forward(X)`, then the backprop callback should be `dX = backprop(dY)`, with
`X.shape == dX.shape` and `Y.shape == dY.shape`.

<infobox>

#### A more technical explanation

Thinc's `forward` functions behave like functions transformed by JAX's
[`jax.vjp`](https://jax.readthedocs.io/en/latest/jax.html#jax.vjp) function.
Some readers may prefer their more technical description, which is that the
`forward` function returns:

A`(primals_out, vjpfun)` pair, where `primals_out` is `fun(*primals)`. `vjpfun`
is a function from a cotangent vector with the same shape as `primals_out` to a
tuple of cotangent vectors with the same shape as `primals`, representing the
vector-Jacobian product of `fun` evaluated at `primals`.

</infobox>

Your backprop callback will often refer to variables in the outer scope. This
allows you to easily **reuse state from the forward pass**. The Python runtime
will increment the reference count of all variables that your backprop callback
references, and then decrement the reference counts once the callback is
destroyed. We can see this working by attaching a `__del__` method to some
classes, which show when the objects are being destroyed.

<grid>

```python
### {small="true"}
class Thing:
    def __del__(self):
        print("Deleted thing")

class Other:
    def __del__(self):
        print("Deleted other")

def outer():
    thing = Thing()
    other = Other()
    def inner():
        return thing
    return inner
```

```python
>>> callback = outer()
Deleted other

>>> callback = None
Deleted thing
```

</grid>

This behavior makes managing memory very easy: objects you reference will be
kept alive, and objects you don't are eligible to be freed. You should therefore
avoid unnecessary references if possible. For instance, if you only need the
shape of an array, it is better to assign that to a local variable, rather than
accessing it via the parent array.

Your backprop callback is not guaranteed to be called, so you should not rely on
it to compute side-effects. It is also valid for code to execute the backprop
callback more than once, so your function should be prepared for that. However,
it is not valid to call the backprop callback if the forward function was
executed with `is_train=False`, so you can implement predict-only optimizations.
It is also invalid for layers to change each others' parameters or dimensions
during execution, so your function does not need to be prepared for that.

Thinc does leave you with the responsibility for **calculating the gradients
correctly**. If you do not get them right, your layer will not learn correctly.
If you're having trouble, you might find
[Thinc's built-in layers](https://github.com/explosion/thinc/blob/master/thinc/layers)
a helpful reference, as they show how the backward pass should look for a number
of different situations. They also serve as examples for how we would suggest
you structure your code to make calculating the backward pass easier. Naming is
especially important: you need to see the order of steps in the forward pass and
unwind them in reverse. For complex cases, it also helps a lot to break out
calculations into **helper functions that return a backprop callback**. Then
your outer layer can simply call the callbacks in reverse. It also helps to
follow a consistent naming scheme for these callbacks. We usually either name
our callbacks by the result returned (like `get_dX)`, or the variable that
you'll pass in (like `bp_dY`).

<infobox variant="warning">

Frustratingly, your layer might still limp on even with a mistake in the
gradient calculation, making the problem hard to detect. As is often the case in
programming, almost correct is the worst kind of incorrect. Often it's best to
check your work with some tooling. Both
[JAX](https://jax.readthedocs.io/en/latest/) and
[Tangent](https://github.com/google/tangent) are very helpful for this.

</infobox>

Often you'll write layers that are not meaningfully differentiable, or for which
you do not need the gradients of the inputs. For instance, you might have a
layer that lower-cases textual inputs. In these cases, the backprop callback
should return **some appropriate falsy value** of the same type as the input to
avoid raising spurious type errors. For instance, if the input is a list of
strings, you would return an empty list; if the input is an array, you can
return an empty array.

### The initialize function {#weights-layers-init}

The last function you may need to define is the initializer, or "init",
function. Like the `forward` function, your `init` function will be stored on
the model instance, and then called by the
[`Model.initialize`](/docs/api-model#initialize) method. You do not have to
expect that the function will be called in other situations.

Your `init` function will be called with an instance of your model and two
optional arguments, which may provide an example batch of inputs (`X`) and an
example batch of outputs (`Y`). The arguments may be provided positionally or by
keyword (so the naming is significant: you must call the arguments `X` and `Y`).

Your model can use the provided example data to **help calculate unset
dimensions**, assist with parameter initialization, or calculate attributes. It
is valid for your construction function to return your model with missing
information, hoping that the information will be filled in later or at
initialization via the example data. If the example data is not provided and
you're left with unset dimensions or other incomplete state, you should raise an
error. It is up to you to decide how you should handle receiving conflict
information at construction and initialization time. Sometimes it will be better
to overwrite the construction data, and other times it will be better to raise
an error.

```python
### Initialize function
def random_chain_init(model, X=None, Y=None):
    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", X.shape[1])
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", Y.shape[1])
    for child_layer in model.layers:
        child_layer.initialize(X=X, Y=Y)
```

The `model.initialize` method will not call your `init` function with any extra
arguments, but you will often want to parameterize your `init` with settings
from the constructor. This is especially common for weight initialization: there
are many possible schemes, and the choice is often an important hyper-parameter.
While you can communicate this information via the `attrs` dict, our favorite
solution is to use
[`functools.partial`](https://docs.python.org/3.6/library/functools.html#functools.partial).
Partial function application lets you fill in a function's arguments at two
different times, instead of doing it all at once --- which is exactly what we
need here. This lets you write the `init` function with extra arguments, which
you provide at creation time when the information is available.

```python
### Passing init args using partial {highlight="1,10,16"}
from functools import partial
from thinc.api import Model

def constructor(nO=None, initial_value=1):
    return Model(
        "my-model",
        forward,
        dims={"nO": nO},
        params={"b": None},
        init=partial(init, initial_value=initial_value)
    )

def forward(model, X, is_train):
    ...

def init(initial_value, model, X=None, Y=None):
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", None)
    if not model.get_dim("nO"):
        raise ValueError(f"Cannot initialize {model.name}: dimension nO unset")
    b = model.ops.alloc1f(model.get_dim("nO"))
    b += initial_value
    model.set_param("b", b)
```

<infobox>

You can find real examples of the partial-application pattern in
[`thinc.layers`](https://github.com/explosion/thinc/blob/master/thinc/layers).
The
[`Relu`](https://github.com/explosion/thinc/blob/master/thinc/layers/relu.py),
[`Mish`](https://github.com/explosion/thinc/blob/master/thinc/layers/mish.py)
and
[`Maxout`](https://github.com/explosion/thinc/blob/master/thinc/layers/maxout.py)
layers provide straightforward examples, among others.

</infobox>

A call to [`Model.initialize`](/docs/api-model#initialize) will trigger
subsequent calls down the model tree, as **each layer is responsible for calling
the initialize method of each of its children** (although not necessarily in
order). Prior to training, you can rely on your model's `init` function being
called at least once in between the constructor and the first execution of your
forward function. It is invalid for a parent to not call `initialize` on one of
its children, and it is invalid to start training without calling
`Model.initialize` first.

However, it _is_ valid to create a layer and then load its state back with
[`Model.from_dict`](/docs/api-model#from_dict),
[`Model.from_bytes`](/docs/api-model#from_bytes) or
[`Model.from_disk`](/docs/api-model#from_disk) without calling
`model.initialize` first. This means that you should not write any side-effects
in your `init` function that will not be replicated by deserialization.
Deserialization will restore dimensions, parameters, node references and
serializable attributes, but it will not replicate any changes you have made to
the layer's node structure, such as changes to the `model.layers` list. You
should therefore avoid making those changes within the `init`, or your model
will not deserialize correctly.

It is valid for your `init` function to be called **more than once**. This
usually happens because the same model instance occurs twice within a a tree
(which is allowed). Your `initialize` function should therefore be prepared to
run twice. If you're simply setting dimensions and allocating and initializing
parameters, having the init function run will generally be unproblematic.
However, in special situations you may need to call external setup code, in
which case having your `init` function run twice could be problematic. The best
solution would probably be to set a global variable, possibly using the
`model.id` attribute to allow the `init` to run once per model instance. You
could also use partial function application to attach the flag in a mutable
argument variable.

### Inspecting and updating model state {#model-state}

As you build more complicated models, you'll often need to inspect your model in
various ways. This is especially important when you're writing your own layers.
Here's a quick summary of the different types of information you can attach and
query.

|                                                                                                                                                                                                                                      |                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`Model.id`](/docs/api-model#attributes)                                                                                                                                                                                             | A numeric identifier, to distinguish different model instances. During [`Model.__init__`](/docs/api-model#init), the `Model.global_id` class attribute is incremented and the next value is used for the `model.id` value.                                                                                                           |
| [`Model.name`](/docs/api-model#attributes)                                                                                                                                                                                           | A string name for the model.                                                                                                                                                                                                                                                                                                         |
| [`Model.layers`](/docs/api-model#properties) [`Model.walk`](/docs/api-model#walk)                                                                                                                                                    | List the immediate sublayers of a model, or iterate over the model's whole subtree (including the model itself).                                                                                                                                                                                                                     |
| [`Model.shims`](/docs/api-model#properties)                                                                                                                                                                                          | Wrappers for external libraries, such as PyTorch and TensorFlow. [`Shim`](/docs/api-model#shim) objects hold a reference to the external object, and provide a consistent interface for Thinc to work with, while also letting Thinc treat them separately from `Model` instances for the purpose of serialization and optimization. |
| [`Model.has_dim`](/docs/api-model#has_dim) [`Model.get_dim`](/docs/api-model#get_dim) [`Model.set_dim`](/docs/api-model#set_dim) [`Model.dim_names`](/docs/api-model#properties)                                                     | Check, get, set and list the layer's **dimensions**. A dimension is an integer value that affects a model's parameters or the shape of its input data.                                                                                                                                                                               |
| [`Model.has_param`](/docs/api-model#has_param) [`Model.get_param`](/docs/api-model#get_param) [`Model.set_param`](/docs/api-model#set_param) [`Model.param_names`](/docs/api-model#properties)                                       | Check, get, set and list the layer's **weights parameters**. A parameter is an array that can have a gradient and can be optimized.                                                                                                                                                                                                  |
| [`Model.has_grad`](/docs/api-model#has_grad) [`Model.get_grad`](/docs/api-model#get_grad) [`Model.set_grad`](/docs/api-model#set_grad) [`Model.inc_grad`](/docs/api-model#inc_grad) [`Model.grad_names`](/docs/api-model#properties) | Check, get, set, increment and list the layer's **weights gradients**. A gradient is an array of the same shape as a weights parameter, that increments values used to update the parameter during training.                                                                                                                         |
| [`Model.has_ref`](/docs/api-model#has_ref) [`Model.get_ref`](/docs/api-model#get_ref) [`Model.set_ref`](/docs/api-model#set_ref) [`Model.ref_names`](/docs/api-model#properties)                                                     | Check, get, set and list the layer's **node references**. A node reference lets you easily refer to particular nodes within your model's subtree. For instance, if you want to expose the embedding table from your model, you can add a reference to it.                                                                            |
| [`Model.attrs`](/docs/api-model#properties)                                                                                                                                                                                          | A dict of the layer's **attributes**. Attributes are other information the layer needs, such as configuration or settings. You should ensure that attribute values you set are either JSON-serializable, or support a `to_bytes` method, or the attribute will prevent model serialization.                                          |

### Naming conventions {#naming-conventions}

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

---

## Serializing models and data {#serializing}

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
yourself first, and then use that object to load in the state.

To make this easier, you'll usually want to put your model creation code inside
a function, and then **register it**. The [registry](/docs/api-config#registry)
allows you to look up the function by name later, so you can pass along all the
details to recreate your model in one message. Check out our
[guide on the config system](/docs/usage-config) for more details.

### Serializing attributes {#serializing-attrs}

When you call [`Model.to_bytes`](/docs/api-model#to_bytes) or
[`Model.to_disk`](/docs/api-model#to_disk), the model and its layers, weights,
parameters and attributes will be serialized to a byte string. Calling
[`Model.from_bytes`](/docs/api-model#from_bytes) or
[`Model.from_disk`](/docs/api-model#from_disk) lets you load a model back in. By
default, Thinc uses MessagePack, which works out-of-the-box for all
JSON-serializable data types. The `serialize_attr` and `deserialize_attr`
functions that Thinc uses under the hood are
[single-dispatch generic functions](https://docs.python.org/3/library/functools.html#functools.singledispatch).
This means that you can **register different versions** of them that are chosen
based on the **value and type of the attribute**.

For example, let's say your model takes attributes that are
[`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html)
objects:

```python
### Model with custom attrs {highlight="4"}
from thinc.api import Model
import pandas as pd

attrs = {"df": pd.DataFrame([10, 20, 30], columns=["a"])}
model = Model("custom-model", lambda X: (X, lambda dY: dY), attrs=attrs)
```

To tell Thinc how to save and load them, you can use the
`@serialize_attr.register` and `@deserialize_attr.register` decorators with the
type `pd.DataFrame`. Whenever Thinc encounters an attribute value that's a
dataframe, it will use these functions to serialize and deserialize it.

```python
### Custom attr serialization {highlight="5-6,11-12"}
from thinc.api import serialize_attr, deserialize_attr
import pandas as pd
import numpy

@serialize_attr.register(pd.DataFrame)
def serialize_dataframe(_, value, name, model):
    """Serialize the value (a dataframe) to bytes."""
    rec = value.to_records(index=False)
    return rec.tostring()

@deserialize_attr.register(pd.DataFrame)
def deserialize_dataframe(_, value, name, model):
    """Deserialize bytes to a dataframe."""
    rec = numpy.frombuffer(value, dtype="i")
    return pd.DataFrame().from_records(rec)
```

The first argument of the function is always an instance of the attribute. This
is used to decide which function to call. The `value` is the value to save or
load – a dataframe to serialize or the bytestring to deserialize. The functions
also give you access to the string name of the current attribute and the `Model`
instance. This is useful if you need additional information or if you want to
perform other side-effects when loading the data back in. For example, you could
check if the model has another attribute specifying the data type of the array
and use that when loading back the data:

```python
### {highlight="4-5"}
@deserialize_attr.register(pd.DataFrame)
def deserialize_dataframe(_, value, name, model):
    """Deserialize bytes to a dataframe."""
    dtype = model.attrs.get("dtype", "i")
    rec = numpy.frombuffer(value, dtype=dtype)
    return pd.DataFrame().from_records(rec)
```

Since the attribute value is used to decide which serialization and
deserialization function to use, make sure that your model defines **default
values** for its attributes. This way, the correct function will be called when
you run `model.from_bytes` or `model.from_disk` to load in the data.

```diff
- attrs = {"df": None}
+ attrs = {"df": pd.DataFrame([10, 20, 30], columns=["a"])}
model = Model("custom-model", lambda X: (X, lambda dY: dY), attrs=attrs)
```
