from thinc.api import Model, Linear, ReLu, Softmax, Dropout, Embed, Maxout, MaxPool
from thinc.api import LayerNorm, MeanPool, residual, chain, clone, concatenate
from thinc.api import with_list2array, add, zero_init, xavier_uniform_init
import numpy

# The maximally idiomatic version:

n_hidden = 10
depth = 4

with Model.define_operators({">>": chain, "**": clone}):
    model = (
        (Linear(n_hidden, init_W=xavier_uniform_init) >> ReLu()) ** depth
        >> Linear(n_hidden, init_W=zero_init)
        >> Softmax()
    )

# Building up to that, step by step...

# The thinc.layers package provides functions that create Model instances.
# (Thinc tries to avoid inheritance, preferring function composition.)
# The Linear function gives you a model that computes Y = X @ W.T + b
# (the function is defined in thinc.layers.linear.forward)
n_in = numpy.zeros((128, 16), dtype="f")
n_out = numpy.zeros((128, 10), dtype="f")

model = Linear(n_in, n_out, init_W=zero_init)

# Models support *dimension inference from data*. You can defer some or all
# of the dimensions.
model = Linear(init_W=zero_init)
assert model.has_dim("nO") is None
assert model.has_dim("nI") is None
X = numpy.zeros((128, 16), dtype="f")
Y = numpy.zeros((128, 10), dtype="f")
model.initialize(X=X, Y=Y)
assert model.get_dim("nI") == 16
assert model.get_dim("nO") == 10

# The chain function wires two model instances together, with a feed-forward
# relationship. Dimension inference is especially helpful here.
n_hidden = 128
model = chain(Linear(n_hidden, init_W=xavier_uniform_init), Linear(init_W=zero_init),)
model.initialize(X=X, Y=Y)
assert model.get_dim("nI") == 16
assert model.get_dim("nO") == 10
assert model.layers[0].get_dim("nO") == n_hidden

# We call functions like 'chain' *combinators*. Combinators one or more models
# as arguments, and return another model instance, without introducing any
# new weight parameters.
# Another useful combinator is concatenate:

model = concatenate([Linear(n_hidden), Linear(n_hidden)])
model.initialize(X=X)
assert model.get_dim("nI") == X.shape[1]
assert model.get_dim("nO") == n_hidden * 2

# The concatenate function produces a layer that runs the child layers separately,
# and then concatenates their outputs together. This is often useful for combining
# features from different sources. For instance, we use this all the time to
# build spaCy's embedding layers.

# Some combinators work on a layer and a numeric argument. For instance, the
# clone combinator creates a number of copies of a layer, and chains them together
# into a deep feed-forward network.
# The shape inference is especially handy here: we want the first and last layers
# to have different shapes, so we can avoid providing any dimensions into the
# layer we clone. We then just have to specify the first layer's output size,
# and we can let the rest of the dimensions be inferred from the data.

model = clone(Linear(), 5)
model.layers[0].set_dim("nO", n_hidden)
model.initialize(X=X, Y=Y)

# We can apply 'clone' to model instances that have child layers, making it easy
# to define more complex architectures. For instance, we often want to attach
# an activation function and dropout to a linear layer, and then repeat that
# substructure a number of times:

# Of course, you can make whatever intermediate functions you find helpful:


def Hidden(dropout=0.2):
    return chain(Linear(), ReLu(), Dropout(dropout))


model = clone(Hidden(0.2), 5)

# Some combinators are unary functions: they take only one model. These are
# usually input and output transformations. For instance, the with_list2array
# combinator produces a model that flattens lists of arrays into a single array,
# and then calls the child layer to get the flattened output. It then reverses
# the transformation on the output.

model = with_list2array(Linear(4, 2))
Xs = [model.ops.alloc_f2d(10, 2, dtype="f")]
Ys = model.predict(Xs)
assert Ys[0].shape == (10, 4)

# The combinator system makes it easy to wire together complex models very concisely.
# A concise notation is a huge advantage, because it lets you read and review your
# model with less clutter --- making it easy to spot mistakes, and easy to make
# changes. For the ultimate in concise notation, you can also take advantage of
# Thinc's operator overloading, which lets you use an infix notation. Operator
# overloading can lead to unexpected results, so you have to enable the overloading
# explicitly in a context manager. This also lets you control how the operators
# are bound, making it easy to use the feature with your own combinators.
# For instance, here is a definition for a text classification network:
nH = 5

with Model.define_operators({">>": chain, "|": concatenate, "+": add, "**": clone}):
    model = (
        with_list2array(
            (Embed(128, column=0) + Embed(64, column=1))
            >> Maxout(nH)
            >> LayerNorm()
            >> Dropout(0.2)
        )
        >> (MaxPool() | MeanPool())
        >> residual(ReLu() >> Dropout(0.2)) ** 2
        >> Softmax()
    )

# The network above will expect a list of arrays as input, where each array
# should have two columns with different numeric identifier features. The
# two features will be embedded using separate embedding tables, and the two
# vectors added and passed through a Maxout layer with layer normalization
# and dropout.
# The sequences then pass through two pooling functions, and the concatenated
# results are passed through 2 ReLu layers with dropout and residual connections.
# Finally, the sequence vectors are passed through an output layer, which has
# a softmax activation.
