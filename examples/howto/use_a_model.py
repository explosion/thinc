from thinc.api import Linear, Adam
import numpy


X = numpy.zeros((128, 10), dtype="f")
dY = numpy.zeros((128, 10), dtype="f")

model = Linear(10, 10)

# Run the model over some data
Y = model.predict(X)

# Get a callback to backpropagate
Y, backprop = model.begin_update(X)

# Run the callback to calculate the gradient with respect to the inputs.
# If the model has trainable parameters, gradients for the parameters are
# accumulated internally, as a side-effect.
dX = backprop(dY)

# The backprop() callback only increments the parameter gradients, it doesn't
# actually change the weights. To increment the weights, call model.finish_update(),
# passing it an optimizer:

optimizer = Adam()
model.finish_update(optimizer)

# You can get and set dimensions, parameters and attributes by name:
dim = model.get_dim("nO")
W = model.get_param("W")
model.set_attr("hello", "world")
assert model.get_attr("hello") == "world"

# You can also retrieve parameter gradients, and increment them explicitly:
dW = model.get_grad("W")
model.inc_grad("W", dW * 0.1)

# Finally, you can serialize models using the model.to_bytes() and model.to_disk()
# methods, and load them back with .from_bytes() and from_disk().
