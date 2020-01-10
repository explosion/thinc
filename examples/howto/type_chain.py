from thinc.api import chain, ReLu, MaxPool, Softmax, chain

# This example should be run with mypy. This is an example of type-level checking 
# for network validity.
#
# We first define an invalid network.
# It's invalid because MaxPool expects Floats3d as input, while ReLu produces
# Floats2d as output. chain has type-logic to verify input and output types
# line up.
#
# You should see the error an error,
# examples/howto/type_chain.py:10: error: Cannot infer type argument 2 of "chain"
bad_model = chain(ReLu(10), MaxPool(), Softmax())

# Now let's try it with a network that does work, just to be sure.
good_model = chain(ReLu(10), ReLu(10), Softmax())

# Finally we can reveal_type on the good model, to see what it thinks.
reveal_type(good_model)
