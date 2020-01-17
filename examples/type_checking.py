from typing import List

from thinc.layers import ReLu, Softmax, chain
from thinc.model import Model

# Define Custom X/Y types
MyModelX = List[List[float]]
MyModelY = List[List[float]]
model: Model[MyModelX, MyModelY] = chain(
    ReLu(12), ReLu(12, dropout=0.2), Softmax(),
)
# ERROR: incompatible type "bool", expected "List[List[float]]"
model(False)
# ERROR: List item 0 has incompatible type "str"; expected "float"
model.begin_update([["0"]])
# ERROR: incompatible type "bool", expected "List[List[float]]"
model.predict(True)


# This example should be run with mypy. This is an example of type-level checking
# for network validity.
#
# We first define an invalid network.
# It's invalid because MaxPool expects Array3d as input, while ReLu produces
# Array2d as output. chain has type-logic to verify input and output types
# line up.
#
# You should see the error an error,
# examples/howto/type_chain.py:10: error: Cannot infer type argument 2 of "chain"
bad_model = chain(ReLu(10), MaxPool(), Softmax())

# Now let's try it with a network that does work, just to be sure.
good_model = chain(ReLu(10), ReLu(10), Softmax())

# Finally we can reveal_type on the good model, to see what it thinks.
reveal_type(good_model)
