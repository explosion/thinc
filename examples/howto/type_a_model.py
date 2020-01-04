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
