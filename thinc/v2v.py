from typing import Optional

from .neural._classes.model import Model  # noqa: F401
from .neural._classes.affine import Affine  # noqa: F401
from .neural._classes.relu import ReLu  # noqa: F401
from .neural._classes.maxout import Maxout  # noqa: F401
from .neural._classes.softmax import Softmax  # noqa: F401
from .neural._classes.mish import Mish  # noqa: F401
from ._registry import registry


@registry.layers.register("ReLu.v1")
def make_ReLu(outputs: int, inputs: Optional[int] = None):
    return ReLu(nO=outputs, nI=inputs)


@registry.layers.register("Mish.v1")
def make_Mish(outputs: int, inputs: Optional[int] = None):
    return Mish(nO=outputs, nI=inputs)


@registry.layers.register("Softmax.v1")
def make_Softmax(outputs: Optional[int] = None, inputs: Optional[int] = None):
    return Softmax(nO=outputs, nI=inputs)


@registry.layers.register("Maxout.v1")
def make_Maxout(outputs: int, inputs: Optional[int] = None, *, pieces: int):
    return Maxout(nO=outputs, nI=inputs, pieces=pieces)
