from typing import Callable, List, Optional, Tuple, TypeVar, cast

from thinc.initializers import xavier_uniform_init, zero_init
from thinc.layers import ReLu, Softmax, chain
from thinc.layers.chain import chain
from thinc.layers.dropout import Dropout
from thinc.layers.layernorm import LayerNorm
from thinc.model import Model, create_init
from thinc.types import Array, Floats1d, Floats2d, Literal


InT = Floats2d
OutT = Floats2d


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")
    Y = model.ops.gemm(X, W, trans2=True)
    Y += b

    def backprop(dY: OutT) -> InT:
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", model.ops.gemm(dY, X, trans1=True))
        return model.ops.gemm(dY, W)

    return Y, backprop


InputUnits = TypeVar("InputUnits", bound=int)
OutputUnits = TypeVar("OutputUnits", bound=int)
BatchUnits = TypeVar("BatchUnits", bound=None)


def MyLayer(
    nO: Optional[InputUnits] = None,
    nI: Optional[OutputUnits] = None,
    *,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
) -> Model[Floats2d[BatchUnits, InputUnits], Floats2d[BatchUnits, OutputUnits]]:
    """Multiply inputs by a weights matrix and adds a bias vector."""
    model: Model[InT, OutT] = Model(
        "affine",
        forward,
        init=create_init({"W": init_W, "b": init_b}),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


# NOTE: the use of Literal[value] as a type annotation is important here :(
units: Literal[256] = 256
predictions: Literal[10] = 10

# Bad configuration, input does not match the units definition
MyModelInput = Floats2d[BatchUnits, Literal[768]]
units_and_model_incompatible: Model[MyModelInput, MyModelInput] = MyLayer(units)
# ERR: Argument 1 to "MyLayer" has incompatible type "Literal[256]"; expected "Optional[Literal[768]]"


# Bad configuration, output dimensions do not match the inferred inputs
BadOutputInput = Floats2d[BatchUnits, Literal[256]]
BadOutputOutput = Floats2d[BatchUnits, Literal[1024]]
bad_output: Model[BadOutputInput, BadOutputOutput] = MyLayer(units, predictions)
# ERR: Argument 2 to "MyLayer" has incompatible type "Literal[10]"; expected "Optional[Literal[1024]]"


# Bad mnist model, affine(units) != softmax(10)
mnist_units: Literal[768] = 768
MNISTInput = Floats2d[BatchUnits, Literal[768]]
MNISTOutput = Floats1d[Literal[10]]
mnist_bad_output_shape: Model[MNISTInput, MNISTOutput] = MyLayer(mnist_units)
# ERR: Incompatible types in assignment (
#   expression has type "Model[Floats2d[Any, Literal[768]], Floats2d[Any, <nothing>]]",
#   variable has type "Model[Floats2d[Any, Literal[768]], Floats1d[Literal[10]]]"
# )

# Good configuration, input/output types match the types of provided nO/nI vars
GoodInput = Floats2d[BatchUnits, Literal[256]]
GoodOutput = Floats2d[BatchUnits, Literal[10]]
model: Model[GoodInput, GoodOutput] = MyLayer(units, predictions)

