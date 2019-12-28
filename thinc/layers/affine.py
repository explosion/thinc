from .base import Model
from ..initializers import xavier_uniform_init, zero_init

Array = "Array"  # TODO: Fix type-check for this


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    W = model.get_param("W")
    b = model.get_param("b")

    Y = model.ops.gemm(X, W, trans2=True)
    Y += b

    def affine_backward(dY):
        model.ops.inc_grad("b", dY.sum(axis=0))
        model.ops.inc_grad("W", model.ops.gemm(dY, X, trans2=True))
        return model.ops.gemm(dY, W)

    return Y, affine_backward


def create_init(init_W: Callable, init_b: Callable) -> Callable:
    def do_affine_init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if X is not None:
            model.set_dim("nI", X.shape[1])
        if Y is not None:
            model.set_dim("nO", Y.shape[1])
        W = model.ops.allocate((model.get_dim("nO"), model.get_dim("nI")))
        b = model.ops.allocate((model.get_dim("nO"),))
        init_W(W, inplace=True)
        init_b(b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)

    return do_affine_init


def make_Affine(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    init_W: Callable = xavier_uniform_init,
    init_b: Callable = zero_init,
) -> Model:
    return Model(
        "affine",
        forward,
        init=create_init(init_W, init_b),
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None},
        layers=[],
        attrs={},
    )
