from typing import Tuple, Callable, Optional

from ..model import Model
from ..config import registry
from ..types import Padded, Array2d


ValT = Array2d


@registry.layers("with_padded2array.v0")
def with_padded2array(layer: Model[ValT, ValT]) -> Model[Padded, Padded]:
    """Apply a layer to a padded batch.
    
    The 3d padded batch is reshaped from a (T, B, N) batch into a (T*B, N) batch, 
    and passed through the layer. The output is then reshaped back and a Padded
    datatype is produced.

    Another way of producing an array from a Padded is to go via a Ragged object.
    After padded2ragged, you'll get a concatenated array with the padding values
    squeezed out. This stops the layer from having to predict over them, at the
    cost of shuffling the memory around more. The latter approach will sometimes
    be faster on CPU, while this approach is usually faster on GPU.
    """
    return Model(
        f"with_padded2array-{layer.name}",
        forward,
        init=init,
        layers=[layer],
    )


def forward(model: Model[Padded, Padded], Xp: Padded, is_train: bool) -> Tuple[Padded, Callable]:
    layer: Model[ValT, ValT] = model.layers[0]
    X = Xp.data.reshape((-1, Xp.data.shape[2]))
    Y2d, get_dX = layer(X, is_train)
    Y = Y2d.reshape((Xp.data.shape[0], Xp.data.shape[1], -1))

    def backprop(dYp: Padded) -> Padded:
        dY = dYp.data.reshape((-1, dYp.data.shape[2]))
        dX2d = get_dX(dY)
        dX = dX2d.reshape((dYp.data.shape[0], dYp.data.shape[1], -1))
        return Padded(dX, dYp.size_at_t, dYp.lengths, dYp.indices)

    return Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices), backprop


def init(
    model: Model[Padded, Padded], X: Optional[Padded] = None, Y: Optional[Padded] = None
) -> None:
    layer: Model[Array2d, Array2d] = model.layers[0]
    layer.initialize(
        X=X.data.reshape((-1, X.data.shape[2])) if X is not None else None,
        Y=Y.data.reshape((-1, Y.data.shape[2])) if Y is not None else None
    )
