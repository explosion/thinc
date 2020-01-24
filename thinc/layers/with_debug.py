from typing import Optional, Callable, Any, Tuple
from thinc.api import Model


def with_debug(
    layer: Model,
    name: Optional[str] = None,
    *,
    on_init: Optional[Callable[[Model, Any, Any], None]] = None,
    on_forward: Optional[Callable[[Model, Any, bool], None]] = None,
    on_backprop: Optional[Callable[[Any], None]] = None,
):
    """Debugging layer that wraps any layer and allows executing callbacks
    during the forward pass, backward pass and initialization. The callbacks
    will receive the same arguments as the functions they're called in.
    """
    name = layer.name if name is None else name

    def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
        if on_forward:
            on_forward(model, X, is_train)
        layer_Y, layer_callback = layer(X, is_train=is_train)

        def backprop(dY: Any) -> Any:
            if on_backprop:
                on_backprop(dY)
            return layer_callback(dY)

        return layer_Y, backprop

    def init(model: Model, X: Any, Y: Any) -> Model:
        if on_init:
            on_init(model, X, Y)
        return layer.initialize(X, Y)

    return Model(f"debug:{name}", forward, init=init)
