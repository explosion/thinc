from typing import Optional, Callable, Any, Tuple

from ..model import Model


do_nothing = lambda *args, **kwargs: None


def with_debug(
    layer: Model,
    name: Optional[str] = None,
    *,
    on_init: Callable[[Model, Any, Any], None] = do_nothing,
    on_forward: Callable[[Model, Any, bool], None] = do_nothing,
    on_backprop: Callable[[Any], None] = do_nothing,
):
    """Debugging layer that wraps any layer and allows executing callbacks
    during the forward pass, backward pass and initialization. The callbacks
    will receive the same arguments as the functions they're called in.
    """
    name = layer.name if name is None else name

    def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
        layer = model.layers[0]
        on_forward(model, X, is_train)
        layer_Y, layer_callback = layer(X, is_train=is_train)

        def backprop(dY: Any) -> Any:
            on_backprop(dY)
            return layer_callback(dY)

        return layer_Y, backprop

    def init(model: Model, X: Any, Y: Any) -> Model:
        on_init(model, X, Y)
        return layer.initialize(X, Y)

    return Model(f"debug:{name}", forward, init=init, layers=[layer])
