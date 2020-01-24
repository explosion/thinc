from typing import Dict, Tuple

from ..types import Array


KeyT = Tuple[int, str]


class ParamServer:
    """Serve parameters for a single process."""

    _params: Dict[KeyT, Array] = {}
    _grads: Dict[KeyT, Array] = {}

    def __init__(self, params: Dict[KeyT, Array] = {}, grads: Dict[KeyT, Array] = {}):
        self._params = dict(params)
        self._grads = dict(grads)

    @property
    def param_keys(self) -> Tuple[KeyT, ...]:
        """Get the names of registered parameter (including unset)."""
        return tuple(self._params.keys())

    @property
    def grad_keys(self) -> Tuple[KeyT, ...]:
        return tuple([key for key in self.param_keys if self.has_grad(*key)])

    def has_param(self, model_id: int, name: str) -> bool:
        return (model_id, name) in self._params

    def has_grad(self, model_id: int, name: str) -> bool:
        return (model_id, name) in self._grads

    def get_param(self, model_id: int, name: str) -> Array:
        return self._params[(model_id, name)]

    def get_grad(self, model_id: int, name: str) -> Array:
        return self._grads[(model_id, name)]

    def set_param(self, model_id: int, name: str, value: Array) -> None:
        self._params[(model_id, name)] = value

    def set_grad(self, model_id: int, name: str, value: Array) -> None:
        self._grads[(model_id, name)] = value

    def inc_grad(self, model_id: int, param_name: str, value: Array) -> None:
        if not self.has_grad(model_id, param_name):  # pragma: no cover
            # Adjustment for Jax
            if hasattr(value, "copy"):
                self._grads[(model_id, param_name)] = value.copy()
            else:
                self._grads[(model_id, param_name)] = value
        else:
            self._grads[(model_id, param_name)] += value
