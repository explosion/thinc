from typing import Dict, Tuple

from ..types import FloatsXd
from ..util import get_array_module


KeyT = Tuple[int, str]


class ParamServer:
    """Serve parameters for a single process."""

    _params: Dict[KeyT, FloatsXd] = {}
    _grads: Dict[KeyT, FloatsXd] = {}

    def __init__(
        self, params: Dict[KeyT, FloatsXd] = {}, grads: Dict[KeyT, FloatsXd] = {}
    ):
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

    def get_param(self, model_id: int, name: str) -> FloatsXd:
        return self._params[(model_id, name)]

    def get_grad(self, model_id: int, name: str) -> FloatsXd:
        return self._grads[(model_id, name)]

    def set_param(self, model_id: int, name: str, value: FloatsXd) -> None:
        self._params[(model_id, name)] = value

    def set_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        self._grads[(model_id, name)] = value

    def inc_grad(self, model_id: int, param_name: str, value: FloatsXd) -> None:
        if not self.has_grad(model_id, param_name):  # pragma: no cover
            # Adjustment for Jax
            if hasattr(value, "copy"):
                self._grads[(model_id, param_name)] = value.copy()
            elif not value.flags["C_CONTIGUOUS"]:
                xp = get_array_module(value)
                self._grads[(model_id, param_name)] = xp.ascontiguousarray(value)
            else:
                self._grads[(model_id, param_name)] = value
        else:
            self._grads[(model_id, param_name)] += value
