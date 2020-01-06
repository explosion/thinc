from typing import Any, Optional, Tuple, Callable, Dict
import contextlib
from pathlib import Path


class Shim:
    """Define a basic interface for external models. Users can create subclasses
    of 'shim' to wrap external libraries. We provide shims for PyTorch.

    The Thinc Model class treats Shim objects as a sort of special type of
    sublayer: it knows they're not actual Thinc Model instances, but it also
    knows to talk to the shim instances when doing things like using transferring
    between devices, loading in parameters, optimization. It also knows Shim
    objects need to be serialized and deserialized with to/from bytes/disk,
    rather than expecting that they'll be msgpack-serializable.
    """

    global_id = 0

    _model: Any
    _optimizer: Optional[Any]

    def __init__(self, model: Any):
        Shim.global_id += 1
        self.id = Shim.global_id
        self._model = model
        self._optimizer = None

    def __call__(
        self, args: Tuple, kwargs: Dict, is_train: bool
    ) -> Tuple[Any, Callable[[Any], Any]]:
        raise NotImplementedError

    def predict(self, args: Tuple, kwargs: Dict) -> Any:
        Y, backprop = self(args, kwargs, is_train=False)
        return Y

    def begin_update(
        self, args: Tuple, kwargs: Dict
    ) -> Tuple[Any, Callable[[Any], Any]]:
        return self(args, kwargs, is_train=True)

    def finish_update(self, optimizer, **kwargs):
        raise NotImplementedError

    @contextlib.contextmanager
    def use_params(self, params):
        raise NotImplementedError

    def to_gpu(self, device_num):
        raise NotImplementedError

    def to_cpu(self):
        raise NotImplementedError

    def to_disk(self, path):
        bytes_data = self.to_bytes()
        with Path(path).open("wb") as file_:
            file_.write(bytes_data)

    def from_disk(self, path) -> "Shim":
        with Path(path).open("rb") as file_:
            bytes_data = file_.read()
        return self.from_bytes(bytes_data)

    def to_bytes(self):
        raise NotImplementedError

    def from_bytes(self, data) -> "Shim":
        raise NotImplementedError
