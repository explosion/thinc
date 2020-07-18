from typing import List, Dict, Tuple, Any, Optional
from timeit import default_timer as timer
from dataclasses import dataclass

from ..types import FloatsXd


KeyT = Tuple[int, str]


def make_key(model_id: int, name: str) -> Tuple[int, str]:
    return (model_id, name)

def encode_pointer(ray, value):
    return [ray.put(value)]
 
def decode_pointer(ray, value):
    if value is None:
        return None
    else:
        return ray.get(value)[0]


class RayHeadProxy:
    """Proxy for the 'head' worker that owns the optimizer and pushes
    parameter updates.
    """
    ray: Any
    conn: Any
    optimizer: Any
    quorum: int
    _key_to_i: Dict[Tuple[int, str], int]
    _grads: List[Optional[FloatsXd]]
    _params: List[FloatsXd]
    _versions: List[int]
    _grad_counts: List[int]

    def __init__(self, connection, optimizer, quorum: int, *, ray=None):
        if ray is None:
            import ray # type: ignore
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self.optimizer = optimizer
        self.quorum = quorum
        self._key_to_i = {}
        self._grads = []
        self._grad_counts = []
        self._params = []
        self._versions = []

    def step_schedules(self):
        self.optimizer.step_schedules()

    def get_param(self, model_id: int, name: str) -> FloatsXd:
        """Get a parameter from the connection."""
        key = make_key(model_id, name)
        i = self._key_to_i[key]
        return self._params[i]

    def set_param(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a parameter to the connection."""
        key = make_key(model_id, name)
        if key not in self._key_to_i:
            self._add_param(key, value)
        else:
            i = self._key_to_i[key]
            self._params[i] = value
            self._versions[i] += 1
            version = self._versions[i]
            self.conn.set_param.remote(
                key,
                version,
                i,
                value
            )

    def set_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a gradient to the connection."""
        key = make_key(model_id, name)
        i = self._key_to_i[key]
        version = self._versions[i]
        self.conn.set_grad.remote(
            version,
            i,
            value
        )

    def inc_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Increment a gradient to the connection."""
        key = make_key(model_id, name)
        i = self._key_to_i[key]
        grads_required = self._increment_local_grad(i, value)
        if grads_required >= 1:
            grads_required = self._maybe_pull_grads(i, grads_required)
        if grads_required <= 0:
            self._update_param((model_id, name))

    def _add_param(self, key: Tuple[int, str], value: FloatsXd) -> None:
        assert key not in self._key_to_i
        i = len(self._key_to_i)
        self._key_to_i[key] = i
        self._params.append(value)
        self._versions.append(0)
        self._grads.append(None)
        self._grad_counts.append(0)
        self.conn.set_param.remote(
            key,
            0,
            i,
            value
        )

    def _update_param(self, key: Tuple[int, str]):
        i = self._key_to_i[key]
        param, _ = self.optimizer(key, self._params[i], self._grads[i])
        self._grads[i] = None
        self._grad_counts[i] = 0
        self._versions[i] += 1
        self._params[i] = param
        self.conn.set_param.remote(key, self._versions[i], i, self._params[i])

    def _increment_local_grad(self, i: int, value: FloatsXd) -> int:
        self._grad_counts[i] += 1
        if self._grads[i] is None:
            self._grads[i] = value
        else:
            self._grads[i] += value
        return self.quorum - self._grad_counts[i]

    def _maybe_pull_grads(self, i: int, grads_required: int) -> int:
        """Check whether the remote has enough gradients to force us to
        update. If so, add them to our local gradient. Returns <= 0 if enough
        gradients for an update.
        """
        remote_grads = self.ray.get(
            self.conn.maybe_get_grads.remote(
                self._versions[i],
                i,
                grads_required
            )
        )
        if remote_grads is None:
            return grads_required
        else:
            self._grad_counts[i] += len(remote_grads)
            for grad in remote_grads:
                self._grads[i] += grad
            return grads_required - len(remote_grads)
 

class RayChildProxy:
    """Experimental"""
    ray: Any
    conn: Any
    _last_update: float
    _poll_freq: float
    _key_to_i: Dict[Tuple[int, str], int]
    _params: List[FloatsXd]
    _versions: List[int]
    _next_params: List[Any]

    def __init__(self, connection, *, ray=None, poll_freq=0.1):
        if ray is None:
            import ray
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self._key_to_i = {}
        self._params = []
        self._versions = []
        self._next_params = []
        self._last_update = timer()
        self._poll_freq = poll_freq
        self._sync_params()

    def get_param(self, model_id: int, name: str):
        """Get a parameter from the connection."""
        # TODO: What to do on first get?
        key = make_key(model_id, name)
        self._maybe_update_param(key)
        self._begin_params_pull() 
        i = self._key_to_i[key]
        return self._params[i]

    def set_param(self, model_id: int, name: str, value):
        """Child proxies don't set parameters, so this is a noop."""
        pass

    def set_grad(self, model_id: int, name: str, value):
        """Child proxies don't set gradients, so this is a noop."""
        pass

    def inc_grad(self, model_id: int, name: str, value):
        """Increment a gradient to the connection."""
        key = make_key(model_id, name)
        has_update = self._maybe_update_param(key)
        if not has_update:
            i = self._key_to_i[key]
            version = self._versions[i]
            self.conn.inc_grad.remote(
                version,
                i,
                value
            )

    def _begin_params_pull(self):
        new_time = timer()
        if (new_time - self._last_update) >= self._poll_freq:
            self.conn._ray_method_num_return_vals["get_updated_params"] = len(self._params)
            self._next_params = self.conn.get_updated_params.remote(self._last_update)
            self._last_update = new_time

    def _maybe_update_param(self, key):
        if key not in self._key_to_i:
            self._sync_params()
        i = self._key_to_i[key]
        if self._next_params[i] is not None:
            maybe_param = self.ray.get(self._next_params[i])
            if maybe_param is not None:
                version, param = maybe_param
                self._params[i] = param
                self._versions[i] = version
                self._next_params[i] = None
                return True
        return False

    def _sync_params(self):
        self._params = []
        self._versions = []
        self._next_params = []
        self._key_to_i = {}
        params = self.ray.get(self.conn.get_params.remote())
        self._last_update = timer()
        for i, (key, version, param) in enumerate(params):
            self._key_to_i[key] = i
            self._params.append(param)
            self._versions.append(version)
            self._next_params.append(None)


@dataclass
class ParamData:
    key: Tuple[int, str]
    version: int
    timestamp: Any
    value: FloatsXd
    grads: List[FloatsXd]


class SharedParams:
    """Experimental"""
    def __init__(self):
        self._params = []

    def get_params(self) -> List[Tuple[Tuple[int, str], int, FloatsXd]]:
        """Get all parameters and their versions."""
        return [(p.key, p.version, p.value) if p is not None else None for p in self._params]

    def get_updated_params(self, since: float) -> List[Optional[Tuple[int, FloatsXd]]]:
        """Return a list with params that have changed since a given timestamp,
        or None for params that have not changed since then.
        """
        updates: List[Optional[Tuple[int, FloatsXd]]] = []
        for p in self._params:
            if p.timestamp is None:
                updates.append(None)
            elif p.timestamp < since:
                updates.append(None)
            else:
                updates.append((p.version, p.value))
        return updates

    def set_param(self, key: Tuple[int, str], version: int, i: int, value: FloatsXd):
        if i == len(self._params):
            self._params.append(None)
        elif i > len(self._params):
            raise IndexError(f"Missing param? {i}, {len(self._params)}")
        self._params[i] = ParamData(
            key=key,
            version=version,
            timestamp=timer(),
            value=value,
            grads=[]
        )

    def maybe_get_grads(
        self,
        version: int,
        i: int,
        grads_required: int
    ) -> Optional[List[FloatsXd]]:
        if i >= len(self._params):
            return None
        elif self._params[i].version != version:
            return None
        elif len(self._params[i].grads) < grads_required:
            return None
        else:
            return self._params[i].grads
    
    def inc_grad(self, version:  int, i: int, value: FloatsXd) -> None:
        if self._params[i] is None:
            return
        elif self._params[i].version != version:
            return
        self._params[i].grads.append(value)
