from typing import Dict, Tuple, Any
from timeit import default_timer as timer
from dataclasses import dataclass
from collections import Counter
import time
import threading
from collections import UserDict
from ..types import FloatsXd


KeyT = Tuple[int, str]


def make_key(model_id: int, name: str) -> Tuple[int, str]:
    return (model_id, name)

class Timer:
    def __init__(self, state):
        self.state = state
        self.sum = 0
        self.n = 0

    def __enter__(self):
        self.start = time.time()
        self.n += 1

    def __exit__(self, *args):
        interval = time.time() - self.start
        self.sum += interval
        print(f"{self.state}: {self.sum / self.n:0.4f}")


class ManyTimer:
    def __init__(self):
        self.timers = {}

    def __call__(self, key):
        if key not in self.timers:
            self.timers[key] = Timer(key)
        return self.timers[key]

def thread_function(fetch_event, next_params, ray_, conn):
    _last_update = 0
    while True:
        fetch_event.wait()
        updates = ray_.get(
            conn.get_updated_params.remote(_last_update)
        )
        new_time = timer()
        _last_update = new_time
        next_params.update(updates)
        print('fetched')
        fetch_event.clear()

class RayProxy:
    """Proxy for the 'head' worker that owns the optimizer and pushes
    parameter updates.
    """
    ray: Any
    conn: Any
    _params: Dict
    _next_params: Dict
    _versions: Dict

    def __init__(self, connection, *, ray=None, poll_freq=0.1, use_thread=False):
        if ray is None:
            import ray # type: ignore
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self._last_update = 0
        self._poll_freq = poll_freq
        self._params = {}
        self._versions = {}
        self._next_params = {}
        self.timers = ManyTimer()

        self.use_thread = use_thread
        if self.use_thread:
            print('starting thread')
            self.fetch_event = threading.Event()
            self.thread = threading.Thread(target=thread_function, args=(
                self.fetch_event, self._next_params, self.ray, self.conn), daemon=True)
            self.thread.start()
            print('threadstarted')

    def set_param(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a parameter to the connection."""
        key = make_key(model_id, name)
        self._params[key] = value
        self._next_params[key] = None
        self._versions[key] = self.ray.get(
            self.conn.set_param.remote(
                key,
                value
            )
        )

    def get_param(self, model_id: int, name: str) -> FloatsXd:
        """Get a parameter from the connection."""
        key = make_key(model_id, name)
        self._maybe_update_param(key)
        return self._params[key]

    def set_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a gradient to the connection."""
        key = make_key(model_id, name)
        self.conn.set_grad.remote(
            key,
            self._versions[key],
            value
        )

    def inc_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Increment a gradient to the connection."""
        key = make_key(model_id, name)
        # with self.timers(f"inc_grad:"):
        self._begin_params_pull()
        self.conn.inc_grad.remote(
            key,
            self._versions[key],
            value
        )

    def _begin_params_pull(self):
        new_time = timer()
        if self.use_thread:
            self.fetch_event.set()
        else:
            if (new_time - self._last_update) >= self._poll_freq:
                updates = self.ray.get(
                    self.conn.get_updated_params.remote(self._last_update)
                )
                self._last_update = new_time
                self._next_params.update(updates)

    def _maybe_update_param(self, key):
        if self._next_params.get(key) is None:
            return False
        else:
            version, param = self._next_params.pop(key)
            self._params[key] = param
            self._versions[key] = version
            return True

    def _sync_params(self):
        self._params = {}
        self._versions = {}
        self._next_params = {}
        params = self.ray.get(self.conn.get_updated_params.remote(0))
        self._last_update = timer()
        for key, (version, param) in params.items():
            self._params[key] = param
            self._versions[key] = version


ObjectID = int

@dataclass
class ParamData:
    key: Tuple[int, str]
    version: int
    timestamp: Any
    value: FloatsXd
    grads: FloatsXd
    grad_count: int


class SharedOptimizer:
    """Provide access to an optimizer for multiple workers. Designed to be
    used as a ray remote actor, connected to a ParamServer via RayProxy.
    """
    def __init__(self, optimizer, quorum, ray=None):
        if ray is None:
            import ray
        self.ray = ray
        self.quorum = quorum
        self.optimizer = optimizer
        self._params = {}
        self._progress = Counter()
        self._n_updates = 0

    def get_quorum(self):
        return self.quorum

    def inc_progress(self, worker_id):
        self._progress[worker_id] += 1

    def get_progress(self):
        return self._progress

    def get_total_progress(self):
        return sum(self._progress.values())

    def step_schedules(self):
        self.optimizer.step_schedules()

    def get_transaction_id(self, key):
        return self._params[key].version

    def get_param(self, key):
        return (self._params[key].version, self._params[key].value)

    def get_updated_params(self, since: float) -> Dict:
        """Return a dict with params that have changed since a given timestamp.
        """
        updates = {}
        for key, p in self._params.items():
            if p.timestamp >= since:
                updates[key] = (p.version, p.value)
        return updates

    def set_param(self, key, value):
        if key in self._params:
            version = self._params[key].version + 1
        else:
            version = 0
        self._params[key] = ParamData(
            key=key,
            value=value,
            version=version,
            grads=None,
            grad_count=0,
            timestamp=timer()
        )
        return self._params[key].version

    def set_grad(self, tid, key, value):
        if key not in self._params:
            return None
        elif tid != self._params[key].version:
            # If we've moved past this version, discard the gradient.
            return None
        else:
            self._params[key].grads = value.copy()
            self._params[key].grad_count = 1
            self._update_if_quorum(key)

    def inc_grad(self, key, tid, value):
        if key not in self._params:
            return None
        elif tid != self._params[key].version:
            return None
        elif self._params[key].grads is None:
            self._params[key].grads = value.copy()
            self._params[key].grad_count = 1
            self._update_if_quorum(key)
        else:
            self._params[key].grads += value
            self._params[key].grad_count += 1
            self._update_if_quorum(key)

    def _update_if_quorum(self, key):
        if key not in self._params:
            return
        if self._params[key].grad_count >= self.quorum:
            params, _ = self.optimizer(
                key,
                self._params[key].value.copy(),
                self._params[key].grads
            )
            self._params[key].value = params
            self._params[key].grads = None
            self._params[key].grad_count = 0
            self._params[key].version += 1
            self._params[key].timestamp = timer()
            self._n_updates += 1
