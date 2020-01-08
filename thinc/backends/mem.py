from typing import Dict, Optional, Tuple
from numpy import prod
from ..types import Array, Shape
from ..util import is_jax_array, copy_array
from .ops import Ops

try:
    import jax.ops
except ImportError:
    pass


class Memory:
    """Serve parameters for a single process."""

    ops: Ops
    _mem: Array
    _offsets: Dict[Tuple[int, str], Tuple[int, int, Shape]]
    _sizes: Dict[Tuple[int, str], int]
    _i: int

    def __init__(self, ops, size=128):
        if size < 0:
            raise ValueError(f"TODO error re negative size {size}")
        self.ops = ops
        self._mem = self.ops.alloc_f2d(2, size)
        self._offsets = {}
        self._sizes = {}
        self._i = 0

    @property
    def weights(self) -> Array:
        return self._mem[0]

    @weights.setter
    def weights(self, value):
        jax.ops.index_update(self._mem, jax.ops.index[0, :value.size], value)
        #self._mem[0, :value.size] = value

    @property
    def gradient(self) -> Array:
        return self._mem[1]

    @gradient.setter
    def gradient(self, value):
        jax.ops.index_update(self._mem, jax.ops.index[1, :value.size], value)
        #self._mem[1, :value.size] = value.size

    def __contains__(self, name: Tuple[int, str]) -> bool:
        return name in self._offsets

    def __getitem__(self, name: Tuple[int, str]) -> Array:
        offset, col, shape = self._offsets[name]
        size = self._sizes[name]
        return self._mem[col, offset : offset + size].reshape(shape)

    def __setitem__(self, name: Tuple[int, str], value) -> None:
        data = self[name]
        copy_array(dst=data, src=value)

    def get(
        self, name: Tuple[int, str], default: Optional[Array] = None
    ) -> Optional[Array]:
        return self[name] if name in self._offsets else default


    def add(self, name: Tuple[int, str], shape: Shape):
        assert name not in self._offsets, "TODO: error"
        self._offsets[name] = (self._i, 0, shape)
        size: int = prod(shape)
        self._sizes[name] = size
        blob = self._get_blob(size)
        return blob[0].reshape(shape)

    def add_gradient(
        self, grad_name: Tuple[int, str], param_name: Tuple[int, str]
    ) -> Array:
        assert grad_name not in self._offsets, "TODO: error"
        offset, _, shape = self._offsets[param_name]
        size = self._sizes[param_name]
        self._offsets[grad_name] = (offset, 1, shape)
        self._sizes[grad_name] = size
        return self._mem[1, offset : offset + size].reshape(shape)

    def _get_blob(self, nr_req: int) -> Array:
        nr_avail = self._mem.shape[1] - (self._i + 1)
        if nr_avail < nr_req:
            self._realloc(max(self._mem.shape[1], nr_req) * 2)
        blob = self._mem[:, self._i : self._i + nr_req]
        self._i += nr_req
        return blob

    def _realloc(self, new_size: int):
        new_mem = self.ops.alloc_f2d(self._mem.shape[0], new_size)
        jax.ops.index_update(new_mem, jax.ops.index[:, :self._i+1], self._mem[:, : self._i + 1])
        self._mem = new_mem
        #print("New mem", type(new_mem), new_mem.shape)
        #if is_jax_array(new_mem):
        #    index = jax.ops.index[:, :self._i + 1]
        #    self._mem = jax.ops.index_update(new_mem, index, self._mem[:, :self._i+1])
        #else:
