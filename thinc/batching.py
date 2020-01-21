from typing import TypeVar, Generic, Union, List, Optional, Iterable, Iterator
from typing import Sequence
from dataclasses import dataclass
from .types import Array, Array1d, Array2d, Array3d, Generator


ItemType = TypeVar("ItemType", bound=Array)
@dataclass
class Arrays(Generic[ItemType]):
    """A batch of irregularly-shaped arrays. Basically just provides
    numpy-style __getitem__. 
    """
    data: List[ItemType]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> "Arrays[ItemType]":
        if isinstance(index, int):
            return Arrays(self.data[index:index+1])
        elif isinstance(index, slice):
            return Arrays(self.data[index])
        else:
            return Arrays([self.data[i] for i in index])


_O = TypeVar("_O")
@dataclass
class Objects(Generic[_O]):
    data: List[_O]
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> "Objects[_O]":
        if isinstance(index, int):
            return Objects(self.data[index:index+1])
        elif isinstance(index, slice):
            return Objects(self.data[index])
        else:
            return Objects([self.data[i] for i in index])


@dataclass
class Padded:
    """A batch of padded sequences, sorted by decreasing length. The data array
    is of shape (step, batch, ...). The auxiliary array size_at_t indicates the
    length of the batch at each timestep, so you can do data[:, :size_at_t[t]] to
    shrink the batch. The lengths array indicates the length of each row b,
    and the indices indicates the original ordering.
    """

    data: Array3d
    size_at_t: Array1d
    lengths: Array1d
    indices: Array1d

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index) -> "Padded":
        if isinstance(index, int):
            # Slice to keep the dimensionality
            return Padded(
                self.data[:, index : index+1],
                self.lengths[index : index+1],
                self.lengths[index : index+1],
                self.indices[index : index+1]
            )
        elif isinstance(index, slice):
            return Padded(
                self.data[:, index],
                self.lengths[index],
                self.lengths[index],
                self.indices[index]
            )
        else:
            # If we get a sequence of indices, we need to be careful that
            # we maintain the length-sorting, while also keeping the mapping
            # back to the original order correct.
            sorted_index = list(sorted(index))
            return Padded(
                self.data[sorted_index],
                self.size_at_t[sorted_index],
                self.lengths[sorted_index],
                self.indices[index] # Use original, to maintain order.
            )


@dataclass
class Ragged:
    """A batch of concatenated sequences, that vary in the size of their
    first dimension. Ragged allows variable-length sequence data to be contiguous
    in memory, without padding.

    Indexing into Ragged is just like indexing into the *lengths* array, except
    it returns a Ragged object with the accompanying sequence data. For instance,
    you can write ragged[1:4] to get a Ragged object with sequences 1, 2 and 3.
    """
    data: Array2d
    lengths: Array1d
    _cumsums: Optional[Array1d]=None

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index: Union[int, slice, Array]) -> "Ragged":
        from .util import get_array_module, is_xp_array
        if isinstance(index, tuple):
            raise IndexError("Ragged arrays do not support 2d indexing.")
        starts = self._get_starts()
        ends = self._get_ends()
        if isinstance(index, int):
            s = starts[index]
            e = ends[index]
            return Ragged(self.data[s:e], self.lengths[index:index+1])
        elif isinstance(index, slice):
            lengths = self.lengths[index]
            cumsums = self._get_cumsums()
            start = cumsums[index.start-1] if index.start >= 1 else 0
            end = start + lengths.sum()
            return Ragged(self.data[start:end], lengths)
        else:
            # There must be a way to do this "properly" :(. Sigh, hate numpy.
            xp = get_array_module(self.data)
            data = xp.vstack([self[int(i)].data for i in index])
            return Ragged(data, self.lengths[index])

    def _get_cumsums(self) -> Array1d:
        if self._cumsums is None:
            self._cumsums = self.lengths.cumsum()
        return self._cumsums

    def _get_starts(self) -> Array1d:
        from .util import get_array_module
        cumsums = self._get_cumsums()
        xp = get_array_module(cumsums)
        zero = xp.array([0], dtype="i")
        return xp.concatenate((zero, cumsums[:-1]))

    def _get_ends(self) -> Array1d:
        return self._get_cumsums()



_P = TypeVar("_P", bound=Sequence)
@dataclass
class Pairs(Generic[_P]):
    one: _P
    two: _P

    def __getitem__(self, index) -> "Pairs[_P]":
        return Pairs(self.one[index], self.two[index])

    def __len__(self) -> int:
        return len(self.one)


Batchable = Union[Pairs, Ragged, Padded, Array, Objects, Arrays, List]
