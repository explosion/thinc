from dataclasses import dataclass
from ..types import Array


@dataclass
class Padded:
    """A batch of padded sequences, sorted by decreasing length. The data array
    is of shape (batch, step, ...). The auxiliary array size_at_t indicates the
    length of the batch at each timestep, so you can do data[:size_at_t[t]] to
    shrink the batch. 
    """
    data: Array
    size_at_t: Array
