from dataclasses import dataclass
from ..types import Array


@dataclass
class Ragged:
    data: Array
    lengths: Array
