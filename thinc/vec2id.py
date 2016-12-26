from .base import Network
from .vec2vec import ReLu
from .vec2vec import Softmax


class MLP(Network):
    Hidden = ReLu
    Output = Softmax
    depth = 2
    nr_out = 2
