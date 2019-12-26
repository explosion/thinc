import copy
import numpy

from .neural._classes.model import Model
from .neural._classes.function_layer import FunctionLayer, wrap
from .neural._classes.feed_forward import FeedForward
from .wire import layerize, noop, chain, clone, concatenate, add
from .wire import flatten_add_lengths, unflatten
from .wire import with_reshape, with_getitem, with_square_sequences
from .wire import with_flatten, uniqued

# Deprecated

def remap_ids(ops=None, column=0):
    id_map = {0: 0}

    def remap_ids_fwd(ids, drop=0.0):
        ids = ids[:, column]
        if not isinstance(ids, numpy.ndarray):
            ids = ids.get()
        n_vector = len(id_map)
        for i, id_ in enumerate(ids):
            id_ = int(id_)
            if id_ not in id_map:
                id_map[id_] = n_vector
                n_vector += 1
            ids[i] = id_map[id_]
        return ops.asarray(ids), None

    model = layerize(remap_ids_fwd)
    if ops is None:
        ops = model.ops
    return model


def Arg(i):
    @layerize
    def arg_forward(batched_inputs, drop=0.0):
        inputs = list(zip(*batched_inputs))
        return inputs[i], None

    return begin_update


