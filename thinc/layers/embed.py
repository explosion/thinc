from .base import Model
from .. import util
from ..initializers import uniform_init


def Embed(nO=None, nV=None, column=0, initializer=uniform_init):
    return Model(
        "embed",
        forward,
        init=create_init(initializer),
        dims={"nO": nO, "nV": nV},
        attrs={"column": column},
        layers=[],
        params={"vectors": None},
    )


def create_init(initializer):
    def init(model, X=None, Y=None):
        if Y is not None:
            model.set_dim(util.get_width(Y))
        shape = (model.get_dim("nV"), model.get_dim("nO"))
        vectors = initializer(model.ops.allocate(shape))
        model.set_param("vectors", vectors)

    return init


def forward(model, ids, is_train):
    nV = model.get_dim("nV")
    vectors = model.get_param("vectors")
    column = model.get_attr("column")
    if ids.ndim == 2:
        ids = ids[:, column]
    ids[ids >= nV] = 0
    output = vectors[ids]

    def backprop_embed(d_output):
        d_vectors = model.ops.allocate(vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("vectors", d_vectors)
        return model.ops.allocate(ids.shape, dtype=ids.dtype)

    return output, backprop_embed
