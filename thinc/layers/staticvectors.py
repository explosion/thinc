from typing import Tuple, Callable, Optional

from ..types import Array, Array2d
from ..model import Model
from ..backends import Ops
from ..config import registry
from ..util import create_thread_local


InT = Array2d
OutT = Array2d

STATE = create_thread_local({"vectors": {}})


@registry.layers("StaticVectors.v0")
def StaticVectors(lang: str, nO: int, *, column: int = 0) -> Model[InT, OutT]:
    return Model(
        "static_vectors",
        forward,
        init=init,
        params={"W": None},
        attrs={"lang": lang, "column": column},
        dims={"nM": None, "nV": None, "nO": nO},
    )


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    column = model.get_attr("column")
    W = model.get_param("W")
    vector_table = _get_vectors(model.ops, model.get_attr("lang"))
    if ids.ndim >= 2:
        ids = model.ops.as_contig(ids[:, column])
    vectors = vector_table[ids * (ids < vector_table.shape[0])]
    vectors = model.ops.as_contig(vectors)
    assert vectors.shape[0] == ids.shape[0]

    def backprop(d_output: OutT) -> InT:
        model.inc_grad("W", model.ops.gemm(d_output, vectors, trans1=True))
        return model.ops.alloc(ids.shape, dtype=ids.dtype)

    output = model.ops.gemm(vectors, W, trans2=True)
    return output, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    vector_table = _get_vectors(model.ops, model.get_attr("lang"))
    model.set_dim("nV", vector_table.shape[0])
    model.set_dim("nM", vector_table.shape[1])
    W = model.ops.alloc_f2d(model.get_dim("nO"), model.get_dim("nM"))
    model.set_param("W", W)
    return model


def _get_vectors(ops: Ops, lang: str) -> Array:
    global STATE
    key = (ops.device_type, lang)
    if key not in STATE.vectors:
        nlp = load_spacy(lang)
        STATE.vectors[key] = ops.asarray(nlp.vocab.vectors.data)
    return STATE.vectors[key]


def load_spacy(lang: str, **kwargs):
    import spacy

    return spacy.load(lang, **kwargs)
