from typing import Tuple, Callable, Optional, TypeVar

from .types import Array
from ..model import Model
from ..util import create_thread_local


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)

STATE = create_thread_local({"vectors": {}})


def StaticVectors(lang: str, nO: int, column: int = 0) -> Model:
    return Model(
        "static_vectors",
        forward,
        init=init,
        params={"W": None},
        attrs={"lang": lang, "column": column},
        dims={"nM": None, "nV": None, "nO": nO},
    )


def forward(
    model: Model, ids: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    column = model.get_attr("column")
    W = model.get_param("W")
    vector_table = _get_vectors(model.ops, model.get_attr("lang"))
    if ids.ndim >= 2:
        ids = model.ops.xp.ascontiguousarray(ids[:, column])
    vectors = vector_table[ids * (ids < vector_table.shape[0])]
    vectors = model.ops.xp.ascontiguousarray(vectors)
    assert vectors.shape[0] == ids.shape[0]

    def backprop(d_output: OutputType) -> InputType:
        model.inc_grad("W", model.ops.gemm(d_output, vectors, trans1=True))
        return model.ops.allocate(ids.shape, dtype=ids.dtype)

    output = model.ops.gemm(vectors, W, trans2=True)
    return output, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    vector_table = _get_vectors(model.ops, model.get_attr("lang"))
    model.set_dim("nV", vector_table.shape[0])
    model.set_dim("nM", vector_table.shape[1])
    W = model.ops.allocate((model.get_dim("nO"), model.get_dim("nM")))
    model.set_param("W", W)


def _get_vectors(ops, lang: str):
    global STATE
    key = (ops.device, lang)
    if key not in STATE.vectors:
        nlp = load_spacy(lang)
        STATE.vectors[key] = ops.asarray(nlp.vocab.vectors.data)
    return STATE.vectors[key]


def load_spacy(lang: str, **kwargs):
    import spacy

    return spacy.load(lang, **kwargs)
