from typing import Dict, Optional, Tuple, Any

from ..types import NlpType, Array


SPACY_MODELS: Dict[str, NlpType] = {}
VECTORS: Dict[Tuple[Any, str], Optional[Array]] = {}


def get_spacy(lang: str, **kwargs) -> NlpType:
    global SPACY_MODELS
    import spacy

    if lang not in SPACY_MODELS:
        SPACY_MODELS[lang] = spacy.load(lang, **kwargs)
    return SPACY_MODELS[lang]


def register_vectors(ops, lang: str, data: Optional[Array]) -> None:  # TODO: ops type
    key = (ops.device, lang)
    VECTORS[key] = data


def get_vectors(ops, lang: str) -> Optional[Array]:  # TODO: ops type
    global VECTORS
    key = (ops.device, lang)
    if key not in VECTORS:
        nlp = get_spacy(lang)
        VECTORS[key] = nlp.vocab.vectors.data
    return VECTORS[key]
