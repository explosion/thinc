# coding: utf8
from __future__ import unicode_literals


SPACY_MODELS = {}
VECTORS = {}


def get_spacy(lang, **kwargs):
    global SPACY_MODELS
    import spacy

    if lang not in SPACY_MODELS:
        SPACY_MODELS[lang] = spacy.load(lang, **kwargs)
    return SPACY_MODELS[lang]

def register_vectors(ops, lang, data):
    key = (ops.device, lang)
    VECTORS[key] = data


def get_vectors(ops, lang):
    global VECTORS
    key = (ops.device, lang)
    if key not in VECTORS:
        nlp = get_spacy(lang)
        VECTORS[key] = nlp.vocab.vectors.data
    return VECTORS[key]
