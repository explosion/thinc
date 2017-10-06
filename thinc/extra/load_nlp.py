import numpy

SPACY_MODELS = {}
VECTORS = {}


def get_spacy(lang, **kwargs):
    global SPACY_MODELS
    import spacy
    if lang not in SPACY_MODELS:
        SPACY_MODELS[lang] = spacy.load(lang, **kwargs)
    return SPACY_MODELS[lang]


def get_vectors(ops, lang):
    global VECTORS
    key = (ops.device, lang)
    if key not in VECTORS:
        nlp = get_spacy(lang)
        nV = max(lex.rank for lex in nlp.vocab)+1
        nM = nlp.vocab.vectors_length
        vectors = numpy.zeros((nV, nM), dtype='float32')
        for lex in nlp.vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector / lex.vector_norm
        VECTORS[key] = ops.asarray(vectors)
    return VECTORS[key]
