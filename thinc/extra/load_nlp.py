import spacy

SPACY_MODELS = {}

def get_spacy(lang, parser=False, tagger=False, entity=False):
    global SPACY_MODELS
    if spacy is None:
        raise ImportError("Could not import spacy. Is it installed?")
    if lang not in SPACY_MODELS:
        SPACY_MODELS[lang] = spacy.load(
            lang, parser=parser, tagger=tagger, entity=entity)
    return SPACY_MODELS[lang]
