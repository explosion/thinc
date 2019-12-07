import json
from thinc.config import Config


EXAMPLE_CONFIG = """
[DEFAULT]

[optimizer]
@: thinc.optimizers.Adam.v1
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.learn_rate]
@: thinc.rates.warmup_linear_rate.v1
start = 0.1
steps = 10000

[pipeline]

[pipeline.parser]
name = "parser"
factory = "parser"

[pipeline.parser.model]
@: spacy.parser.v1
hidden_depth = 1
hidden_width = 64
token_vector_width = 128

[pipeline.parser.model.tok2vec]
@: spacy.Tok2Vec.v1
width = ${pipeline.parser.model:token_vector_width}

[pipeline.parser.model.tok2vec.embed]
@: spacy.MultiFeatureHashEmbed.v1
width = ${pipeline.parser.model.tok2vec:width}

[pipeline.parser.model.tok2vec.embed.hidden]
@: thinc.v2v.MLP.v1
depth = 1
pieces = 3
layer_norm = true
outputs = ${pipeline.parser.model.tok2vec.embed:width}

[pipeline.parser.model.tok2vec.encode]
@: spacy.MaxoutWindowEncoder.v1
depth = 4
pieces = 3
window_size = 1

[pipeline.parser.model.lower]
@: spacy.ParserLower.v1

[pipeline.parser.model.upper]
@: thinc.v2v.Affine.v1
"""


def test_read_config():
    byte_string = EXAMPLE_CONFIG.encode("utf8")
    cfg = Config().from_bytes(byte_string)
    assert cfg["optimizer"]["learn_rate"]["start"] == 0.1
    assert cfg["pipeline"]["parser"]["factory"] == "parser"
    assert cfg["pipeline"]["parser"]["model"]["tok2vec"]["width"] == 128
    print(json.dumps(cfg, indent=2))
