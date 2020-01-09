from dataclasses import dataclass
from typing import *
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import thinc.api
from thinc.model import Model
from thinc.api import chain
from thinc.layers.pytorchwrapper import PyTorchWrapper
from thinc.layers.ragged2list import ragged2list
from thinc.layers.softmax import Softmax
from thinc.layers.chain import chain
from thinc.types import Padded, Ragged, Floats2d, Array
from thinc.util import evaluate_model_on_arrays


CONFIG = """
[common]
starter = "albert-base-v2"

[training]
batch_size = 128
n_epoch = 10

[model]

[model.tokenizer]
@layers = "transformers_tokenizer.v0"
name = ${common:starter}

[model.transformer]
@layers = "transformers_model.v0"
name = ${common:starter}

[model.output_layer]
@layers = "output_layer.v0"
"""

def padded2ragged() -> Model[Padded, Ragged]:
    ...


def with_array(layer: Model[Array, Array]) -> Model:
    ...


@dataclass
class TokensPlus:
    """Dataclass to hold the output of the Huggingface 'encode_plus' method."""
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    overflowing_tokens: List[int]
    num_truncated_tokens: int
    special_tokens_mask: List[int]


@thinc.api.registry.layers("transformers_tokenizer.v0")
def transformers_tokenizer(name: str) -> Model[List[str], List[TokensPlus]]:
    return Model(
        "tokenizer",
        _tokenizer_forward,
        attrs={"tokenizer":  AutoTokenizer.from_pretrained(name)}
    )


def _tokenizer_forward(model, texts, is_train):
    tokenizer = model.get_attr("tokenizer")
    tokens = []
    for text in texts:
        info_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True
        )
        tokens.append(TokensPlus(**info_dict))
    return tokens, lambda d_tokens: d_tokens


@thinc.api.registry.layers("transformers_model.v0")
def transformers_model(name) -> Model[List[TokensPlus], Padded]:
    transformer = AutoModel.from_pretrained(name)
    return PyTorchWrapper(transformer)



@thinc.api.registry.layers("output_layer_example.v0")
def output_layer_example() -> Model[Padded, List[Floats2d]]:
    return chain(
        padded2ragged(),
        with_array(Softmax()),
        ragged2list()
    )


@thinc.api.registry.layers("transformer_tagger_example.v0")
def transformer_tagger_example(
    tokenizer: Model[List[str], List[TokensPlus]],
    transformer: Model[List[Array], Padded],
    output_layer: Model[Padded, List[Floats2d]]
) -> Model[List[str], List[Floats2d]]:
    model = thinc.layers.chain.chain(
        tokenizer,
        transformer,
        output_layer
    )
    print(reveal_type(model))
    reveal_locals()
    return model

_dummy_tokenizer = cast(Model[List[str], List[TokensPlus]], None)
_dummy_transformer = cast(Model[List[Array], Padded], None)
_dummy_output = cast(Model[Padded, List[Floats2d]], None)
_dummy_model = transformer_tagger_example(_dummy_tokenizer, _dummy_transformer, _dummy_output)

def load_config(path: Optional[Path]):
    from thinc.api import Config, registry
    if path is None:
        config = Config().from_str(CONFIG)
    else:
        config = Config().from_disk(path)
    # The make_from_config function constructs objects for you, whenever
    # you have blocks with an @ key. For instance, in the optimizer block,
    # we write @optimizers = "Adam.v1". This tells Thinc to use the optimizers
    # registry to fetch the "Adam.v1" function. You can register your own
    # functions as well, and build up trees of objects.
    return registry.make_from_config(config)
 

def load_data():
    from thinc.api import to_categorical
    train_data, check_data, nr_class = ml_datasets.ud_ancora_pos_tags()
    train_X, train_y = zip(*train_data)
    dev_X, dev_y = zip(*check_data)
    nb_tag = max(max(y) for y in train_y) + 1
    train_y = [to_categorical(y, nb_tag) for y in train_y]
    dev_y = [to_categorical(y, nb_tag) for y in dev_y]
    return (train_X, train_y), (dev_X, dev_y)


def main(path: Optional[Path]=None):
    thinc.api.require_gpu()
    thinc.api.use_pytorch_for_gpu_memory()
    C = load_config(path)
    model = C["model"]
    optimizer = C["optimizer"]
    calculate_loss = C["loss"]
    cfg = C["training"]
    
    (train_X, train_Y), (dev_X, dev_Y) = load_data()

    for epoch in range(cfg["n_epoch"]):
        for inputs, truths in thinc.api.get_shuffled_batches(train_X, train_Y, cfg["batch_size"]):
            guesses, backprop = model(inputs, is_train=True)
            loss, d_guesses = calculate_loss(guesses, truths)
            backprop(d_guesses)
            model.finish_update(optimizer)
            optimizer.step_schedules()
        print(epoch, evaluate_model_on_arrays(model, dev_X, dev_Y, 128))


if __name__ == "__main__":
    main()
