from dataclasses import dataclass
from typing import *
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import thinc.api
from thinc.model import Model
from thinc.api import PyTorchWrapper, ragged2list, Softmax, chain
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
    def _forward(model, X, is_train):
        seqs = X.unpad(X.data)
        ragged = Ragged(
            model.ops.flatten(seqs)
            model.ops.astype([len(seq) for seq in seqs], dtype="i")
        )

        def backprop(d_ragged):
            d_seqs = model.ops.unflatten(d_ragged.data, d_ragged.lengths)
            return Padded(*model.ops.square_sequences(d_seqs))

        return ragged, backprop

    return Model("padded2ragged", _forward)


def with_ragged2array(layer: Model[Array, Array]) -> Model[Ragged, Ragged]:
    def _forward(model, Xr: Ragged, is_train):
        Y, backprop_layer = model.layers[0](Xr.data, is_train)

        def backprop(dYr):
            return Ragged(backprop_layer(dYr.data), dYr.lengths)
        
        return Ragged(Y, Xr.lengths)

    return Model("with_ragged2array", _forward, layers=[layer])


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
        with_ragged2array(Softmax()),
        ragged2list()
    )


@thinc.api.registry.layers("transformer_tagger_example.v0")
def transformer_tagger_example(
    tokenizer: Model[List[str], List[TokensPlus]],
    transformer: Model[List[Array], Padded],
    output_layer: Model[Padded, List[Floats2d]]
) -> Model[List[str], List[Floats2d]]:
    return chain(
        tokenizer,
        transformer,
        output_layer
    )


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
