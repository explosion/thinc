import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import thinc.api
from thinc.model import Model
from thinc.api import PyTorchWrapper, Softmax, chain, with_array, padded2list
from thinc.api import torch2xp, xp2torch, categorical_crossentropy
from thinc.types import Padded, Array1d, Array2d, Array, ArgsKwargs
from thinc.util import evaluate_model_on_arrays
import ml_datasets


CONFIG = """
[common]
starter = "distilbert-base-uncased"

[training]
batch_size = 128
n_epoch = 10

[optimizer]
@optimizers = "Adam.v1"

[model]
@layers = "transformer_tagger_example.v0"

[model.tokenizer]
@layers = "transformers_tokenizer.v0"
name = ${common:starter}

[model.transformer]
@layers = "transformers_model.v0"
name = ${common:starter}

[model.output_layer]
@layers = "output_layer_example.v0"
"""


@dataclass
class TokensPlus:
    """Dataclass to hold the output of the Huggingface 'batch_encode_plus' method."""

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_len: int
    overflowing_tokens: Optional[torch.Tensor]=None
    num_truncated_tokens: Optional[torch.Tensor]=None
    special_tokens_mask: Optional[torch.Tensor]=None


#@thinc.api.registry.layers("transformer_tagger_example.v0")
#def transformer_tagger_example(
#    tokenizer: Model[List[str], List[TokensPlus]],
#    transformer: Model[List[TokensPlus], Padded],
#    output_layer: Model[Padded, List[Array2d]],
#) -> Model[List[str], List[Array2d]]:
#    return chain(tokenizer, transformer, output_layer)
@thinc.api.registry.layers("transformer_tagger_example.v0")
def transformer_tagger_example(
    tokenizer: Model,
    transformer: Model,
    output_layer: Model,
) -> Model:
    return chain(tokenizer, transformer, output_layer)



@thinc.api.registry.layers("transformers_tokenizer.v0")
def transformers_tokenizer(name: str) -> Model[List[str], List[TokensPlus]]:
    return Model(
        "tokenizer",
        _tokenizer_forward,
        attrs={"tokenizer": AutoTokenizer.from_pretrained(name)},
    )


def _tokenizer_forward(model, texts, is_train) -> Tuple[TokensPlus, Callable]:
    tokenizer = model.get_attr("tokenizer")
    token_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_masks=True,
        return_input_lengths=True,
        return_tensors="pt"
    ) 
    return TokensPlus(**token_data), lambda d_tokens: d_tokens


@thinc.api.registry.layers("transformers_model.v0")
def transformers_model(name) -> Model[List[TokensPlus], List[Array2d]]:
    return PyTorchWrapper(
        AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_inputs,
        convert_outputs=convert_transformer_outputs
    )


@thinc.api.registry.layers("output_layer_example.v0")
def output_layer_example() -> Model[Ragged, List[Array2d]]:
    return chain(with_array(Softmax()))


def convert_transformer_input(model, tokens: List[TokensPlus], is_train):
    input_ids = [x.input_ids for x in tokens]
    attn_mask = [x.attention_mask for x in tokens]
    kwargs={
        "input_ids": [torch.tensor(x, dtype="int64") for x in input_ids],
        "attention_mask": [torch.tensor(x) for x in attn_mask]
    }
    return ArgsKwargs(args={}, kwargs=kwargs), lambda dX: []


def convert_transformer_output(model, torches, is_train):
    tokvecs = torches[0]
    # TODO: Unpad

    def backprop(d_tokvecs):
        # TODO: Repad
        args = (tokvecs,)
        kwargs = {"grad_tensors": xp2torch(d_tokvecs)}
        return ArgsKwargs(args=args, kwargs=kwargs)

    return torch2xp(tokvecs), backprop


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


def main(path: Optional[Path] = None):
    thinc.api.require_gpu()
    thinc.api.use_pytorch_for_gpu_memory()
    C = load_config(path)
    model = C["model"]
    optimizer = C["optimizer"]
    calculate_loss = categorical_crossentropy
    cfg = C["training"]

    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.ud_ancora_pos_tags()
    # Pass in a small batch of data, to fill in missing shapes.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    train_data: List[Tuple[Array2d, Array2d]]
    dev_data:   List[Tuple[Array2d, Array2d]]
    train_data = list(zip(train_X, train_Y))
    train_data = list(zip(dev_X, dev_Y))

    for epoch in range(cfg["n_epoch"]):
        random.shuffle(train_data)
        for batch in thinc.api.minibatch(train_data, cfg["batch_size"]):
            inputs, truths = zip(*batch)
            guesses, backprop = model(inputs, is_train=True)
            loss, d_guesses = calculate_loss(guesses, truths)
            backprop(d_guesses)
            model.finish_update(optimizer)
            optimizer.step_schedules()
        print(epoch, evaluate_model_on_arrays(model, dev_X, dev_Y, 128))


if __name__ == "__main__":
    main()
