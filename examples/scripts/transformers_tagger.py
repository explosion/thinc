from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import thinc.api
from thinc.model import Model
from thinc.api import PyTorchWrapper, Softmax, chain, with_array, padded2list
from thinc.api import torch2xp, xp2torch
from thinc.types import Padded, Array1d, Array2d, Array, ArgsKwargs
from thinc.util import evaluate_model_on_arrays
import ml_datasets


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


@dataclass
class TokensPlus:
    """Dataclass to hold the output of the Huggingface 'encode_plus' method."""

    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    overflowing_tokens: List[int]
    num_truncated_tokens: int
    special_tokens_mask: List[int]


@thinc.api.registry.layers("transformer_tagger_example.v0")
def transformer_tagger_example(
    tokenizer: Model[List[str], List[TokensPlus]],
    transformer: Model[List[TokensPlus], Padded],
    output_layer: Model[Padded, List[Array2d]],
) -> Model[List[str], List[Array2d]]:
    return chain(tokenizer, transformer, output_layer)


@thinc.api.registry.layers("transformers_tokenizer.v0")
def transformers_tokenizer(name: str) -> Model[List[str], List[TokensPlus]]:
    return Model(
        "tokenizer",
        _tokenizer_forward,
        attrs={"tokenizer": AutoTokenizer.from_pretrained(name)},
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
            return_special_tokens_mask=True,
        )
        tokens.append(TokensPlus(**info_dict))
    return tokens, lambda d_tokens: d_tokens


@thinc.api.registry.layers("transformers_model.v0")
def transformers_model(name) -> Model[List[TokensPlus], Padded]:
    return PyTorchWrapper(
        AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_input,
        convert_outputs=convert_transformer_output
    )


@thinc.api.registry.layers("output_layer_example.v0")
def output_layer_example() -> Model[Padded, List[Array2d]]:
    return chain(with_array(Softmax()), padded2list())


def convert_transformer_input(model: Model, Xs: List[TokensPlus], is_train: bool) -> Tuple[ArgsKwargs, Callable]:
    inputs = {
        "input_ids": _pad_and_convert(model.ops, [x.input_ids for x in Xs]),
        "token_type_ids": _pad_and_convert(model.ops, [x.token_type_ids for x in Xs]),
        "attention_mask": _pad_and_convert(model.ops, [x.attention_mask for x in Xs])
    }
    return ArgsKwargs(args=(), kwargs=inputs), lambda d_inputs: d_inputs


def _pad_and_convert(ops: thinc.api.Ops, X: List[List[int]], dtype="int64"):
    arrays1d: List[Array1d] = [ops.asarray(x, dtype=dtype) for x in X]
    padded = ops.list2padded([x.reshape((-1, 1)) for x in arrays1d])
    array2d = padded.data.reshape((padded.data.shape[0], padded.data.shape[1]))
    return xp2torch(array2d, requires_grad=False)


def convert_transformer_output(model: Model, Ytorch: Tuple, is_train: bool) -> Tuple[Tuple, Callable]:
    Yxp = tuple(torch2xp(y) for y in Ytorch)

    def backprop(dYxp: Tuple[Array]):
        dYtorch = tuple(xp2torch(dy, requires_grad=True) for dy in dYxp)
        return ArgsKwargs(args=((Ytorch,),), kwargs={"grad_tensors": dYtorch})

    return Yxp, backprop


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


def main(path: Optional[Path] = None):
    thinc.api.require_gpu()
    thinc.api.use_pytorch_for_gpu_memory()
    C = load_config(path)
    model = C["model"]
    optimizer = C["optimizer"]
    calculate_loss = C["loss"]
    cfg = C["training"]

    (train_X, train_Y), (dev_X, dev_Y) = load_data()

    for epoch in range(cfg["n_epoch"]):
        for inputs, truths in thinc.api.get_shuffled_batches(
            train_X, train_Y, cfg["batch_size"]
        ):
            guesses, backprop = model(inputs, is_train=True)
            loss, d_guesses = calculate_loss(guesses, truths)
            backprop(d_guesses)
            model.finish_update(optimizer)
            optimizer.step_schedules()
        print(epoch, evaluate_model_on_arrays(model, dev_X, dev_Y, 128))


if __name__ == "__main__":
    main()
