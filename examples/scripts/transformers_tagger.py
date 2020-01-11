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
    """Dataclass to hold the output of the Huggingface 'encode_plus' method."""

    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    overflowing_tokens: Optional[List[int]]=None
    num_truncated_tokens: Optional[int]=None
    special_tokens_mask: Optional[List[int]]=None


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
    # TODO: Having trouble with the shape inference, because the output
    # layer is padded2list.
    n_classes = 17
    return chain(with_array(Softmax(n_classes)), padded2list())


def convert_transformer_input(model: Model, Xs: List[TokensPlus], is_train: bool) -> Tuple[ArgsKwargs, Callable]:
    inputs = {
        "input_ids": _pad_and_convert(model.ops, [x.input_ids for x in Xs]),
        "attention_mask": _pad_and_convert(model.ops, [x.attention_mask for x in Xs])
        # Not passed for distilbert
        #"token_type_ids": _pad_and_convert(model.ops, [x.token_type_ids for x in Xs]),
    }
    return ArgsKwargs(args=(), kwargs=inputs), lambda d_inputs: d_inputs


def _pad_and_convert(ops: thinc.api.Ops, X: List[List[int]], dtype="int64"):
    arrays1d: List[Array1d] = [ops.asarray(x, dtype=dtype) for x in X]
    padded = ops.list2padded([x.reshape((-1, 1)) for x in arrays1d])
    array2d = padded.data.reshape((padded.data.shape[0], padded.data.shape[1]))
    array2d = ops.asarray(array2d, dtype=dtype)
    return xp2torch(array2d, requires_grad=False)


def convert_transformer_output(model: Model, torch_outs: Tuple, is_train: bool) -> Tuple[Padded, Callable]:
    Yxp = xp2torch(torch_outs[0])
    # TODO: We need the other padding info here. Maybe should make a wrapper?
    Yp = Padded()

    def backprop(dYxp: Padded) -> ArgsKwargs:
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
