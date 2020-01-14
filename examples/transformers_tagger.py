"""Train a transformer tagging model, using Huggingface's Transformers."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import thinc
from thinc.api import PyTorchWrapper, Softmax, chain, with_array, Model, Config
from thinc.api import torch2xp, xp2torch, sequence_categorical_crossentropy
from thinc.api import minibatch, prefer_gpu, use_pytorch_for_gpu_memory
from thinc.types import Array2d, ArgsKwargs
import ml_datasets
import tqdm
import typer


CONFIG = """
[model]
@layers = "TransformersTagger.v0"
starter = "bert-base-multilingual-cased"

[optimizer]
@optimizers = "RAdam.v1"
learn_rate = 2e-5

[optimizer.schedules.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.01
warmup_steps = 3000
total_steps = 6000

[training]
batch_size = 128
words_per_subbatch = 3000
n_epoch = 10
"""


@dataclass
class TokensPlus:
    """Dataclass to hold the output of the Huggingface 'batch_encode_plus' method."""

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_len: List[int]
    overflowing_tokens: Optional[torch.Tensor] = None
    num_truncated_tokens: Optional[torch.Tensor] = None
    special_tokens_mask: Optional[torch.Tensor] = None


@thinc.registry.layers("TransformersTagger.v0")
def TransformersTagger(
    starter: str, n_tags: int = 17
) -> Model[List[str], List[Array2d]]:
    return chain(
        TransformersTokenizer(starter),
        Transformer(starter),
        with_array(Softmax(n_tags)),
    )


@thinc.registry.layers("transformers_tokenizer.v0")
def TransformersTokenizer(name: str) -> Model[List[List[str]], List[TokensPlus]]:
    return Model(
        "tokenizer",
        _tokenizer_forward,
        attrs={"tokenizer": AutoTokenizer.from_pretrained(name)},
    )


def _tokenizer_forward(model, texts: List[List[str]], is_train: bool):
    tokenizer = model.get_attr("tokenizer")
    token_data = tokenizer.batch_encode_plus(
        [(text, None) for text in texts],
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_masks=True,
        return_input_lengths=True,
        return_tensors="pt",
    )
    return TokensPlus(**token_data), lambda d_tokens: d_tokens


@thinc.registry.layers("transformers_model.v0")
def Transformer(name) -> Model[List[TokensPlus], List[Array2d]]:
    return PyTorchWrapper(
        AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_inputs,
        convert_outputs=convert_transformer_outputs,
    )


def convert_transformer_inputs(model, tokens: TokensPlus, is_train):
    kwargs = {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "token_type_ids": tokens.token_type_ids,
    }
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def convert_transformer_outputs(model, inputs_outputs, is_train):
    layer_inputs, torch_outputs = inputs_outputs
    torch_tokvecs: torch.Tensor = torch_outputs[0]
    # Free the memory as soon as we can
    torch_outputs = None
    lengths = list(layer_inputs.input_len)
    tokvecs: List[Array2d] = model.ops.unpad(torch2xp(torch_tokvecs), lengths)
    # Remove the BOS and EOS markers.
    tokvecs = [arr[1:-1] for arr in tokvecs]

    def backprop(d_tokvecs: List[Array2d]) -> ArgsKwargs:
        # Restore entries for bos and eos markers.
        row = model.ops.alloc_f2d(1, d_tokvecs[0].shape[1])
        d_tokvecs = [model.ops.xp.vstack((row, arr, row)) for arr in d_tokvecs]
        return ArgsKwargs(
            args=(torch_tokvecs,),
            kwargs={"grad_tensors": xp2torch(model.ops.pad(d_tokvecs))},
        )

    return tokvecs, backprop


def evaluate_sequences(
    model, Xs: List[Array2d], Ys: List[Array2d], batch_size: int
) -> float:
    correct = 0.0
    total = 0.0
    for batch in minibatch(zip(Xs, Ys), size=batch_size):
        X, Y = zip(*batch)
        Yh = model.predict(X)
        for yh, y in zip(Yh, Y):
            correct += (y.argmax(axis=1) == yh.argmax(axis=1)).sum()
            total += y.shape[0]
    return correct / total


def minibatch_by_words(pairs, max_words):
    """Group pairs of sequences into minibatches under max_words in size,
    considering padding. The size of a padded batch is the length of its
    longest sequence multiplied by the number of elements in the batch.
    """
    batch = []
    for X, Y in pairs:
        batch.append((X, Y))
        n_words = max(len(xy[0]) for xy in batch) * len(batch)
        if n_words >= max_words:
            # We went *over* the cap, so don't emit the batch with this
            # example -- move that example into the next one.
            yield batch[:-1]
            batch = [(X, Y)]
    if batch:
        yield batch


def main(path: Optional[Path] = None, out_dir: Optional[Path] = None):
    if prefer_gpu():
        print("Using gpu!")
        use_pytorch_for_gpu_memory()
    # You can edit the CONFIG string within the file, or copy it out to
    # a separate file and pass in the path.
    if path is None:
        config = Config().from_str(CONFIG)
    else:
        config = Config().from_disk(path)
    # make_from_config constructs objects whenever you have blocks with an @ key.
    # In the optimizer block we write @optimizers = "Adam.v1". This tells Thinc
    # to use registry.optimizers to fetch the "Adam.v1" function. You can
    # register your own functions as well and build up trees of objects.
    C = thinc.registry.make_from_config(config)
    model = C["model"]
    optimizer = C["optimizer"]
    calculate_loss = sequence_categorical_crossentropy
    cfg = C["training"]

    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.ud_ancora_pos_tags()
    # Convert the outputs to cupy (if we're using that)
    train_Y = list(map(model.ops.asarray, train_Y))
    dev_Y = list(map(model.ops.asarray, dev_Y))
    # Pass in a small batch of data, to fill in missing shapes
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    for epoch in range(cfg["n_epoch"]):
        train_data: List[Tuple[Array2d, Array2d]] = list(zip(train_X, train_Y))
        random.shuffle(train_data)
        train_data = tqdm.tqdm(train_data, leave=False)
        # Transformers often learn best with large batch sizes -- larger than
        # fits in GPU memory. But you don't have to backprop the whole batch
        # at once. Here we consider the "logical" batch size (number of examples
        # per update) separately from the physical batch size.
        for outer_batch in minibatch(train_data, cfg["batch_size"]):
            # For the physical batch size, what we care about is the number
            # of words (considering padding too). We also want to sort by
            # length, for efficiency.
            outer_batch.sort(key=lambda xy: len(xy[0]), reverse=True)
            for batch in minibatch_by_words(outer_batch, cfg["words_per_subbatch"]):
                inputs, truths = zip(*batch)
                guesses, backprop = model(inputs, is_train=True)
                backprop(calculate_loss(guesses, truths))
            # At the end of the batch, we call the optimizer with the accumulated
            # gradients, and advance the learning rate schedules.
            model.finish_update(optimizer)
            optimizer.step_schedules()
        # You might want to evaluate more often than once per epoch; that's up
        # to you.
        score = evaluate_sequences(model, dev_X, dev_Y, 128)
        print(epoch, score)
        if out_dir:
            model.to_disk(out_dir / f"{epoch}.bin")


if __name__ == "__main__":
    typer.run(main)
