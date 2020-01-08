import typer
from typing import *
from pathlib import Path
import thinc.api


CONFIG = """
[common]
starter = "albert"

[training]
batch_size = 128
n_epoch = 10

[model]
@layers = "output_layer_example.v0"

[model.tokenizer]
@layers = "transformers_tokenizer.v0"
name = ${common:starter}

[model.transformer]
@layers = "transformers_model.v0"
name = ${common:starter}

[model.output_layer]
@layers = "output_layer.v0"

"""

@thinc.api.registry.layers("transformers_tokenizer.v0")
def transformers_tokenizer(name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    return wrap_transformers_tokenizer(tokenizer)


@thinc.api.registry.layers("transformers_model.v0")
def transformers_model(name):
    from transformers import AutoModel

    transformer = AutoModel.from_pretrained(name)
    return wrap_transformers_model(transformer)


def wrap_transformers_model(model):
    raise NotImplementedError


def wrap_transformers_tokenizer(tokenizer):
    raise NotImplementedError


@thinc.api.registry.layers("output_layer_example.v0")
def output_layer_example():
    return Softmax()


@thinc.api.registry.layers("transformer_tagger_example.v0")
def transformer_tagger_example(tokenizer, transformer, output_layer):
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


def main(path: Path=None):
    thinc.api.require_gpu()
    thinc.api.use_pytorch_for_gpu_memory()
    C = load_config(path)
    model = C["model"]
    optimizer = C["optimizer"]
    calculate_loss = C["loss"]
    cfg = C["training"]
    
    (train_X, train_Y), (dev_X, dev_Y) = load_data()

    for epoch in range(cfg["n_epoch"]):
        for inputs, truths in get_shuffled_batches(train_X, train_Y, cfg["batch_size"]):
            guesses, backprop = model(inputs, is_train=True)
            loss, d_guesses = calculate_loss(guesses, truths)
            backprop(d_guesses)
            model.finish_update(optimizer)
            optimizer.step_schedules()
        print(epoch, evaluate(model, dev_X, dev_Y))


if __name__ == "__main__":
    typer.run(main)
