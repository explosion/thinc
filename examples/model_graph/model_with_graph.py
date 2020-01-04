from typing import List, Dict
from thinc.layers import (
    chain,
    ReLu,
    Softmax,
    Affine,
    Mish,
    ExtractWindow,
    Maxout,
    concatenate,
)
from thinc.optimizers import Adam
from thinc.model import Model
from thinc.util import get_shuffled_batches
import ml_datasets
import tqdm

try:
    import pydot
except ImportError:
    raise ValueError("pydot and svgwrite are required.\n\tpip install pydot svgwrite")


CONFIG = """
[hyper_params]
n_hidden = 512
dropout = 0.2

[model]
@layers = "chain.v1"

[model.layers.relu1]
@layers = "ReLu.v1"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.layers.relu2]
@layers = "ReLu.v1"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.layers.softmax]
@layers = "Softmax.v1"

[optimizer]
@optimizers = "Adam.v1"
learn_rate = ${hyper_params:learn_rate}
"""


def load_mnist():
    from thinc.backends import NumpyOps
    from thinc.util import to_categorical

    ops = NumpyOps()
    mnist_train, mnist_dev, _ = ml_datasets.mnist()
    train_X, train_Y = ops.unzip(mnist_train)
    train_Y = to_categorical(train_Y, nb_classes=10)
    dev_X, dev_Y = ops.unzip(mnist_dev)
    dev_Y = to_categorical(dev_Y, nb_classes=10)
    return (train_X, train_Y), (dev_X, dev_Y)


def main(n_hidden=32, dropout=0.2, n_iter=10, batch_size=128):
    # Define the model
    model = chain(
        ExtractWindow(3),
        ReLu(n_hidden, dropout=dropout, normalize=True),
        Maxout(n_hidden * 4),
        Affine(n_hidden * 2),
        ReLu(n_hidden, dropout=dropout, normalize=True),
        Affine(n_hidden),
        ReLu(n_hidden, dropout=dropout),
        Softmax(),
    )

    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = load_mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    dot = model_to_dot(
        model,
        show_shapes=True,
        show_classes=False,
        show_layer_names=True,
        rankdir="TR",
        dpi=64,
    )
    output = dot.create_svg().decode("utf-8")
    with open("model.svg", "w") as f:
        f.write(output)


def model_to_dot(
    model: Model,
    show_shapes: bool = False,
    show_layer_names: bool = True,
    show_classes: bool = False,
    rankdir: str = "TB",
    dpi: int = 96,
) -> pydot.Dot:
    dot = pydot.Dot()
    dot.set("rankdir", rankdir)
    dot.set("concentrate", True)
    dot.set("dpi", dpi)
    dot.set_node_defaults(shape="record")
    if not isinstance(model, Model):
        raise ValueError("can only graph thinc.model.Model instances")
    nodes: Dict[int, pydot.Node] = {}
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        class_name = layer.__class__.__name__
        # Create node's label.
        label = ""
        if show_layer_names:
            label += layer_name
        if show_classes:
            label = f"{class_name}|{label}"
        if show_shapes:
            output_shape = layer._dims.get("nO", None)
            input_shape = layer._dims.get("nI", None)
            in_label = f"{'?' if input_shape is None else input_shape}"
            out_label = f"{'?' if output_shape is None else output_shape}"
            # nodes_in: InputLayer\n|{input:|output:}|{{[(?, ?)]}|{[(?, ?)]}}
            label = "{%s|(%s, %s)}" % (label, in_label, out_label,)
        node = pydot.Node(layer.id, label=label)
        dot.add_node(node)
        nodes[layer.id] = node

    for i, layer in enumerate(model.layers):
        if i == 0:
            continue
        from_node: pydot.Node = nodes[model.layers[i - 1].id]
        to_node: pydot.Node = nodes[layer.id]
        if not dot.get_edge(from_node, to_node):
            dot.add_edge(pydot.Edge(from_node, to_node))
    return dot


if __name__ == "__main__":
    import plac

    plac.call(main)

