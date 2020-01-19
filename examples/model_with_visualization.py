# pip install thinc ml_datasets typer pydot svgwrite
from typing import Dict, Union, Optional
from pathlib import Path
from thinc.api import chain, ReLu, Softmax, Linear, expand_window, Maxout, Model
import ml_datasets
import typer
import pydot


def pydot_visualizer(
    model: Model,
    *,
    show_shapes: bool = True,
    show_layer_names: bool = True,
    show_classes: bool = False,
    rankdir: str = "LR",
    fontname: str = "arial",
    fontsize: str = "10",
    output: Optional[Union[Path, str]] = None,
    file_format: str = "svg",
):
    """Convert a Thinc model to a PyDot / Graphviz visualization. Requires
    pydot, svgwrite and GraphViz (via apt-get, brew etc.) to be installed.
    """
    dot = pydot.Dot()
    dot.set("rankdir", rankdir)
    dot.set("concentrate", True)
    dot.set_node_defaults(shape="record", fontname=fontname, fontsize=fontsize)
    dot.set_edge_defaults(arrowsize="0.7")
    nodes: Dict[int, pydot.Node] = {}
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        class_name = layer.__class__.__name__
        label = ""
        if show_layer_names:
            label += layer_name
        if show_classes:
            label = f"{class_name}|{label}"
        if show_shapes:
            output_shape = layer.get_dim("nO") if layer.has_dim("nO") else None
            input_shape = layer.get_dim("nI") if layer.has_dim("nI") else None
            in_label = f"{'?' if input_shape is None else input_shape}"
            out_label = f"{'?' if output_shape is None else output_shape}"
            # nodes_in: InputLayer\n|{input:|output:}|{{[(?, ?)]}|{[(?, ?)]}}
            label = f"{label}|({in_label}, {out_label})"
        # Hack to work around "bad label name" problem
        label = label.replace(">", "&gt;")
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
    if output is not None:
        dot.write(output, format=file_format)
    return dot


def main(
    n_hidden: int = 32,
    dropout: float = 0.2,
    n_iter: int = 10,
    batch_size: int = 128,
    output: str = "model.svg",
    file_format: str = "svg",
):
    # Define the model
    model: Model = chain(
        expand_window(3),
        ReLu(nO=n_hidden, dropout=dropout, normalize=True),
        Maxout(nO=n_hidden * 4),
        Linear(nO=n_hidden * 2),
        ReLu(nO=n_hidden, dropout=dropout, normalize=True),
        Linear(nO=n_hidden),
        ReLu(nO=n_hidden, dropout=dropout),
        Softmax(),
    )
    # Load the data
    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
    # Set any missing shapes for the model.
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    pydot_visualizer(model, output=output, file_format=file_format)


if __name__ == "__main__":
    typer.run(main)
