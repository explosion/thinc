from typing import Dict, Union, Optional
from pathlib import Path

from ..model import Model

try:
    import pydot

    has_pydot = True
except ImportError:
    has_pydot = False


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
) -> "pydot.Dot":
    """Convert a Thinc model to a PyDot / Graphviz visualization. Requires
    GraphViz and PyDot to be installed.
    """
    if not has_pydot:
        raise ValueError(
            "pydot and svgwrite are required: pip install pydot svgwrite\n"
            "Also make sure you have GraphViz installed (via apt-get, brew etc.)"
        )
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
