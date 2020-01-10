from mypy.checker import TypeChecker
from mypy.errorcodes import ErrorCode
from mypy.options import Options
from mypy.plugin import FunctionContext, Plugin
from mypy.types import Instance, Type

thinc_model_fullname = "thinc.model.Model"


def plugin(version: str):
    return ThincPlugin


class ThincPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_function_hook(self, fullname: str):
        if fullname == "thinc.layers.chain.chain":
            return chain_callback
        return None


def chain_callback(ctx: FunctionContext) -> Type:
    api: TypeChecker = ctx.api
    layer1_args, layer2_args, layers_args = ctx.args
    layer1_types, layer2_types, layers_types = ctx.arg_types
    if (
        layer1_types[0].type.fullname != thinc_model_fullname
        or layer2_types[0].type.fullname != thinc_model_fullname
    ):
        return ctx.default_return_type
    arg_in_type = layer1_types[0].args[0]
    arg_out_type = layer2_types[0].args[1]
    chain_check_2_layers(
        l1_arg=layer1_args[0],
        l1_type=layer1_types[0],
        l2_arg=layer2_args[0],
        l2_type=layer2_types[0],
        api=api,
    )
    last_arg = layer2_args[0]
    last_type = layer2_types[0]
    for arg, type_ in zip(layers_args, layers_types):
        chain_check_2_layers(
            l1_arg=last_arg, l1_type=last_type, l2_arg=arg, l2_type=type_, api=api
        )
        last_arg = arg
        last_type = type_
        arg_out_type = type_.args[1]
    return Instance(ctx.default_return_type.type, [arg_in_type, arg_out_type])


def chain_check_2_layers(
    *,
    l1_arg: Instance,
    l1_type: Instance,
    l2_arg: Instance,
    l2_type: Instance,
    api: TypeChecker
):
    if l1_type.args[1] != l2_type.args[0]:
        api.fail(
            "Layer mismatch, output not compatible with next layer",
            context=l1_arg,
            code=error_layer_output,
        )
        api.fail(
            "Layer mismatch, input not compatible with previous layer",
            context=l2_arg,
            code=error_layer_input,
        )


error_layer_input = ErrorCode("layer-mismatch-input", "Invalid layer input", "Thinc")
error_layer_output = ErrorCode("layer-mismatch-output", "Invalid layer output", "Thinc")
