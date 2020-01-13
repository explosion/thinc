from mypy.errorcodes import ErrorCode
from mypy.options import Options
from mypy.plugin import FunctionContext, Plugin, CheckerPluginInterface
from mypy.types import Instance, Type
from mypy.nodes import Expression

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
    api = ctx.api
    l1_args, l2_args, layers_args = ctx.args
    l1_types, l2_types, layers_types = ctx.arg_types
    l1_type_instance = l1_types[0]
    l2_type_instance = l2_types[0]
    l1_arg = l1_args[0]
    l2_arg = l2_args[0]
    if not (
        isinstance(l1_type_instance, Instance)
        and isinstance(l2_type_instance, Instance)
        and isinstance(ctx.default_return_type, Instance)
    ):
        return ctx.default_return_type
    if (
        l1_type_instance.type.fullname != thinc_model_fullname
        or l2_type_instance.type.fullname != thinc_model_fullname
    ):
        return ctx.default_return_type
    arg_in_type = l1_type_instance.args[0]
    arg_out_type = l2_type_instance.args[1]
    chain_check_2_layers(
        l1_arg=l1_arg,
        l1_type=l1_type_instance,
        l2_arg=l2_arg,
        l2_type=l2_type_instance,
        api=api,
    )
    last_arg = l2_arg
    last_type = l2_type_instance
    for arg, type_ in zip(layers_args, layers_types):
        if not isinstance(type_, Instance):
            continue
        chain_check_2_layers(
            l1_arg=last_arg, l1_type=last_type, l2_arg=arg, l2_type=type_, api=api
        )
        last_arg = arg
        last_type = type_
        arg_out_type = type_.args[1]
    return Instance(ctx.default_return_type.type, [arg_in_type, arg_out_type])


def chain_check_2_layers(
    *,
    l1_arg: Expression,
    l1_type: Instance,
    l2_arg: Expression,
    l2_type: Instance,
    api: CheckerPluginInterface
):
    if l1_type.args[1] != l2_type.args[0]:
        api.fail(
            "Layer mismatch, output not compatible with next layer",
            l1_arg,
            code=error_layer_output,
        )
        api.fail(
            "Layer mismatch, input not compatible with previous layer",
            l2_arg,
            code=error_layer_input,
        )


error_layer_input = ErrorCode("layer-mismatch-input", "Invalid layer input", "Thinc")
error_layer_output = ErrorCode("layer-mismatch-output", "Invalid layer output", "Thinc")
