from mypy.checker import TypeChecker
from mypy.errorcodes import ErrorCode
from mypy.options import Options
from mypy.plugin import FunctionContext, Plugin
from mypy.types import Instance, Type

try:
    pass
except ImportError:
    pass


def plugin(version: str):
    return ThincPlugin


class ThincPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_function_hook(self, fullname: str):
        if fullname == "thinc.layers.chain.chain":
            return call_callback
        return None


def call_callback(ctx: FunctionContext) -> Type:
    api: TypeChecker = ctx.api
    previous_arg_type = ctx.arg_types[0]
    first_arg_type = previous_arg_type
    previous_arg = ctx.args[0]
    # mypy passes several times, in intermediate passes, the node is a tuple
    if previous_arg_type[0].type.fullname != "thinc.model.Model":
        return ctx.default_return_type
    for arg, arg_type in zip(ctx.args[1:], ctx.arg_types[1:]):
        # mypy passes several times, in intermediate passes, the last arg is empty
        if not arg_type or arg_type[0].type.fullname != "thinc.model.Model":
            return ctx.default_return_type
        previous_arg_out = previous_arg_type[0].args[1]
        current_arg_in = arg_type[0].args[0]
        if previous_arg_out != current_arg_in:
            api.fail(
                "Layer mismatch, output not compatible with next layer",
                context=previous_arg[0],
                code=error_layer_output,
            )
            api.fail(
                "Layer mismatch, input not compatible with previous layer",
                context=arg[0],
                code=error_layer_input,
            )
        previous_arg_type = arg_type
    arg_in_type = first_arg_type[0].args[1]
    arg_out_type = arg_type[0].args[1]
    return Instance(ctx.default_return_type.type, [arg_in_type, arg_out_type])


error_layer_input = ErrorCode("layer-mismatch-input", "Invalid layer input", "Thinc")
error_layer_output = ErrorCode("layer-mismatch-output", "Invalid layer output", "Thinc")
