from typing import Dict, List
import itertools
from mypy.errors import Errors
from mypy.errorcodes import ErrorCode
from mypy.options import Options
from mypy.plugin import FunctionContext, Plugin, CheckerPluginInterface
from mypy.types import Instance, Type, CallableType, TypeVarType
from mypy.nodes import Expression, CallExpr, NameExpr, FuncDef, Decorator, MypyFile
from mypy.checker import TypeChecker
from mypy.subtypes import is_subtype

thinc_model_fullname = "thinc.model.Model"
chained_out_fullname = "thinc.types.XY_YZ_OutT"
intoin_outtoout_out_fullname = "thinc.types.XY_XY_OutT"


def plugin(version: str):
    return ThincPlugin


class ThincPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_function_hook(self, fullname: str):
        return function_hook


def function_hook(ctx: FunctionContext) -> Type:
    try:
        return get_reducers_type(ctx)
    except AssertionError:
        # Add more function callbacks here
        return ctx.default_return_type


def get_reducers_type(ctx: FunctionContext) -> Type:
    assert isinstance(ctx.context, CallExpr)
    assert isinstance(ctx.api, TypeChecker)
    assert isinstance(ctx.default_return_type, Instance)
    assert isinstance(ctx.context.callee, NameExpr)
    assert isinstance(ctx.context.callee.node, (FuncDef, Decorator))
    assert isinstance(ctx.context.callee.node.type, CallableType)
    assert isinstance(ctx.context.callee.node.type.ret_type, Instance)
    assert ctx.context.callee.node.type.ret_type.args
    assert len(ctx.context.callee.node.type.ret_type.args) == 2
    out_type = ctx.context.callee.node.type.ret_type.args[1]
    assert isinstance(out_type, TypeVarType)
    assert out_type.fullname
    if out_type.fullname not in {intoin_outtoout_out_fullname, chained_out_fullname}:
        return ctx.default_return_type
    args = list(itertools.chain(*ctx.args))
    arg_types = []
    for arg_type in itertools.chain(*ctx.arg_types):
        assert isinstance(arg_type, Instance)
        arg_types.append(arg_type)
    arg_pairs = list(zip(args[:-1], args[1:]))
    arg_types_pairs = list(zip(arg_types[:-1], arg_types[1:]))
    if out_type.fullname == chained_out_fullname:
        for (arg1, arg2), (type1, type2) in zip(arg_pairs, arg_types_pairs):
            assert isinstance(type1, Instance)
            assert isinstance(type2, Instance)
            assert type1.type.fullname == thinc_model_fullname
            assert type2.type.fullname == thinc_model_fullname
            check_chained(
                l1_arg=arg1, l1_type=type1, l2_arg=arg2, l2_type=type2, api=ctx.api
            )
        return Instance(
            ctx.default_return_type.type, [arg_types[0].args[0], arg_types[-1].args[1]]
        )
    elif out_type.fullname == intoin_outtoout_out_fullname:
        for (arg1, arg2), (type1, type2) in zip(arg_pairs, arg_types_pairs):
            assert isinstance(type1, Instance)
            assert isinstance(type2, Instance)
            assert type1.type.fullname == thinc_model_fullname
            assert type2.type.fullname == thinc_model_fullname
            check_intoin_outtoout(
                l1_arg=arg1, l1_type=type1, l2_arg=arg2, l2_type=type2, api=ctx.api
            )
        return Instance(
            ctx.default_return_type.type, [arg_types[0].args[0], arg_types[0].args[1]]
        )
    assert False, "Thinc mypy plugin error: it should return before this point"


def check_chained(
    *,
    l1_arg: Expression,
    l1_type: Instance,
    l2_arg: Expression,
    l2_type: Instance,
    api: CheckerPluginInterface,
):
    if not is_subtype(l1_type.args[1], l2_type.args[0]):
        api.fail(
            f"Layer outputs type ({l1_type.args[1]}) but the next layer expects ({l2_type.args[0]}) as an input",
            l1_arg,
            code=error_layer_output,
        )
        api.fail(
            f"Layer input type ({l2_type.args[0]}) is not compatible with output ({l1_type.args[1]}) from previous layer",
            l2_arg,
            code=error_layer_input,
        )


def check_intoin_outtoout(
    *,
    l1_arg: Expression,
    l1_type: Instance,
    l2_arg: Expression,
    l2_type: Instance,
    api: CheckerPluginInterface,
):
    if l1_type.args[0] != l2_type.args[0]:
        api.fail(
            f"Layer input ({l1_type.args[0]}) not compatible with next layer input ({l2_type.args[0]})",
            l1_arg,
            code=error_layer_input,
        )
        api.fail(
            f"Layer input ({l2_type.args[0]}) not compatible with previous layer input ({l1_type.args[0]})",
            l2_arg,
            code=error_layer_input,
        )
    if l1_type.args[1] != l2_type.args[1]:
        api.fail(
            f"Layer output ({l1_type.args[1]}) not compatible with next layer output ({l2_type.args[1]})",
            l1_arg,
            code=error_layer_output,
        )
        api.fail(
            f"Layer output ({l2_type.args[1]}) not compatible with previous layer output ({l1_type.args[1]})",
            l2_arg,
            code=error_layer_output,
        )


error_layer_input = ErrorCode("layer-mismatch-input", "Invalid layer input", "Thinc")
error_layer_output = ErrorCode("layer-mismatch-output", "Invalid layer output", "Thinc")


class IntrospectChecker(TypeChecker):
    def __init__(
        self,
        errors: Errors,
        modules: Dict[str, MypyFile],
        options: Options,
        tree: MypyFile,
        path: str,
        plugin: Plugin,
    ):
        self._error_messages: List[str] = []
        super().__init__(errors, modules, options, tree, path, plugin)
