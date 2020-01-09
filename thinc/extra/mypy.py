from typing import Callable, Dict, List, Optional
from typing import Type as TypingType
from typing import TypeVar, Union
from mypy.types import UnboundType

from mypy.mro import MroError, calculate_mro
from mypy.nodes import (
    ARG_STAR2,
    GDEF,
    MDEF,
    Argument,
    Block,
    ClassDef,
    Expression,
    FuncBase,
    NameExpr,
    RefExpr,
    StrExpr,
    SymbolNode,
    SymbolTable,
    SymbolTableNode,
    TupleExpr,
    TypeInfo,
    Var,
)
from mypy.plugin import (
    AnalyzeTypeContext,
    ClassDefContext,
    FunctionContext,
    Plugin,
    SemanticAnalyzerPluginInterface,
)
from mypy.plugins.common import add_method
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    NoneTyp,
    Type,
    TypeOfAny,
    UninhabitedType,
    UnionType,
)
from mypy.typevars import fill_typevars_with_any

from ..types import Final, Literal

try:
    from mypy.types import get_proper_type
except ImportError:
    get_proper_type = lambda x: x


FLOATS_1D = "thinc.types.Floats1d"  # type: Final
FLOATS_2D = "thinc.types.Floats2d"  # type: Final
FLOATS_3D = "thinc.types.Floats3d"  # type: Final
FLOATS_4D = "thinc.types.Floats4d"  # type: Final
FLOATS_ND = "thinc.types.FloatsNd"  # type: Final

INTS_1D = "thinc.types.Ints1d"  # type: Final
INTS_2D = "thinc.types.Ints1d"  # type: Final
INTS_3D = "thinc.types.Ints1d"  # type: Final
INTS_4D = "thinc.types.Ints1d"  # type: Final
INTS_ND = "thinc.types.Ints1d"  # type: Final

MODEL_TYPE = "thinc.model.Model"  # type: Final

# See https://github.com/python/mypy/issues/6617 for plugin API updates.

_TYPED_DECORATORS = {
    "thinc.config.registry.layers",
    "thinc.config.registry.optimizers",
    "thinc.config.registry.schedules",
}


def _change_decorator_function_type(
    decorated: CallableType, decorator: CallableType,
) -> CallableType:
    """Replaces revealed argument types by mypy with types from decorated."""
    decorator.arg_types = decorated.arg_types
    decorator.arg_kinds = decorated.arg_kinds
    decorator.arg_names = decorated.arg_names
    return decorator


def _analyze_decorator(ctx: FunctionContext):
    """Tells us what to do when one of the typed decorators is called."""
    if not isinstance(ctx.arg_types[0][0], CallableType):
        return ctx.default_return_type
    if not isinstance(ctx.default_return_type, CallableType):
        return ctx.default_return_type
    return _change_decorator_function_type(
        ctx.arg_types[0][0], ctx.default_return_type,
    )


def _analyze_model(ctx: AnalyzeTypeContext) -> UnboundType:
    """Adjust shapes for model definitions?"""
    return ctx.type


class ThincPlugin(Plugin):
    """Thinc plugin to do static analysis on user-defined layers and models."""

    def get_function_hook(
        self, fullname: str,
    ) -> Optional[Callable[[FunctionContext], Type]]:
        """Add better types to thinc decorators.

        See: https://github.com/python/mypy/issues/3157"""
        if fullname in _TYPED_DECORATORS:
            return _analyze_decorator
        return None

    def get_type_analyze_hook(
        self, fullname: str
    ) -> Optional[Callable[[AnalyzeTypeContext], Type]]:
        """Customize behaviour of the type analyzer for given full names.
        This method is called during the semantic analysis pass whenever mypy sees an
        unbound type. For example, while analysing this code:
            from lib import Special, Other
            var: Special
            def func(x: Other[int]) -> None:
                ...
        this method will be called with 'lib.Special', and then with 'lib.Other'.
        The callback returned by plugin must return an analyzed type,
        i.e. an instance of `mypy.types.Type`.
        """
        if MODEL_TYPE in fullname:
            return _analyze_model
        return None


def plugin(version: str) -> "TypingType[Plugin]":
    return ThincPlugin
