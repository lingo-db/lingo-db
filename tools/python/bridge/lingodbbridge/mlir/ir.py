from ._mlir_libs._mlir  import ir

# Convenience decorator for registering user-friendly Attribute builders.
def register_attribute_builder(kind, replace=False):
    def decorator_builder(func):
        AttrBuilder.insert(kind, func, replace=replace)
        return func

    return decorator_builder

class Context(ir._BaseContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
ir.Context=Context
from ._mlir_libs._mlir.ir import *
from ._mlir_libs._mlir import register_type_caster
