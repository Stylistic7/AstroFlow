from __future__ import annotations
import copyreg, importlib, sys, pickle
from ray.remote_function import RemoteFunction
from beartype import beartype


@beartype
def _module_binding(module: str, qualname: str) -> object:
    mod = sys.modules.get(module) or importlib.import_module(module)
    return getattr(mod, qualname, None)


def _reduce_remote_function(rf: RemoteFunction) -> tuple:
    fn = rf._function
    return _reconstruct_remote_function, (fn.__module__, fn.__qualname__)


def _reconstruct_remote_function(module: str, qualname: str) -> RemoteFunction:
    bound = _module_binding(module, qualname)
    if isinstance(bound, RemoteFunction):
        return bound
    raise pickle.UnpicklingError(f"Cannot resolve RemoteFunction {module}.{qualname}")


@beartype
def enable_function_fqname_pickling() -> None:
    copyreg.pickle(RemoteFunction, _reduce_remote_function)
