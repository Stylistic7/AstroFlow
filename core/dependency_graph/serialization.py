from __future__ import annotations
import copyreg, importlib, sys, pickle
from ray.remote_function import RemoteFunction


def _module_binding(module: str, qualname: str):
  mod = sys.modules.get(module) or importlib.import_module(module)
  return getattr(mod, qualname, None)


def _reduce_remote_function(rf: RemoteFunction):
  print(f'### _reduce_remote_function {rf}')
  fn = rf._function
  return _reconstruct_remote_function, (fn.__module__, fn.__qualname__)


def _reconstruct_remote_function(module: str, qualname: str) -> RemoteFunction:
  bound = _module_binding(module, qualname)
  print(f'### _reconstruct_remote_function {module=} {qualname=} {bound=}')
  if isinstance(bound, RemoteFunction):
    print(f'### returning {bound=}')
    return bound
  raise pickle.UnpicklingError(
      f"Cannot resolve RemoteFunction {module}.{qualname}")


def enable_function_fqname_pickling() -> None:
  print("### enable_function_fqname_pickling")
  copyreg.pickle(RemoteFunction, _reduce_remote_function)
