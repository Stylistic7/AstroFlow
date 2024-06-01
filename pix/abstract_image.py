from dataclasses import dataclass, field
import uuid
# import functools
from enum import Enum

# TODO: There are two major types of operations:
#
# 1. Running an existing JavaScript program, like ImageSolver or WBPP. Thse have
# to be run via independent PixInsight shell scripts.
#
# 2. Running a Process or series of Processes via custom JavaScripts. These can
# be chained together, saved as a tempfile, and run as a single shell script.
#
# I might be able to use a custom set of decorators to achieve all of this.
#
#

# class FunctionType(Enum):
#   RETURNS_JAVASCRIPT = 1
#   GREEN = 2
#   BLUE = 3

# def register(function_type):
#   def wrapped(fn):
#     @functools.wraps(fn)
#     def wrapped_f(*args, **kwargs):
#       return fn(*args, **kwargs)

#     # wrapped_f.__doc__ += "(command=%s)" % function_type
#     wrapped_f.function_type = function_type
#     return wrapped_f

#   return wrapped


@dataclass
class JavaScript:
  code: str = None
  open_views: list[str] = field(default_factory=list)
  saved_files: list[str] = field(default_factory=list)


@dataclass
class AbstractImage:
  view_id: str = None
  filename: str = None
  dirty: bool = False
  tag: str = None

  # def __del__(self):
  #   print(f'### AbstractImage deleted (view_id {self.view_id} '
  #         f'filename {self.filename})')
