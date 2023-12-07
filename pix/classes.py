from dataclasses import dataclass, field
from enum import Enum


# TODO: Remove these somehow. Maybe make them just lists of strings. If the
# string is a valid filename, it's a file. Otherwise, it's a view id?
class FileList(list):
  pass


class ViewIdList(list):
  pass


class ArgumentType(Enum):
  FILENAMES = 1
  VIEW_IDS = 2


@dataclass
class ShellScriptTask:
  javascript: str = None
  javascript_filename: str = None
  parameters: dict = field(default_factory=dict)
  arguments: dict = field(default_factory=dict)
