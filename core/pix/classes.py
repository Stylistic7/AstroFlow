from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Any


class ArgumentType(Enum):
  FILENAMES = 1
  VIEW_IDS = 2


@dataclass
class ShellScriptTask:
  javascript: str = None
  javascript_filename: str = None
  parameters: dict = field(default_factory=dict)
  arguments: dict = field(default_factory=dict)


@dataclass
class Image:
  tmp_dir_obj: object = None
  filename: str = None
  view_id: str = None
  filter: str = None
  tag: str = None
  metadata: dict = field(default_factory=dict)
  dirty: bool = False

  def __str__(self):
    return self.filename


ImageList = list[Image]
FilenameList = list[str]


class PixInsightFn(Protocol):

  def __call__(self, filenames: FilenameList, *args: Any,
               **kwargs: Any) -> ImageList:
    ...
