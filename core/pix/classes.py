from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Any, overload, TypeAlias, Callable

# TODO: Rename the `classes.py` files to `types.py`


class ArgumentType(Enum):
  FILENAMES = 1
  VIEW_IDS = 2


@dataclass
class ShellScriptTask:
  javascript: str | None = None
  javascript_filename: str | None = None
  parameters: dict = field(default_factory=dict)
  arguments: dict = field(default_factory=dict)


@dataclass
class Image:
  tmp_dir_obj: object | None = None
  filename: str | None = None
  view_id: str | None = None
  filter: str | None = None
  tag: str | None = None
  metadata: dict = field(default_factory=dict)
  dirty: bool = False


class ImageList(list[Image]):

  def __getitem__(self, key: int | str) -> Image | None:
    if isinstance(key, int):
      return super().__getitem__(key)
    return next((i for i in self if i.filter == key), None)

  # def __contains__(self, filters: str) -> Image | None:
  #   return next(iter(i for i in self if i.filter == filter), None) is not None

  def channels(self, filters: str) -> list[Image]:
    return [self[f] for f in filters]


FilenameList = list[str]

# class PixInsightFn(Protocol):

#   @overload
#   def __call__(self, filenames: FilenameList) -> ImageList:
#     ...

#   @overload
#   def __call__(self, filenames: FilenameList, *args: Any,
#                **kwargs: Any) -> ImageList:
#     ...

PixInsightFn: TypeAlias = Callable[..., Any]
