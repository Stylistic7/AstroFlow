import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Any, overload, TypeAlias, Callable

# TODO: Rename the `classes.py` files to `types.py`

_FILTER_MAP = {
    'L': ['l', 'lum', 'luminance'],
    'R': ['r', 'red'],
    'G': ['g', 'green'],
    'B': ['b', 'blue'],
    'S': ['s', 'sii', 'sulfur', 'sulphur'],
    'H': ['h', 'ha', 'hα', 'halpha', 'h-alpha'],
    'O': ['o', 'oiii', 'oxygen', 'o-iii'],
}

_CANON = {}
for canon, aliases in _FILTER_MAP.items():
  for alias in aliases:
    _CANON[alias.lower()] = canon


def normalize_filter(name: str | None) -> str | None:
  if not name:
    return None
  key = re.sub(r'[^a-z0-9α]', '', name.lower())
  return _CANON.get(key)


def is_filter_compatible(a: str | None, b: str | None) -> bool:
  return normalize_filter(a) == normalize_filter(b)


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

    return next((i for i in self if is_filter_compatible(i.filter, key)), None)

  def __contains__(self, filters: str) -> Image | None:
    return all(
        any(is_filter_compatible(i.filter, filter)
            for i in self)
        for filter in filters)

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
