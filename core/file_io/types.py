import os
from pathlib import Path
from dataclasses import dataclass
from typing import TypeVar, Generic, Self
from abc import ABC


class ContentHashable(ABC):

  def resolve_at(self, root: Path | str) -> Self:
    return type(self)(Path(root) / self)


class ContentHashableFile(type(Path()), ContentHashable):
  pass


class FitsFilePath(type(Path()), ContentHashable):
  pass


class XisfFilePath(type(Path()), ContentHashable):
  pass


class TiffFilePath(type(Path()), ContentHashable):
  pass


T = TypeVar("T", bound=Path)


@dataclass
class WithSidecar(Generic[T], ContentHashable):
  path: T

  def __fspath__(self) -> str:
    return os.fspath(self.path)

  def __str__(self) -> str:
    return str(self.path)

  def __truediv__(self, other):
    return self.path / other

  def __getattr__(self, name):
    path = object.__getattribute__(self, "path")
    return getattr(path, name)

  def resolve_at(self, root: Path | str) -> Self:
    resolved = self.path.resolve_at(root)
    return type(self)(resolved)
