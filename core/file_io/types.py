import os
from pathlib import Path
from dataclasses import dataclass
from typing import TypeVar, Generic
from abc import ABC


class ContentHashable(ABC):
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
