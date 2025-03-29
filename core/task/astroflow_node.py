import ray
import inspect
from pathlib import Path
from functools import wraps
import structlog
from typing import Any
import astropy.io.fits as pyfits  # type: ignore
import time

from context import context
from file_io.generic import get_content_hash, set_content_hash
from task.task import make_task_decorator, ContentHashProtocol

log = structlog.get_logger()

_MAX_PERFORMANCE_SAMPLES = 100


class ArtifactNotFoundException(Exception):
  pass


class WrongContentHashException(Exception):
  pass


class ImageFileContentHash(ContentHashProtocol):

  def get(self, path: Path) -> str:
    return get_content_hash(path)

  def set(self, path: Path, content_hash: str) -> None:
    set_content_hash(path, content_hash)


task = make_task_decorator(cache=context, content_hash=ImageFileContentHash())
