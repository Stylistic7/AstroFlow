import os
from dataclasses import dataclass, field
import hashlib


def hash_bytes(b):
  return hashlib.sha256(b).hexdigest()


def hash_string(s):
  return hash_bytes(bytes(s, 'utf-8'))


@dataclass
class OutputFile:
  filename: str
  tags: dict = field(default_factory=dict)

  content_hash: str = None
  # input_stack_ids: list[str] = field(default_factory=list)
  # input_images: list[str] = field(default_factory=list)
  view_id: str = None

  def hash_file(self, assert_exists=False):
    if assert_exists:
      assert self.filename
      assert os.path.exists(self.filename)

    if self.filename and os.path.exists(self.filename):
      self.content_hash = hash_bytes(open(self.filename, 'rb').read())

  def __post_init__(self):
    self.hash_file()