from dataclasses import dataclass
import hashlib


def hash_bytes(b):
  return hashlib.sha256(b).hexdigest()


def hash_string(s):
  return hash_bytes(bytes(s, 'utf-8'))


@dataclass
class OutputFile:
  filename: str
  content_hash: str = None

  def __post_init__(self):
    assert self.filename
    self.content_hash = hash_bytes(open(self.filename, 'rb').read())