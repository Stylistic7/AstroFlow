import os

from cache.content_hashed import ContentHashed
from protocols import ContentHash, ContentHashProtocol
from file_io.types import ContentHashable
from file_io.generic import get_content_hash, set_content_hash
from typing import Any
from dataclasses import replace


class ContentHasher(ContentHashProtocol):

  # TODO: Having both `ContentHashed` and `ContentHashable` is a bit weird.
  # The former is for objects like dataclasses that have `content_hash` fields.
  # The latter is for objects like TiffFilePath that represent files with
  # headers in which a content hash can be stored.
  def has(self, obj: Any) -> bool:
    return isinstance(obj, (ContentHashable, ContentHashed))

  def get(self, obj: Any) -> ContentHash:
    if isinstance(obj, ContentHashable):
      if os.path.exists(obj):
        return ContentHash(get_content_hash(obj))
    elif isinstance(obj, ContentHashed):
      return obj.content_hash

  def set(self, obj: Any, content_hash: ContentHash) -> Any:
    if isinstance(obj, ContentHashable):
      set_content_hash(obj, content_hash)
      return obj
    elif isinstance(obj, ContentHashed):
      return replace(obj, content_hash=content_hash)
    else:
      return obj
