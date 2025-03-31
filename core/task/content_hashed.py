from dataclasses import dataclass
from typing import NewType
from task.task import ContentHash


@dataclass(frozen=True, kw_only=True)
class ContentHashed:
  content_hash: ContentHash = ''
