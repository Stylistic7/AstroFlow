from typing import Any, NewType, Protocol, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol):

  def __getitem__(self, key: str) -> Any:
    ...

  def __setitem__(self, key: str, value: Any) -> None:
    ...

  def get(self, key: str, default: Any = None) -> Any:
    ...

  def __contains__(self, key: str) -> bool:
    ...


ContentHash = NewType('ContentHash', str)


@runtime_checkable
class ContentHashProtocol(Protocol):

  def has(self, obj: Any) -> bool:
    ...

  def get(self, obj: Any) -> ContentHash:
    ...

  def set(self, obj: Any, content_hash: ContentHash) -> Any:
    ...
