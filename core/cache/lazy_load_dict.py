import diskcache
from typing import Any, Iterable, Callable
from types import MethodType

Key = str | tuple[str, ...]


# TODO: I don't think this is going to work across Ray nodes. The ContextActor
# probably needs to re-create the LazyLoadDict with its own context.get() func.
class LazyLoadDict:
    __slots__ = ("name", "_orig_get")

    def __init__(self, name: str, orig_get: Callable[[Any], Any]):
        self.name = name
        self._orig_get = orig_get

    def __getitem__(self, k: Any) -> Any:
        return self._orig_get((self.name, k))

    def keys(self) -> Iterable[Any]:
        ks = self._orig_get(self.name) or ()
        return iter(ks)

    def __iter__(self):
        return self.keys()

    def values(self):
        for k in self.keys():
            yield self._orig_get((self.name, k))

    def items(self):
        for k in self.keys():
            yield k, self._orig_get((self.name, k))

    def __len__(self):
        ks = self._orig_get(self.name) or ()
        return len(ks)


def add_lazy_load_functionality(
    cache: diskcache.Cache, lazy_names: list[str]
) -> diskcache.Cache:
    lazy = set(lazy_names)

    # capture originals (bound to this instance)
    _orig_get = cache.get
    _orig_set = cache.set

    def _wrap_get(
        self: diskcache.Cache, key: Key, default: Any = None, **kw
    ) -> Any:
        if key in lazy:
            return LazyLoadDict(key, lambda kk: _orig_get(kk, None, **kw))
        return _orig_get(key, default, **kw)

    def _wrap_set(self: diskcache.Cache, key: Key, value: Any, **kw) -> bool:
        if key in lazy:
            if not isinstance(value, dict):
                raise TypeError("Lazy key must be set to a dict")
            # Write subentries first; publish index last (more consistent for readers)
            for subk, subv in value.items():
                _orig_set((key, subk), subv, **kw)
            return _orig_set(key, list(value.keys()), **kw)
        return _orig_set(key, value, **kw)

    def _or_equals(self: diskcache.Cache, key: Key, value: Any) -> None:
        if key in lazy:
            with self.transact():
                keys = _orig_get(key, dict())
                new_items = list(value.keys())
                keys = [k for k in keys if k not in new_items] + new_items
                _orig_set(key, keys)
                for k, v in value.items():
                    _orig_set((key, k), v)
        else:
            with self.transact():
                new_value = _orig_get(key) | value
                _orig_set(key, new_value)

    def _and_equals(
        self: diskcache.Cache, key: Key, value: Any, length_limit: int | None
    ) -> None:
        assert key not in lazy
        with self.transact():
            old_value = _orig_get(key, [])
            new_value = old_value + value
            if length_limit is not None:
                new_value = new_value[-length_limit:]
            _orig_set(key, new_value)

    cache.get = MethodType(_wrap_get, cache)
    cache.set = MethodType(_wrap_set, cache)

    cache.or_equals = MethodType(_or_equals, cache)
    cache.and_equals = MethodType(_and_equals, cache)

    return cache
