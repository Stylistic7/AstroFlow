import diskcache
from typing import Any, Iterable, Callable
from types import MethodType


def _reconstruct_lazy_typed(base_type: type, name: str):
    from context.context import context

    def orig_get_fn(kk):
        if isinstance(kk, tuple):
            return context.get(kk)
        return context.get_lazy_keys(kk)

    cls = _make_lazy_subclass(base_type)
    inst = cls.__new__(cls)
    dict.__init__(inst)
    inst._lazy_init(name, orig_get_fn)
    return inst


def _reconstruct_lazy_untyped(name: str):
    from context.context import context

    def orig_get_fn(kk):
        if isinstance(kk, tuple):
            return context.get(kk)
        return context.get_lazy_keys(kk)

    return LazyLoadDict(name, orig_get_fn)

Key = str | tuple[str, ...]

_lazy_subclass_cache: dict[type, type] = {}


def _make_lazy_subclass(base_type: type) -> type:
    """Return a lazy-loading subclass of base_type (which must be a dict subclass).

    The returned class passes isinstance checks for base_type while keeping all
    values stored lazily in the diskcache rather than in the dict's own storage.
    """
    if base_type in _lazy_subclass_cache:
        return _lazy_subclass_cache[base_type]

    class LazySubclass(base_type):
        def _lazy_init(self, name: str, orig_get: Callable) -> None:
            self._lazy_name = name
            self._lazy_orig_get = orig_get

        def __getitem__(self, k: Any) -> Any:
            return self._lazy_orig_get((self._lazy_name, k))

        def get(self, k: Any, default: Any = None) -> Any:
            if k in list(self.keys()):
                return self[k]
            return default

        def keys(self) -> Iterable[Any]:
            ks = self._lazy_orig_get(self._lazy_name) or ()
            return iter(ks)

        def __iter__(self):
            return self.keys()

        def values(self):
            for k in self.keys():
                yield self._lazy_orig_get((self._lazy_name, k))

        def items(self):
            for k in self.keys():
                yield k, self._lazy_orig_get((self._lazy_name, k))

        def __len__(self) -> int:
            ks = self._lazy_orig_get(self._lazy_name) or ()
            return len(ks)

        def __contains__(self, k: Any) -> bool:
            return k in list(self.keys())

        def __reduce__(self):
            return (_reconstruct_lazy_typed, (base_type, self._lazy_name))

    LazySubclass.__name__ = f"Lazy{base_type.__name__}"
    LazySubclass.__qualname__ = f"Lazy{base_type.__name__}"
    _lazy_subclass_cache[base_type] = LazySubclass
    return LazySubclass


class LazyLoadDict:
    __slots__ = ("name", "_orig_get")

    def __init__(self, name: str, orig_get: Callable[[Any], Any]):
        self.name = name
        self._orig_get = orig_get

    def __getitem__(self, k: Any) -> Any:
        return self._orig_get((self.name, k))

    def get(self, k: Any, default: Any) -> Any:
        if k in self.keys():
            return self.__getitem__(k)
        else:
            return default

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

    def __reduce__(self):
        return (_reconstruct_lazy_untyped, (self.name,))


def add_lazy_load_functionality(
    cache: diskcache.Cache,
    lazy_names: list[str],
    lazy_types: dict[str, type] | None = None,
) -> diskcache.Cache:
    lazy = set(lazy_names)
    lazy_types = lazy_types or {}

    # capture originals (bound to this instance)
    _orig_get = cache.get
    _orig_set = cache.set

    def _wrap_get(
        self: diskcache.Cache, key: Key, default: Any = None, **kw
    ) -> Any:
        if key in lazy:
            orig_get_fn = lambda kk: _orig_get(kk, None, **kw)
            if key in lazy_types:
                cls = _make_lazy_subclass(lazy_types[key])
                inst = cls.__new__(cls)
                dict.__init__(inst)
                inst._lazy_init(key, orig_get_fn)
                return inst
            return LazyLoadDict(key, orig_get_fn)
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
                new_value = _orig_get(key, dict()) | value
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

    cache._orig_get = _orig_get

    cache.get = MethodType(_wrap_get, cache)
    cache.set = MethodType(_wrap_set, cache)

    cache.or_equals = MethodType(_or_equals, cache)
    cache.and_equals = MethodType(_and_equals, cache)

    return cache
