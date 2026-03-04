from typing import Any, Callable
import inspect
import xxhash
from dataclasses import fields

from file_io.types import ContentHashable
from protocols import ContentHash
from cache.content_hashed import ContentHashed
from file_io.generic import get_content_hash


# TODO: Maybe unify these.
def hash_string(x: Any) -> str:
    return xxhash.xxh64(x).hexdigest()


def _maybe_use_content_hash(obj):
    if isinstance(obj, ContentHashable):
        return ContentHash(get_content_hash(obj))
    elif hasattr(obj, "content_hash"):
        return obj.content_hash
    else:
        return make_hashable(obj)


def hash_list(L: Any, debug: bool = False) -> str:
    L2 = [_maybe_use_content_hash(x) for x in L]

    if debug:
        for i, o in zip(L, L2):
            print(f"### L type {type(i)} L2 type {type(o)} {str(o)} ")

    return xxhash.xxh64(str(L2)).hexdigest()


def hash_list_int(L: Any) -> int:
    L = [_maybe_use_content_hash(x) for x in L]
    return xxhash.xxh64(str(L)).intdigest()


def hash_dict_int(x: dict) -> int:
    return hash_list_int(tuple(make_hashable(x)))


def hash_dataclass(x: Any) -> int:
    return hash_list_int(make_hashable(getattr(x, f.name)) for f in fields(x))


def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted((k, _maybe_use_content_hash(v)) for k, v in value.items())
        )
    elif isinstance(value, (tuple, list, set)):
        return tuple(_maybe_use_content_hash(v) for v in value)
    elif isinstance(value, frozenset):
        return tuple(sorted(value))
    else:
        return value


def hash_source_code(func: Callable) -> bytes:
    return xxhash.xxh64(inspect.getsource(func)).hexdigest()


def hash_call(serialized_func: bytes, *args, **kwargs) -> str:
    kwargs_list = sorted(kwargs.items())
    return hash_list([serialized_func, *args, *kwargs_list], debug=False)
