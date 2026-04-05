from typing import Any
from collections.abc import Callable
import inspect
import xxhash
from dataclasses import fields
from beartype import beartype

from file_io.types import ContentHashable
from protocols import ContentHash
from cache.content_hashed import ContentHashed
from file_io.generic import get_content_hash


# TODO: Maybe unify these.
@beartype
def hash_string(x: Any) -> str:
    return xxhash.xxh64(x).hexdigest()


def _maybe_use_content_hash(obj: Any) -> Any:
    if isinstance(obj, ContentHashable):
        return ContentHash(get_content_hash(obj))
    elif hasattr(obj, "content_hash") and obj.content_hash:
        return obj.content_hash
    else:
        return make_hashable(obj)


@beartype
def hash_list(L: Any, debug: bool = False) -> str:
    L2 = [_maybe_use_content_hash(x) for x in L]

    if debug:
        for i, o in zip(L, L2):
            print(f"### L type {type(i)} L2 type {type(o)} {str(o)} ")

    return xxhash.xxh64(str(L2)).hexdigest()


@beartype
def hash_list_int(L: Any) -> int:
    L = [_maybe_use_content_hash(x) for x in L]
    return xxhash.xxh64(str(L)).intdigest()


@beartype
def hash_dict_int(x: dict) -> int:
    return hash_list_int(tuple(make_hashable(x)))


@beartype
def hash_dataclass(x: Any) -> int:
    return hash_list_int(make_hashable(getattr(x, f.name)) for f in fields(x))


@beartype
def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted((k, _maybe_use_content_hash(v)) for k, v in value.items())
        )
    elif isinstance(value, set):
        return tuple(sorted(_maybe_use_content_hash(v) for v in value))
    elif isinstance(value, (tuple, list)):
        return tuple(_maybe_use_content_hash(v) for v in value)
    elif isinstance(value, frozenset):
        return tuple(sorted(value))
    elif _is_dataclass_instance(value):
        return tuple(
            (f.name, _maybe_use_content_hash(getattr(value, f.name)))
            for f in fields(value)
        )
    else:
        return value


def _is_dataclass_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


@beartype
def hash_source_code(func: Callable) -> str:
    return xxhash.xxh64(inspect.getsource(func)).hexdigest()


@beartype
def hash_call(serialized_func: str, *args, **kwargs) -> str:
    kwargs_list = sorted(kwargs.items())
    return hash_list([serialized_func, *args, *kwargs_list], debug=False)
