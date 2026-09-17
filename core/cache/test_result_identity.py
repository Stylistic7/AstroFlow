"""The pure-Merkle result stamp: a derived value's content hash IS its
producer's call hash, so a trace-time handle (stamped with the same call
hash) and the runtime value hash alike. Identity ROOTS opt out by
declaration — their content hash names the data they identify.
"""
import dataclasses

import ray

from cache.content_hashed import ContentHashed
from decorators import node
from hashing import hash_call


@dataclasses.dataclass(frozen=True)
class _Derived(ContentHashed):
    payload: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _Carrier:
    """Not a ContentHashed, but carries a `content_hash` field (the shape of
    `ImageMetadata`)."""
    content_hash: str
    payload: int


@node
def derive(x: int) -> _Derived:
    return _Derived(content_hash="own-" + str(x), payload=x)


@node(identity_root=True)
def identify(x: int) -> _Derived:
    return _Derived(content_hash="membership-" + str(x), payload=x)


@node
def carry(x: int) -> _Carrier:
    return _Carrier(content_hash="", payload=x)


def _call_hash(op, *args) -> str:
    return hash_call(op._fn_hash, *args)


def test_derived_result_hashes_as_its_call(test_context):
    r = ray.get(derive.remote(3))
    assert r.content_hash == _call_hash(derive, 3)
    assert r.payload == 3


def test_identity_root_keeps_its_own_hash(test_context):
    r = ray.get(identify.remote(3))
    assert r.content_hash == "membership-3"


def test_any_dataclass_with_a_content_hash_field_is_stamped(test_context):
    r = ray.get(carry.remote(5))
    assert r.content_hash == _call_hash(carry, 5)


def test_cache_hit_returns_the_stamped_value(test_context):
    first = ray.get(derive.remote(7))
    again = ray.get(derive.remote(7))
    assert again.content_hash == first.content_hash == _call_hash(derive, 7)
