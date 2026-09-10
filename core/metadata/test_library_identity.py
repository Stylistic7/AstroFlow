"""A moved library must re-key the library ref.

Frame identity is content-addressed and therefore path-independent — which is
correct, and which is exactly why `hash_list([metadata_dict])` cannot see a
move: `make_hashable` collapses every `ImageMetadata` value to its own
`content_hash`. Relocating the library (Astro/Astro2 -> AstroRaw, 2026-08-18)
thus changed no ref hash, invalidated nothing, and every cached Run went on
serving paths that no longer existed; every WBPP stack died on
FileNotFoundError and it took three rounds of manual eviction to clear.
"""
from pathlib import Path

import numpy as np

from cache.context_ref import ContextRef
from hashing import hash_list
from metadata.classes import ImageMetadata, MetadataDict
from metadata.image_metadata import _library_identity


def _library(root: str) -> MetadataDict:
    return MetadataDict({
        h: ImageMetadata(
            content_hash=h, stable_hash=h, filename=Path(f"{root}/{h}.fit"),
            header={"FILTER": "Ha"}, width_pixels=4, height_pixels=4,
            channels=1, dtype=np.dtype("float32"),
        )
        for h in ("aaaa", "bbbb", "cccc")
    })


def test_hashing_the_dict_alone_is_blind_to_a_move():
    """The premise. If this ever starts failing, `_library_identity` has
    become redundant and should go."""
    assert hash_list([_library("/Volumes/Astro/Raw")]) == \
        hash_list([_library("/Volumes/AstroRaw/Raw")])


def test_identity_moves_with_the_files():
    assert _library_identity(_library("/Volumes/Astro/Raw")) != \
        _library_identity(_library("/Volumes/AstroRaw/Raw"))


def test_identity_is_order_independent():
    a = _library("/Volumes/Astro/Raw")
    b = MetadataDict({k: a[k] for k in reversed(list(a))})
    assert _library_identity(a) == _library_identity(b)


def test_stored_ref_content_hash_follows_the_identity(test_context):
    """The identity is what `store_large` keys the ref on — not just an
    input to a comparison. Same bytes stored, two different refs."""
    here = _library("/Volumes/Astro/Raw")
    there = _library("/Volumes/AstroRaw/Raw")
    a = ContextRef.store_large(here, _library_identity(here))
    b = ContextRef.store_large(there, _library_identity(there))
    assert a.content_hash != b.content_hash
    assert a.cache_key != b.cache_key
    assert {k: str(v.filename) for k, v in a.get().items()} == \
        {k: str(v.filename) for k, v in here.items()}
