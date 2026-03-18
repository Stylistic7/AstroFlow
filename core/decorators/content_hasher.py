import os

from cache.content_hashed import ContentHashed
from protocols import ContentHash, ContentHashProtocol
from file_io.types import ContentHashable
from file_io.generic import get_content_hash, set_content_hash
from hashing import hash_list
from typing import Any
from dataclasses import replace


def _is_context_ref(obj: Any) -> bool:
    """Check if obj is a ContextRef without importing to avoid circular deps."""
    return type(obj).__name__ == "ContextRef" and hasattr(obj, "content_hash")


def _is_context_ref_dict(obj: Any) -> bool:
    """Check if obj is a dict whose values are all ContextRefs."""
    return isinstance(obj, dict) and bool(obj) and all(_is_context_ref(v) for v in obj.values())


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
        elif isinstance(obj, list) and all(isinstance(x, ContentHashed) for x in obj):
            return [
                replace(x, content_hash=hash_list([content_hash, i]))
                for i, x in enumerate(obj)
            ]
        elif isinstance(obj, dict) and all(
            isinstance(x, ContentHashed) for x in obj.values()
        ):
            return type(obj)({
                k: replace(v, content_hash=hash_list([content_hash, i]))
                for i, (k, v) in enumerate(obj.items())
            })
        elif _is_context_ref_dict(obj):
            # For dicts of ContextRefs, derive a per-entry hash from the
            # call_hash and the entry index, then re-store each object with
            # the updated content_hash. This ensures downstream hash_call()
            # computations see a stable, non-empty content_hash.
            from dataclasses import replace as dc_replace
            from cache.context_ref import ContextRef
            result = type(obj)()
            for i, (k, ref) in enumerate(obj.items()):
                derived = hash_list([content_hash, i])
                obj_val = ref.get()
                if isinstance(obj_val, ContentHashed):
                    updated_obj = dc_replace(obj_val, content_hash=derived)
                    new_ref = ContextRef.store(ref.cache_key, updated_obj)
                else:
                    new_ref = ContextRef(
                        cache_key=ref.cache_key,
                        content_hash=derived,
                        ref=ref.ref,
                    )
                result[k] = new_ref
            return result
        else:
            return obj
