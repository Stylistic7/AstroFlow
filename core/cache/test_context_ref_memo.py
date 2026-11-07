"""ContextRef.get reads its value once per handle (pure: no Ray)."""
import pickle

import cache.context_ref as context_ref
from cache.context_ref import ContextRef


class _CountingStore(dict):
    reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self.get(key)


def test_get_reads_once_per_handle_and_never_pickles_the_copy(monkeypatch):
    store = _CountingStore({("library", "h"): {"a": 1}})
    monkeypatch.setattr(context_ref, "_active_context", store)
    ref = ContextRef(cache_key=("library", "h"), content_hash="h")

    assert ref.get() is ref.get()
    assert store.reads == 1

    copy = pickle.loads(pickle.dumps(ref))
    assert "_value" not in copy.__dict__ and copy == ref
    assert copy.get() == {"a": 1} and store.reads == 2


def test_a_missing_value_is_read_again(monkeypatch):
    store = _CountingStore()
    monkeypatch.setattr(context_ref, "_active_context", store)
    ref = ContextRef(cache_key=("library", "h"), content_hash="h")

    assert ref.get() is None and ref.get() is None
    assert store.reads == 2
