"""`DefectTable`: the library's hot-pixel candidates as flat arrays, built
from per-frame sets, unioned, subset and hashed by content. Pure."""
import pickle

import numpy as np

from metadata.classes import DefectTable

SETS = {"c" * 16: {30, 10, 20}, "a" * 16: {5}, "b" * 16: None, "d" * 16: {7: 1.5, 3: 0.2}}


def test_from_sets_sorts_frames_and_pixels_and_drops_empty_frames():
    t = DefectTable.from_sets(SETS)
    assert t.frames.tolist() == ["a" * 16, "c" * 16, "d" * 16]
    assert t.offsets.tolist() == [0, 1, 4, 6] and t.pixels.dtype == np.int32
    assert t.pixels_of("c" * 16).tolist() == [10, 20, 30]
    assert t.pixels_of("d" * 16).tolist() == [3, 7]             # a {pixel: strength} dict keeps its pixels
    assert "a" * 16 in t and "b" * 16 not in t and "z" not in t and len(t) == 3
    assert t.counts().tolist() == [1, 3, 2]


def test_subset_drop_and_concat_rebuild_consistent_tables():
    t = DefectTable.from_sets(SETS)
    s = t.subset(["d" * 16, "a" * 16, "x" * 16])
    assert s.frames.tolist() == ["a" * 16, "d" * 16] and s.pixels.tolist() == [5, 3, 7]
    assert t.drop(["c" * 16]) == s and t.drop([]) is t
    other = DefectTable.from_sets({"b" * 16: {1, 2}})
    u = DefectTable.concat([s, other, t.subset(["c" * 16])])
    assert u == DefectTable.from_sets({**{k: v for k, v in SETS.items() if v}, "b" * 16: {1, 2}})
    assert DefectTable.concat([]).frames.tolist() == [] and len(DefectTable.from_sets({})) == 0


def test_content_hash_is_by_value_and_survives_pickle_and_read_only_arrays():
    a, b = DefectTable.from_sets(SETS), DefectTable.from_sets(dict(reversed(list(SETS.items()))))
    assert a.content_hash == b.content_hash and hash(a) == hash(b) and a == b
    assert DefectTable.from_sets({"a" * 16: {6}}).content_hash != a.content_hash
    back = pickle.loads(pickle.dumps(a, protocol=5))
    assert back == a and back.content_hash == a.content_hash
    for arr in (back.frames, back.offsets, back.pixels):     # what a reader of the object store holds
        arr.setflags(write=False)
    assert back.subset(["c" * 16]).pixels.tolist() == [10, 20, 30]
    assert DefectTable.of(a) is a and DefectTable.of(SETS) == a
