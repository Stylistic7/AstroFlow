"""`previews/run_video.py`: frame order and the burnt-in annotation."""
from types import SimpleNamespace

import numpy as np

from previews.run_video import annotate, chronological


def test_frames_come_out_oldest_first():
    frames = [("c", "2025-10-05T03:00:00"), ("a", "2024-09-30T04:00:00"), ("b", "2024-09-30T04:10:00")]
    run = SimpleNamespace(frames=tuple(h for h, _ in frames),
                          metadata={h: SimpleNamespace(header={"DATE-OBS": d}) for h, d in frames})
    assert [h for h, _ in chronological(run)] == ["a", "b", "c"]


def test_annotation_marks_excluded_frames_and_keeps_the_frame_shape():
    im = np.full((120, 200), 90, np.uint8)
    kept = annotate(im.copy(), "2024-09-30 04:00   5000 stars", True)
    excluded = annotate(im.copy(), "2024-09-30 04:00   5000 stars", False)
    assert kept.shape == excluded.shape == (120, 200) and kept.dtype == np.uint8
    assert kept[:20, :80].max() > 200, "the text is burnt in (anti-aliased, so not exactly 255)"
    # The bar runs the full width; its left end carries the label's outline, so
    # judge it clear of the text.
    assert kept[-2:, :].max() == 90 and excluded[-2:, 150:].min() == 200, "only the excluded frame has the bar"
