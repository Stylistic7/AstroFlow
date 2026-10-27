"""`stable_center` is the oldest frame's own solved centre, carried up by the
oldest member at every level — like `stable_hash`. `center` stays the centroid.
Pure: no Ray, no context."""
import datetime
from pathlib import Path

import numpy as np
import pytest

from attributes.astrometric_attributes import AstrometricAttributes, covering_polygon
from metadata.classes import ImageMetadata, MetadataDict
from plate_solving import PlateSolution, get_image_scale, get_rotation_angle
from testing.test_stacks import make_composite_stack
from units import Degrees


def _frame(content_hash: str, date_obs: datetime.datetime, ra: float, dec: float) -> ImageMetadata:
    nx, ny, scale_deg = 9576, 6388, 2.21 / 3600.0
    header = {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": ra, "CRVAL2": dec, "CRPIX1": nx / 2, "CRPIX2": ny / 2,
        "CD1_1": scale_deg, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": scale_deg,
    }
    solution = PlateSolution(
        header=header, num_stars=500, center=(Degrees(ra), Degrees(dec)),
        image_scale=get_image_scale(header), rotation_angle=get_rotation_angle(header),
    )
    return ImageMetadata(
        content_hash=content_hash, stable_hash=content_hash,
        filename=Path(f"/nonexistent/{content_hash}.fit"),
        header={"DATE-OBS": date_obs, "FILTER": "H"},
        width_pixels=nx, height_pixels=ny, channels=1, dtype=np.dtype(np.uint16),
        plate_solution=solution,
    )


def test_from_frames_anchors_on_the_oldest_frame_not_the_centroid():
    night = datetime.datetime(2026, 10, 12, 20, 0, 0)
    metadata = MetadataDict({
        "later": _frame("later", night + datetime.timedelta(days=1), 5.30, 62.60),
        "oldest": _frame("oldest", night, 4.90, 62.40),
        "latest": _frame("latest", night + datetime.timedelta(days=2), 5.30, 62.60),
    })
    attrs = AstrometricAttributes.from_frames(["later", "oldest", "latest"], metadata)
    assert attrs.stable_center == pytest.approx((4.90, 62.40))
    assert attrs.center[0] > 5.0 and attrs.center[1] > 62.5     # the centroid drifted toward the later frames
    assert attrs.center != attrs.stable_center


def test_from_objects_takes_the_anchor_members_stable_center():
    night = datetime.datetime(2026, 10, 12, 20, 0, 0)
    a = AstrometricAttributes.from_frames(["a"], MetadataDict({"a": _frame("a", night, 10.0, 60.0)}))
    b = AstrometricAttributes.from_frames(["b"], MetadataDict({"b": _frame("b", night, 10.4, 60.2)}))
    composite = AstrometricAttributes.from_objects([b, a], primary=b, combine=covering_polygon, anchor=a)
    assert composite.stable_center == a.stable_center
    assert composite.center == pytest.approx((10.2, 60.1), abs=0.01)


def test_target_inherits_the_oldest_stacks_stable_center():
    from targets.types import Target
    old = make_composite_stack(
        stable_hash="old", oldest_date=datetime.datetime(2026, 10, 7, 20, 0, 0),
        newest_date=datetime.datetime(2026, 10, 7, 21, 0, 0),
        center=(Degrees(50.1), Degrees(50.1)), stable_center=(Degrees(50.1), Degrees(50.1)),
        content_hash="old_content", filter="H",
    )
    new = make_composite_stack(
        stable_hash="new", oldest_date=datetime.datetime(2026, 10, 25, 20, 0, 0),
        newest_date=datetime.datetime(2026, 10, 25, 21, 0, 0),
        center=(Degrees(49.9), Degrees(49.9)), stable_center=(Degrees(49.9), Degrees(49.9)),
        content_hash="new_content", filter="O",
    )
    target = Target.from_stacks([new, old])
    assert target.stable_hash == "old"
    assert target.stable_center == old.stable_center
    assert target.center == pytest.approx((50.0, 50.0), abs=0.01)
