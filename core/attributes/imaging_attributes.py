from collections import Counter
from dataclasses import dataclass
from typing import Any
import numpy as np
from beartype import beartype
from collections.abc import Iterable

from context import context
from metadata.classes import MetadataDict, FrameHash
from units import Celsius, Seconds


# A single run of frames should share one nominal exposure, but headers
# round-trip through float and may differ by µs (0.5 vs 0.500001). This
# tolerance lets a real run pass while still catching mixed exposures
# (e.g., 60s and 300s darks accidentally grouped).
EXPOSURE_TOLERANCE_SECONDS = 0.01


def get_unique_quality(values: Iterable[Any], allow_empty: bool = False) -> Any:
    distinct = set(values)
    if allow_empty and not distinct:
        return None
    assert len(distinct) == 1, f"non-unique values: {sorted(distinct)}"
    return next(iter(distinct))


def _camera_label(values: Iterable[Any]) -> str:
    """A human-facing NAME for the camera — never an identity.

    `INSTRUME` is free text the capture software chooses, so the frames of
    one physical sensor may legitimately disagree about it: on 2026-02-16 a
    reconnecting driver renamed one camera mid-session from "ASI Camera (1)"
    to "ZWO ASI2600MM Duo". Identity lives in `camera_id_hash`, derived from
    defect pixels and sensor geometry; this string only labels the UI and
    master filenames.

    So this must NEVER assert — a disagreement means a renamed driver, not a
    mixed-camera stack, and asserting would abort identification over
    cosmetics. Pick the most common spelling, ties broken alphabetically, so
    the label is deterministic across processes (a bare `set` iteration
    would not be — see project_determinism_guards).
    """
    names = [str(v) for v in values if v not in (None, "")]
    if not names:
        return ""
    counts = Counter(names)
    return min(counts, key=lambda n: (-counts[n], n))


def _aggregate_exposure(values: list[float], strict: bool = True) -> Seconds:
    """Reduce per-frame EXPOSURE to one value for a stack/run.

    Bias frames may lack EXPOSURE entirely; callers default missing to
    0.0 before passing in, so an all-zeros list (a bias run) returns 0.
    Tolerates small float jitter; asserts loud on real mismatch.

    `strict=False` is for FLATS: they are shot with auto-exposure chasing
    a changing sky, and calibration identification deliberately pools a
    flat run across EXPOSURE — a spread is expected there, and the mean
    is informational (stacking normalizes flats). A sky-flat run with
    9.96..10.34s exposures killed identify_calibration_stacks wholesale
    on 2026-09-05 until this exemption.
    """
    if not values:
        return Seconds(0.0)
    spread = max(values) - min(values)
    if strict:
        assert spread < EXPOSURE_TOLERANCE_SECONDS, (
            f"non-uniform exposure: spread={spread:.4f}s, values={sorted(set(values))}"
        )
    return Seconds(float(np.mean(values)))


@beartype
@dataclass(frozen=True)
class ImagingAttributes:
    image_type: str
    # DISPLAY LABEL ONLY (see `_camera_label`). Camera IDENTITY is
    # `BaseStack.camera_id_hash`, derived from defect pixels + sensor
    # geometry. Never match, group, or key on this string.
    camera: str
    filter: str | None
    mean_temperature: Celsius
    # Single exposure duration shared by every frame in this stack/run.
    # Mixed exposures break stacking — WBPP groups by EXPOSURE and emits
    # one master per group, AstroFlow's master cache assumes a single
    # output per stack — so we enforce uniformity at the type. Stack
    # identification clusters by exposure to keep this guarantee.
    exposure: Seconds
    integration: Seconds
    width_pixels: int
    height_pixels: int

    def __post_init__(self):
        # Fail-fast at the PRODUCER. `exposure` must be a scalar
        # Seconds (a float subclass), never a tuple/list — a recurring
        # bug where some live path constructs a 1-tuple `(30.0,)` that
        # bypasses beartype and only detonates much later in
        # resolve_calibration's replace(). Raising here makes the
        # constructing traceback the actual culprit. Runs on every
        # __init__ incl. replace(); pickle restore bypasses this (and
        # __init__) — if this never fires but the tuple still appears,
        # the producer is a deserialization boundary.
        for _name in ("exposure", "integration"):
            _v = getattr(self, _name)
            if isinstance(_v, bool) or not isinstance(_v, (int, float)):
                raise TypeError(
                    f"ImagingAttributes.{_name} must be a scalar Seconds, "
                    f"got {type(_v).__name__} {_v!r} "
                    f"(image_type={self.image_type!r}, camera={self.camera!r}, "
                    f"filter={self.filter!r})"
                )

    def __setstate__(self, state):
        # Pickle / copy.deepcopy restore bypasses __init__, beartype AND
        # __post_init__ (default dataclass unpickle = __new__ +
        # __dict__.update). This is the ONLY hook that runs on unpickle,
        # and the recurring tuple-`exposure` bug enters via a
        # deserialization boundary (Ray object store / diskcache /
        # ContextRef / a snapshot fixture). Validate here so the
        # *restoring* traceback names the live producer instead of
        # detonating later in resolve_calibration's replace().
        self.__dict__.update(state)
        for _name in ("exposure", "integration"):
            if _name in self.__dict__:
                _v = self.__dict__[_name]
                if isinstance(_v, bool) or not isinstance(_v, (int, float)):
                    raise TypeError(
                        f"ImagingAttributes.{_name} restored from pickle is "
                        f"{type(_v).__name__} {_v!r}, not a scalar Seconds "
                        f"(image_type={self.__dict__.get('image_type')!r}, "
                        f"camera={self.__dict__.get('camera')!r}, "
                        f"filter={self.__dict__.get('filter')!r}). Object "
                        f"serialized with the OLD tuple-exposure schema — "
                        f"this traceback is the deserializing producer."
                    )

    @classmethod
    def from_frames(
        cls, frames: Iterable[FrameHash], metadata: MetadataDict | None = None
    ) -> "ImagingAttributes":
        metadata_list = [
            i for i in metadata.values() if i.content_hash in frames
        ]

        # TODO: Unify these with above.
        def get_unique_quality(key: str, allow_empty: bool = False) -> Any:
            values = list(
                set(i.header[key] for i in metadata_list if key in i.header)
            )
            if allow_empty and len(values) == 0:
                return None
            assert len(values) == 1, (
                f"Key {key} had non-unique values {values} {[i.filename for i in metadata_list]}"
            )
            return values[0]

        image_type = get_unique_quality("IMAGETYP")
        camera = _camera_label(i.header.get("INSTRUME") for i in metadata_list)
        width_pixels = get_unique_quality("NAXIS1")
        height_pixels = get_unique_quality("NAXIS2")
        filter = get_unique_quality("FILTER", allow_empty=True)

        # Bias frames sometimes ship without EXPOSURE; treat absent as 0
        # so a bias run reduces to 0.0 cleanly.
        exposures = [
            float(i.header.get("EXPOSURE", 0.0)) for i in metadata_list
        ]
        exposure = _aggregate_exposure(
            exposures, strict="flat" not in str(image_type).lower()
        )
        integration = Seconds(sum(exposures))

        mean_temperature = Celsius(np.mean(
            [i.header.get("CCD-TEMP", 0) for i in metadata_list]
        ))

        return cls(
            image_type,
            camera,
            filter,
            mean_temperature,
            exposure,
            integration,
            width_pixels,
            height_pixels,
        )

    @classmethod
    def from_objects(
        cls, objects: Iterable["ImagingAttributes"]
    ) -> "ImagingAttributes":
        image_type = get_unique_quality([o.image_type for o in objects])
        camera = _camera_label(o.camera for o in objects)
        filter = list(set(o.filter for o in objects))
        mean_temperature = Celsius(np.mean([o.mean_temperature for o in objects]))
        exposure = _aggregate_exposure(
            [float(o.exposure) for o in objects],
            strict="flat" not in str(image_type).lower(),
        )
        integration = Seconds(np.sum([o.integration for o in objects]))
        width_pixels = get_unique_quality([o.width_pixels for o in objects])
        height_pixels = get_unique_quality([o.height_pixels for o in objects])

        return cls(
            image_type,
            camera,
            filter,
            mean_temperature,
            exposure,
            integration,
            width_pixels,
            height_pixels,
        )
