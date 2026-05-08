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


def _aggregate_exposure(values: list[float]) -> Seconds:
    """Reduce per-frame EXPOSURE to one value for a stack/run.

    Bias frames may lack EXPOSURE entirely; callers default missing to
    0.0 before passing in, so an all-zeros list (a bias run) returns 0.
    Tolerates small float jitter; asserts loud on real mismatch.
    """
    if not values:
        return Seconds(0.0)
    spread = max(values) - min(values)
    assert spread < EXPOSURE_TOLERANCE_SECONDS, (
        f"non-uniform exposure: spread={spread:.4f}s, values={sorted(set(values))}"
    )
    return Seconds(float(np.mean(values)))


@beartype
@dataclass(frozen=True)
class ImagingAttributes:
    image_type: str
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
        camera = get_unique_quality("INSTRUME")
        width_pixels = get_unique_quality("NAXIS1")
        height_pixels = get_unique_quality("NAXIS2")
        filter = get_unique_quality("FILTER", allow_empty=True)

        # Bias frames sometimes ship without EXPOSURE; treat absent as 0
        # so a bias run reduces to 0.0 cleanly.
        exposures = [
            float(i.header.get("EXPOSURE", 0.0)) for i in metadata_list
        ]
        exposure = _aggregate_exposure(exposures)
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
        camera = get_unique_quality([o.camera for o in objects])
        filter = list(set(o.filter for o in objects))
        mean_temperature = Celsius(np.mean([o.mean_temperature for o in objects]))
        exposure = _aggregate_exposure([float(o.exposure) for o in objects])
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
