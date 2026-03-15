from dataclasses import dataclass
from typing import Any
import numpy as np
from beartype import beartype
from collections.abc import Iterable

from context import context
from metadata.classes import MetadataDict, FrameHash


def get_unique_quality(values: Iterable[str], allow_empty: bool = False) -> Any:
    if allow_empty and len(values) == 0:
        return None
    assert len(values) == 1
    return values[0]


@beartype
@dataclass(frozen=True)
class ImagingAttributes:
    image_type: str
    camera: str
    filter: str | None
    mean_temperature: float
    exposure_s: tuple[float, ...]
    integration_s: float
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

        def get_quality_set(key: str) -> list[Any]:
            return list(
                set(i.header[key] for i in metadata_list if key in i.header)
            )

        image_type = get_unique_quality("IMAGETYP")
        camera = get_unique_quality("INSTRUME")
        width_pixels = get_unique_quality("NAXIS1")
        height_pixels = get_unique_quality("NAXIS2")
        exposure_s = tuple(float(e) for e in get_quality_set("EXPOSURE"))
        filter = get_unique_quality("FILTER", allow_empty=True)

        integration_s = float(
            sum(
                i.header["EXPOSURE"]
                for i in metadata_list
                if "EXPOSURE" in i.header
            )
        )

        mean_temperature = np.mean(
            [i.header.get("CCD-TEMP", 0) for i in metadata_list]
        )

        return cls(
            image_type,
            camera,
            filter,
            mean_temperature,
            exposure_s,
            integration_s,
            width_pixels,
            height_pixels,
        )

    @classmethod
    def from_objects(
        cls, objects: Iterable["ImagingAttributes"]
    ) -> "ImagingAttributes":
        image_type = get_unique_quality(o.image_type for o in objects)
        camera = get_unique_quality(o.camera for o in objects)
        filter = list(set(o.filter for o in objects))
        mean_temperature = np.mean(o.mean_temperature for o in objects)
        exposure_s = tuple({v for o in objects for v in o.exposure_s})
        integration_s = np.sum(o.integration_s for o in objects)
        width_pixels = get_unique_quality(o.width_pixels for o in objects)
        height_pixels = get_unique_quality(o.height_pixels for o in objects)

        return cls(
            image_type,
            camera,
            filter,
            mean_temperature,
            exposure_s,
            integration_s,
            width_pixels,
            height_pixels,
        )
