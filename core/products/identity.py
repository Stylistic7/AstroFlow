import dataclasses
import math
from dataclasses import dataclass
from enum import StrEnum

from catalogs import position_string
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic, Viewbox
from stacks.composite_stack import BaseStack, Stack
from stacks.run import Run
from targets.types import Target


class PresentationType(StrEnum):
    IMAGE = "Image"
    THUMBNAIL = "Thumbnail"
    TILE_PYRAMID = "TilePyramid"
    VIDEO = "Video"


@dataclass(frozen=True)
class ObjectId:
    def cache_key(self) -> str:
        parts = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if hasattr(v, 'stable_hash'):
                parts.append(v.stable_hash)
            else:
                parts.append(str(v))
        return "_".join(parts)

    def presentation_cache_key(
        self,
        presentation: PresentationType,
        object_type: type,
        level: "QualityLevel",
        implementation: "Implementation",
    ) -> tuple:
        return (
            str(presentation),
            object_type.__name__,
            str(level),
            str(implementation),
            self.cache_key(),
        )


@dataclass(frozen=True)
class StackId(ObjectId):
    stack: BaseStack

    def filename_stem(self) -> str:
        stack = self.stack
        if "bias" in stack.image_type:
            return (
                f"bias_{stack.stable_hash}_{stack.camera}_"
                f"{stack.mean_temperature:.0f}C_{stack.stable_hash}"
            )
        elif "dark" in stack.image_type:
            return (
                f"dark_{stack.stable_hash}_{stack.camera}_"
                f"{stack.mean_temperature:.0f}C_{stack.exposure:.0f}s_"
                f"{stack.stable_hash}"
            )
        elif "flat" in stack.image_type:
            return (
                f"flat_{stack.stable_hash}_{stack.camera}_{stack.filter}_"
                f"{stack.stable_hash}"
            )
        elif isinstance(stack, Stack) and "light" in stack.image_type:
            ra_dec = position_string.ra_dec_string(*stack.center)
            return (
                f"light_{stack.stable_hash}_{stack.name}_{ra_dec}_"
                f"{stack.camera}_{stack.image_scale:.02f}aspp_"
                f"{math.degrees(stack.rotation_angle):.2f}deg_{stack.filter}"
            )

        raise ValueError(f'unknown image_type "{stack.image_type}"')


@dataclass(frozen=True)
class TargetId(ObjectId):
    target: Target


@dataclass(frozen=True)
class MosaicChannelId(ObjectId):
    mosaic: Mosaic
    filter: str

    def filename_stem(self) -> str:
        return f"final_{self.mosaic.stable_hash}_{self.filter}"


@dataclass(frozen=True)
class MosaicId(ObjectId):
    mosaic: Mosaic

    def filename_stem(self) -> str:
        return f"final_{self.mosaic.stable_hash}_None"


@dataclass(frozen=True)
class MergeStepId(ObjectId):
    mosaic: Mosaic
    input_0: ImageMetadata
    input_1: ImageMetadata

    def filename_stem(self) -> str:
        return f"merged_{self.mosaic.stable_hash}_{self.input_0.stable_hash}_{self.input_1.stable_hash}"


@dataclass(frozen=True)
class MergeProgressId(ObjectId):
    mosaic: Mosaic
    filter: str


@dataclass(frozen=True)
class PlacedImageId(ObjectId):
    mosaic: Mosaic
    stack: Stack
    filter: str
    viewbox: Viewbox

    def filename_stem(self) -> str:
        return f"placed_{self.mosaic.stable_hash}_{self.stack.stable_hash}_{self.filter}"


@dataclass(frozen=True)
class CalibratedStackId(ObjectId):
    stack: Stack
    substack: Run
    flat_stack: Run


@dataclass(frozen=True)
class UncorrectedRunId(ObjectId):
    stack: Stack
    substack: Run
