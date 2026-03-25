from pathlib import Path

from file_io.types import (
    TiffFilePath,
    XisfFilePath,
)
from mosaics.types import Mosaic
from stacks.composite_stack import CompositeStack
from mosaics.types import Mosaic, Viewbox
from file_io.generic import get_content_hash


def mosaic_filename_tiff(mosaic: Mosaic, filter: str) -> TiffFilePath:
    return TiffFilePath(f"final_{mosaic.stable_hash}_{filter}.tiff")


def mosaic_filename_xisf(mosaic: Mosaic, filter: str) -> XisfFilePath:
    return XisfFilePath(f"final_{mosaic.stable_hash}_{filter}.xisf")


def tile_filename(mosaic: Mosaic, filter: str) -> TiffFilePath:
    filename = mosaic_filename_tiff(mosaic, filter)
    return TiffFilePath(
        filename.with_stem(filename.stem + "_autostretched").with_suffix(
            ".tiff"
        )
    )


def merged_filename_tiff(
    mosaic: Mosaic, image_0: Path, image_1: Path
) -> TiffFilePath:
    hash_0 = get_content_hash(image_0)
    hash_1 = get_content_hash(image_1)

    return TiffFilePath(f"merged_{mosaic.stable_hash}_{hash_0}_{hash_1}.tiff")


def merged_filename_xisf(
    mosaic: Mosaic, image_0: Path, image_1: Path
) -> XisfFilePath:
    hash_0 = get_content_hash(image_0)
    hash_1 = get_content_hash(image_1)

    return XisfFilePath(f"merged_{mosaic.stable_hash}_{hash_0}_{hash_1}.xisf")


def placed_filename(
    mosaic: Mosaic,
    stack: CompositeStack,
    filter: str,
    viewbox: Viewbox,
) -> TiffFilePath:
    return TiffFilePath(
        f"placed_{mosaic.stable_hash}_{stack.stable_hash}_{filter}.tiff"
    )
