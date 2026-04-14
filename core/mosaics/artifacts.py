from file_io.types import (
    FitsFilePath,
    TiffFilePath,
    XisfFilePath,
)
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic, Viewbox
from stacks.composite_stack import Stack


def mosaic_filename_tiff(mosaic: Mosaic, filter: str) -> TiffFilePath:
    return TiffFilePath(f"final_{mosaic.stable_hash}_{filter}.tiff")


def mosaic_filename_xisf(mosaic: Mosaic, filter: str) -> XisfFilePath:
    return XisfFilePath(f"final_{mosaic.stable_hash}_{filter}.xisf")


def mosaic_filename_fits(mosaic: Mosaic, filter: str) -> FitsFilePath:
    return FitsFilePath(f"final_{mosaic.stable_hash}_{filter}.fits")


def tile_filename(mosaic: Mosaic, filter: str) -> TiffFilePath:
    filename = mosaic_filename_tiff(mosaic, filter)
    return TiffFilePath(
        filename.with_stem(filename.stem + "_autostretched").with_suffix(
            ".tiff"
        )
    )


def merged_filename_tiff(
    mosaic: Mosaic, image_0: ImageMetadata, image_1: ImageMetadata
) -> TiffFilePath:
    return TiffFilePath(
        f"merged_{mosaic.stable_hash}_{image_0.stable_hash}_{image_1.stable_hash}.tiff"
    )


def merged_filename_xisf(
    mosaic: Mosaic, image_0: ImageMetadata, image_1: ImageMetadata
) -> XisfFilePath:
    return XisfFilePath(
        f"merged_{mosaic.stable_hash}_{image_0.stable_hash}_{image_1.stable_hash}.xisf"
    )


def merged_filename_fits(
    mosaic: Mosaic, image_0: ImageMetadata, image_1: ImageMetadata
) -> FitsFilePath:
    return FitsFilePath(
        f"merged_{mosaic.stable_hash}_{image_0.stable_hash}_{image_1.stable_hash}.fits"
    )


def placed_filename(
    mosaic: Mosaic,
    stack: Stack,
    filter: str,
    viewbox: Viewbox,
) -> TiffFilePath:
    return TiffFilePath(
        f"placed_{mosaic.stable_hash}_{stack.stable_hash}_{filter}.tiff"
    )
