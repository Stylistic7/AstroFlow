from pathlib import Path
from shapely import box  # type: ignore

from file_io.types import (
    TiffFilePath,
    XisfFilePath,
    ContentHashableFile,
    WithSidecar,
)
from mosaics.types import Mosaic
from customization.types import CustomizedStack
from mosaics.types import Mosaic, Viewbox
from file_io.generic import get_content_hash


def mosaic_filename_tiff(mosaic: Mosaic, filter: str) -> WithSidecar:
    return WithSidecar(
        TiffFilePath(f"final_{mosaic.stable_hash}_{filter}.tiff")
    )


def mosaic_filename_xisf(mosaic: Mosaic, filter: str) -> WithSidecar:
    return WithSidecar(
        XisfFilePath(f"final_{mosaic.stable_hash}_{filter}.xisf")
    )


def tile_filename(mosaic: Mosaic, filter: str) -> TiffFilePath:
    filename = Path(mosaic_filename_tiff(mosaic, filter))
    return TiffFilePath(
        filename.with_stem(filename.stem + "_autostretched").with_suffix(
            ".tiff"
        )
    )


def merged_filename_tiff(
    mosaic: Mosaic, image_0: WithSidecar, image_1: WithSidecar
) -> WithSidecar:
    hash_0 = get_content_hash(image_0)
    hash_1 = get_content_hash(image_1)

    return WithSidecar(
        TiffFilePath(f"merged_{mosaic.stable_hash}_{hash_0}_{hash_1}.tiff")
    )


def merged_filename_xisf(
    mosaic: Mosaic, image_0: WithSidecar, image_1: WithSidecar
) -> WithSidecar:
    hash_0 = get_content_hash(image_0)
    hash_1 = get_content_hash(image_1)

    return WithSidecar(
        XisfFilePath(f"merged_{mosaic.stable_hash}_{hash_0}_{hash_1}.xisf")
    )


def placed_filename(
    mosaic: Mosaic,
    stack: CustomizedStack,
    filter: str,
    viewbox: Viewbox,
) -> TiffFilePath:
    return TiffFilePath(
        f"placed_{mosaic.stable_hash}_{stack.stable_hash}_{filter}.tiff"
    )
