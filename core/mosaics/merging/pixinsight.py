import os
import ray
import numpy as np
import structlog
from beartype import beartype
from pathlib import Path
from dataclasses import dataclass

from file_io.generic import read_header, read_image, update_header
from file_io.sidecar import get_section, set_section
from pix import Image
from pix.tasks import PhotometricMosaic
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from previews.create_tile_pyramid import default_make_tile_pyramid
from decorators import node
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from products import makes_image
from products.identity import MosaicChannelId, MergedImageId, PresentationType
from hashing import hash_list

log = structlog.get_logger()

THUMBNAIL = PresentationType.THUMBNAIL

@dataclass(frozen=True)
class MergedImage:
    pass

def _panel_count(image_metadata: ImageMetadata) -> int:
    """Number of mosaic panels contained in this intermediate.

    Placed images set `panel_count=1` in their sidecar; merges record
    the sum of their inputs. `discard_sidecar` is surgical about which
    sections it drops, so `panel_count` survives across producer reruns.
    Missing entry defaults to 1, keeping the chain monotonic for any
    pre-existing files left over from before this mechanism existed.
    """
    return get_section(image_metadata.filename, "panel_count") or 1

def _set_panel_count(path, count: int) -> None:
    set_section(Path(path), "panel_count", count)



@node(resources={"fastest_storage": 0.001})
@makes_image(identity=MergedImageId)
def merge_images_subframe(
    mosaic: Mosaic, filter: str,
    input_0: ImageMetadata, input_1: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Cheap pixel-overlap blend for the Subframe preview pipeline.
    No PixInsight; pure numpy averaging in the overlap region."""
    path_0 = input_0.filename.resolve(strict=True)
    path_1 = input_1.filename.resolve(strict=True)

    output_filename = merge_images_subframe.filename_fn(mosaic, filter, input_0, input_1)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    stable_hash = hash_list([input_0.stable_hash, input_1.stable_hash])
    filters = set(read_header(i)["FILTER"] for i in [path_0, path_1])
    assert len(filters) == 1
    filter = next(iter(filters))

    image_0 = read_image(path_0)
    image_1 = read_image(path_1)

    assert not np.isnan(np.min(image_0))
    assert not np.isnan(np.min(image_1))
    overlap = np.where((image_0 > 0) & (image_1 > 0))
    output_image = np.maximum(image_0, image_1)
    output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
    assert not np.isnan(np.min(output_image))

    write_tiff_image(Path(output_filename), output_image, {"FILTER": filter, "STBLHASH": stable_hash})

    new_count = _panel_count(input_0) + _panel_count(input_1)
    _set_panel_count(output_filename, new_count)
    log.info("Merge images (subframe) complete", output=output_filename, panel_count=new_count)
    return WithSidecar(output_filename)


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=MergedImageId)
def merge_images(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool,
    input_0: ImageMetadata, input_1: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """PhotometricMosaic merge of two placed inputs. Pipeline-agnostic:
    PI Master and PI Downscaled both call this with their own placed-
    image inputs; the inputs' content_hashes differ between pipelines,
    so the output filename (via merged_filename_xisf) lands in
    different files automatically.

    `_maybe_publish_progress` uses the input filenames to derive a
    per-pipeline progress key — the cache key includes the placed-
    image content hashes, which already distinguish full-res from
    downscaled mosaic progress without an explicit mode arg.
    """
    path_0 = input_0.filename.resolve(strict=True)
    path_1 = input_1.filename.resolve(strict=True)
    output_filename = merge_images.filename_fn(mosaic, filter, input_0, input_1)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    images = Image.open_many([path_0, path_1], filters=[filter, filter])
    result = PhotometricMosaic(images)
    result.save_to(output_filename, mock=mock_pixinsight)
    # PhotometricMosaic writes "none" for FILTER; stamp the real value back.
    update_header(Path(output_filename), {
        "FILTER": filter,
        "STBLHASH": hash_list([input_0.stable_hash, input_1.stable_hash]),
    })

    new_count = _panel_count(input_0) + _panel_count(input_1)
    _set_panel_count(output_filename, new_count)

    log.info("Merge images complete", output=output_filename, panel_count=new_count)
    return WithSidecar(output_filename)
