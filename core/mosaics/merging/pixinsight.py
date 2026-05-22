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
from previews.types import (
    ALL_IMPLEMENTATIONS,
    PIXINSIGHT, SUBFRAME, MASTER, DOWNSCALED_MASTER,
)
from previews.create_tile_pyramid import default_make_tile_pyramid
from decorators import node
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from mosaics.artifacts import merged_filename_tiff, merged_filename_xisf
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

def _progress_tile_path(mosaic: Mosaic, filter: str, mode: str) -> Path:
    """Per-(mosaic, filter, mode) tile pyramid that the web server serves
    while the canonical Master output is still in progress."""
    from context import context
    return (
        Path(context.output_dir) / "Mosaic" / str(mode)
        / f"progress_{mosaic.stable_hash}_{filter}.tiff"
    )

def _maybe_publish_progress(
    mode: str,
    mosaic: Mosaic,
    filter: str,
    impl: str,
    output_filename: Path,
    new_count: int,
) -> None:
    """Publish `output_filename` as this (mosaic, filter, mode, impl)'s
    visible progress IFF it's strictly more complete than the current
    best. "More complete" means strictly larger panel count — equal
    counts (e.g., the second strip catching up to the first) do NOT
    displace the current best, so the displayed image only ever gains
    panels.
    """
    from context import context

    best_key = (
        "mosaic_best_merge",
        mosaic.stable_hash,
        filter,
        str(mode),
        str(impl),
    )
    best = context.get(best_key)
    if best is not None and new_count <= best.get("panel_count", 0):
        return

    progress_path = _progress_tile_path(mosaic, filter, mode)
    try:
        default_make_tile_pyramid(Path(output_filename), progress_path)
    except Exception as e:
        log.warning("Failed to write progress tile pyramid",
                    output=str(output_filename), error=str(e))
        return

    # Same key shape that store_mosaic_color_channel_xisf eventually
    # overwrites with the final mosaic's thumbnail, so the handoff from
    # in-progress to final is automatic. Op-named: this is the in-progress
    # thumbnail produced by merge_images and overwritten by the final
    # store_mosaic_color_channel_xisf — both surface via the same key
    # because the viewer cares about "the mosaic's current thumbnail",
    # not which step produced it.
    thumbnail_key = MosaicChannelId(mosaic=mosaic, filter=filter).presentation_cache_key(
        THUMBNAIL, "store_mosaic_color_channel_xisf",
    )
    try:
        from previews.operations import resize_array, autostretch, array_to_png
        arr = read_image(output_filename)
        png = array_to_png(autostretch(np.clip(resize_array(arr, 2048), 0.0, 1.0)))
        context[thumbnail_key] = png
    except Exception as e:
        log.warning("Failed to render progress thumbnail",
                    output=str(output_filename), error=str(e))

    context[best_key] = {"path": str(output_filename), "panel_count": new_count}

@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(PIXINSIGHT, [SUBFRAME], merged_filename_tiff,
             identity=MergedImageId)
@makes_image(PIXINSIGHT, [MASTER, DOWNSCALED_MASTER], merged_filename_xisf,
             identity=MergedImageId)
def merge_images(
    mode: str,
    mosaic: Mosaic,
    filter: str,
    impl: str,
    mock_pixinsight: bool,
    input_0: ImageMetadata,
    input_1: ImageMetadata,
) -> WithSidecar:
    path_0 = input_0.filename.resolve(strict=True)
    path_1 = input_1.filename.resolve(strict=True)

    # The op has TWO @makes_image registrations (Subframe + Master);
    # each baked a per-mode filename_fn. Pick the right one by `mode`.
    output_fn = merge_images.filename_fns[(MergedImageId, mode)]
    output_filename = output_fn(mosaic, filter, input_0, input_1)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    if mode == SUBFRAME:
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
    else:
        images = Image.open_many([path_0, path_1], filters=[filter, filter])
        result = PhotometricMosaic(images)
        result.save_to(output_filename, mock=mock_pixinsight)
        # PhotometricMosaic doesn't preserve FILTER from the inputs and
        # writes a placeholder ("none"); stamp the canonical FILTER and
        # a stable hash back onto the output so downstream consumers
        # (compute_image_metadata, MergeStepId chaining) see real values.
        update_header(Path(output_filename), {
            "FILTER": filter,
            "STBLHASH": hash_list([input_0.stable_hash, input_1.stable_hash]),
        })

    new_count = _panel_count(input_0) + _panel_count(input_1)
    _set_panel_count(output_filename, new_count)

    # SUBFRAME is a fast preview path with no "in progress" semantics —
    # there's no final Master mosaic to hand off to, so the progress-
    # publication machinery only applies to MASTER / DOWNSCALED_MASTER.
    if mode != SUBFRAME:
        _maybe_publish_progress(mode, mosaic, filter, impl, Path(output_filename), new_count)

    log.info("Merge images complete", output=output_filename, panel_count=new_count)
    return WithSidecar(output_filename)
