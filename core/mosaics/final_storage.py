import shutil
from pathlib import Path
import numpy as np

from mosaics.types import Mosaic
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from pix import Image
from products import makes_image
from products.identity import MosaicChannelId, PresentationType
from decorators import node

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID
from file_io.types import FitsFilePath, TiffFilePath, XisfFilePath, WithSidecar
from metadata.classes import ImageMetadata


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=MosaicChannelId, display="Mosaic")
def store_mosaic_color_channel_xisf(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool,
    input: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """Persist the merged per-filter PixInsight mosaic master.

    Pipeline-agnostic: PI Master and PI Downscaled both call this
    with their own merged input. Different `input` (different
    upstream `merge_images` result) → different bound-args
    call_hash → different output filename. The framework's
    runtime tap-path recorder (in `compute_graph._record_pipeline_
    taps`) captures whatever path the body wrote to."""
    output = store_mosaic_color_channel_xisf.filename_fn(
        mosaic, filter, mock_pixinsight, input,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(input.filename, filter=filter)
    image.save_to(output, mock=mock_pixinsight)
    return WithSidecar(output)


@node(resources={"fastest_storage": 0.001})
@makes_image(identity=MosaicChannelId, display="Subframe")
def store_mosaic_color_channel_subframe(
    mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Persist the Subframe preview mosaic (autostretched 8-bit
    TIFF). Separate op because the algorithm (autostretch +
    write_tiff) differs from the XISF/FITS variants."""
    filename = input.filename
    assert filename.exists()
    output = store_mosaic_color_channel_subframe.filename_fn(
        mosaic, filter, input,
    )
    image = read_image(filename)
    autostretched = (autostretch(image) * 255.0).astype(np.uint8)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_tiff_image(output, autostretched, {"FILTER": filter}, tile=(256, 256))
    return WithSidecar(output)


@node
@makes_image(identity=MosaicChannelId, display="Mosaic")
def store_mosaic_color_channel_oss(
    mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """Persist the merged per-filter OSS mosaic master.
    Pipeline-agnostic — same reasoning as the XISF variant."""
    output = store_mosaic_color_channel_oss.filename_fn(
        mosaic, filter, input,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input.filename), str(output))
    return WithSidecar(FitsFilePath(output))
