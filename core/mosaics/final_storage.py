import shutil
from pathlib import Path
import numpy as np

from mosaics.types import Mosaic
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from decorators import node

from file_io.types import FitsFilePath, TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata


@node(
    display="Subframe",
    final=True,
    resources={"fastest_storage": 0.001},
)
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


@node(
    display="Mosaic",
    final=True,
)
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
