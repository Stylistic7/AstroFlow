from pathlib import Path
import numpy as np

from mosaics.types import Mosaic
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from decorators import node

from file_io.types import TiffFilePath, WithSidecar
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


# `store_mosaic_color_channel_oss` was deleted: it was a vestigial
# `shutil.copy2` of `merge_images_oss`'s output. The OSS variadic merge IS the
# finished channel, so `merge_images_oss` now carries `display="Mosaic",
# final=True` directly and there is no separate store/copy step.
