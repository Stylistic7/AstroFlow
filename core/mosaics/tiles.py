from pathlib import Path

from mosaics.types import Mosaic
from pix.classes import ImageList
from pix.decorators import PixInsight
from pix.tasks import SaveAs
from decorators import cached


@PixInsight()
def save_as_png(images: ImageList) -> ImageList:
    output_filenames = [Path(i.filename).with_suffix(".png") for i in images]
    return SaveAs(images, output_filenames)


@cached.remote
def save_as_tiles(mosaic: Mosaic, filter: str, output_dir: Path) -> None:
    filename = (
        Path(output_dir)
        / "Mosaics"
        / mosaic.name
        / f"final_merged_{mosaic.stable_hash}_{filter}.fits"
    )
    save_as_png([filename])
