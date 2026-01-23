import ray
from pathlib import Path
import numpy as np

from context import context
from mosaics.artifacts import mosaic_filename_tiff, mosaic_filename_xisf
from file_io.generic import read_image, read_header, copy_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from pix.decorators import PixInsight
from pix.classes import ImageList
from pix.tasks import SaveImages
from products import makes_image
from decorators import cached, create_thumbnail, measure_performance
from mosaics.types import Mosaic
from previews.types import QualityLevel


@PixInsight()
def save_uncompressed(images: ImageList, output_filenames: list[Path]):
  SaveImages(images, output_filenames, compression=False)


@makes_image(Mosaic).as_master(mosaic_filename_xisf)
@cached.remote(resources={'PixInsight': 1})
@create_thumbnail
@measure_performance
def store_mosaic_color_channel_master(mosaic: Mosaic, filter: str,
                                      input: Path) -> Path:
  output = store_mosaic_color_channel_master.filename_fn(mosaic, filter)
  output.parent.mkdir(parents=True, exist_ok=True)
  if context.get('mock_pixinsight', False):
    print(f'### store_mosaic_color_channel_master mocking PixInsight; '
          f'copying {input=} to {output=}')
    copy_image(input, output)
  else:
    save_uncompressed([input], [output])

  return output


@makes_image(Mosaic).as_subframe(mosaic_filename_tiff)
@cached.remote(resources={'fastest_storage': 0.001})
@create_thumbnail
@measure_performance
def store_mosaic_color_channel_subframe(mosaic: Mosaic, filter: str,
                                        input: Path) -> Path:
  output = store_mosaic_color_channel_subframe.filename_fn(mosaic, filter)
  print(f'### store_mosaic_color_channel_subframe: {filter=} {input=} {output=}')
  image = read_image(input)
  autostretched = (autostretch(image) * 255.0).astype(np.uint8)
  write_tiff_image(output, autostretched, {'FILTER': filter})

  return output


@ray.remote
def store_mosaic_color_channel(mode: QualityLevel, mosaic: Mosaic, filter: str,
                               input: Path) -> Path:
  print(f'### store_mosaic_color_channel {mode=} {str(mosaic)} {filter=} {input=}')               
  if mode == QualityLevel.SUBFRAME:
    return store_mosaic_color_channel_subframe.remote(mosaic, filter, input)
  else:
    return store_mosaic_color_channel_master.remote(mosaic, filter, input)
