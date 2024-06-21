from typing import Optional, Any
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore
import xxhash
import numpy as np

from xisf import XISF  # type: ignore


def read_xisf_header(filename: Path | str) -> dict:
  xisf = XISF(filename)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()
  fits_header = ims_meta[0]['FITSKeywords']
  header = {
      k: file_meta[k]['value'] for k in file_meta.keys()
  } | {
      k: fits_header[k][0]['value'] for k in fits_header.keys()
  }
  return header


def read_xisf_image(filename: Path | str) -> np.ndarray:
  xisf = XISF(str(filename))
  return xisf.read_image(0).squeeze().astype(np.float32)


def hash_of_xisf_image_data(filename: Path | str) -> Optional[str]:
  xisf = XISF(filename)
  return xxhash.xxh64(xisf.read_image(0)).hexdigest()


# TODO: Add a round-trip unit test.
def convert_xisf_to_fits(xisf_filename: Path | str,
                         fits_filename: Path | str) -> None:
  xisf = XISF(xisf_filename)
  image = xisf.read_image(0).squeeze()
  img_meta = xisf.get_images_metadata()[0]
  fits_meta = img_meta['FITSKeywords']

  hdu = pyfits.PrimaryHDU(image)
  for k, values in fits_meta.items():
    for v in values:
      try:
        value = eval(v['value'])
      except:
        value = v['value']

      hdu.header[k] = (value, v['comment'])

  hdu.writeto(fits_filename)


def convert_fits_to_xisf(fits_filename: Path | str,
                         xisf_filename: Path | str,
                         add_noise: bool = False,
                         metadata: dict | None = None) -> None:
  fits = pyfits.open(str(fits_filename))
  header = fits[0].header
  image = np.expand_dims(fits[0].data, -1)
  fits.close()

  if add_noise:
    image += np.random.randint(low=0,
                               high=2,
                               size=image.shape,
                               dtype=image.dtype)

  file_meta = {}
  if metadata:
    for k, v in metadata.items():
      file_meta[k] = {"id": k, "type": "String", "value": v}

  fits_meta = {}
  for k, v in header.items():
    fits_meta[k] = [{'value': str(v), 'comment': ''}]

  ims_meta = {'FITSKeywords': fits_meta}

  XISF.write(xisf_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta,
             xisf_metadata=file_meta)


def add_xisf_metadata(filename, dest_filename, metadata):
  """Adds the given metadata to the file.
  
  filename and dest_filename can be the same.
  """
  xisf = XISF(filename)
  image = xisf.read_image(0)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()

  print('### metadata is', metadata)

  for k, v in metadata.items():
    file_meta[k] = {
        "id": k,
        "type": "String",
        "value": v,
    }

  XISF.write(dest_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)
