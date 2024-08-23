import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
from xisf import XISF  # type: ignore
import numpy as np
from typing import Dict, Any, Optional


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

  hdu.writeto(fits_filename, overwrite=True)


def convert_fits_to_xisf(fits_filename: Path | str,
                         xisf_filename: Path | str,
                         add_noise: bool = False,
                         metadata: Optional[Dict] = None) -> None:
  if metadata is None:
    metadata = {}

  fits = pyfits.open(str(fits_filename))
  header = fits[0].header
  image = np.expand_dims(fits[0].data, -1)
  fits.close()

  if add_noise:
    image += np.random.randint(low=0,
                               high=2,
                               size=image.shape,
                               dtype=image.dtype)

  file_meta: Dict[str, Any] = {}

  fits_meta: Dict[str, Any] = {}
  for k, v in header.items():
    fits_meta[k] = [{'value': str(v), 'comment': ''}]
  for k, v in metadata.items():
    fits_meta[k] = [{'value': str(v), 'comment': ''}]

  ims_meta = {'FITSKeywords': fits_meta}

  XISF.write(xisf_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta,
             xisf_metadata=file_meta)
