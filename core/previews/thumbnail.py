import numpy as np

from previews.operations import autostretch


# TODO: Average flat frames before applying them.
def make_thumbnail(light_frame, light_image, bias_frame, bias_image, dark_frame,
                   dark_image, flat_frame, flat_image, rotate):

  if bias_image is None or dark_image is None or flat_image is None:
    print('### no calibration images')
    if rotate:
      light_image = np.rot90(light_image, 2)
    return autostretch(light_image)

  image_sizes = {
      light_image.shape, dark_image.shape, bias_image.shape, flat_image.shape
  }
  assert len(image_sizes) == 1, (
      f"Shapes don't match: light {light_image.shape}, "
      f"bias {bias_image.shape}, "
      f"dark {dark_image.shape}, "
      f"flat {flat_image.shape}")

  light_dark_scale = light_frame.header['EXPOSURE'] / dark_frame.header[
      'EXPOSURE']
  flat_dark_scale = flat_frame.header['EXPOSURE'] / dark_frame.header['EXPOSURE']

  image = (light_image -
           bias_image) - light_dark_scale * (dark_image - bias_image)
  flat_image = (flat_image -
                bias_image) - flat_dark_scale * (dark_image - bias_image)
  image = np.divide(image,
                    flat_image,
                    out=np.zeros_like(image),
                    where=flat_image != 0)

  if rotate:
    image = np.rot90(image, 2)

  return autostretch(np.clip(image, 0.0, 1.0))
