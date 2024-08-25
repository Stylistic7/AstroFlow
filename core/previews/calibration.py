import numpy as np


# TODO: Average flat frames before applying them.
def calibrate_image(light_metadata, light_image, bias_metadata, bias_image,
                    dark_metadata, dark_image, flat_metadata, flat_image,
                    rotate):

  if bias_image is None or dark_image is None or flat_image is None:
    print('### no calibration images')
    if rotate:
      light_image = np.rot90(light_image, 2)
    return light_image

  image_sizes = {
      light_image.shape, dark_image.shape, bias_image.shape, flat_image.shape
  }
  assert len(image_sizes) == 1, (
      f"Shapes don't match: light {light_image.shape}, "
      f"bias {bias_image.shape}, "
      f"dark {dark_image.shape}, "
      f"flat {flat_image.shape}")

  light_dark_scale = light_metadata.header['EXPOSURE'] / dark_metadata.header[
      'EXPOSURE']
  flat_dark_scale = flat_metadata.header['EXPOSURE'] / dark_metadata.header[
      'EXPOSURE']

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

  return image
