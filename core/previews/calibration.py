import numpy as np

from metadata.classes import ImageMetadata


# TODO: Average flat frames before applying them.
def calibrate_image(light_metadata: ImageMetadata | None,
                    light_image: np.ndarray | None,
                    bias_metadata: ImageMetadata | None,
                    bias_image: np.ndarray | None,
                    dark_metadata: ImageMetadata | None,
                    dark_image: np.ndarray | None,
                    flat_metadata: ImageMetadata | None,
                    flat_image: np.ndarray | None, rotate: bool) -> np.ndarray:

  assert light_image is not None

  if bias_image is None or dark_image is None or flat_image is None:
    if rotate:
      light_image = np.rot90(light_image, 2)
    return light_image

  assert light_metadata is not None and dark_metadata is not None and flat_metadata is not None
  assert light_image is not None and dark_image is not None and flat_image is not None

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
  assert flat_image is not None
  image = np.divide(image,
                    flat_image,
                    out=np.zeros_like(image),
                    where=flat_image != 0)

  if rotate:
    image = np.rot90(image, 2)

  return image
