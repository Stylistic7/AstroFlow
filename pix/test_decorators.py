import unittest
import tempfile
import os

from .decorators import PixInsight
from .abstract_image import AbstractImage
from .blurxterminator import BlurXTerminator
from .starxterminator import StarXTerminator


# This is my best take on the API. The `deconvolved` is not used directly
# anywhere inside the function, but the tracer should see it and save it to
# disk after the function is done. End users may have a file where they can
# write (and share) many such functions, and then just associate them with
# stacks / targets / mosaics by name. These are legit Python functions, not
# fragments or other bastardizations.
@PixInsight(dry_run=True)
def bxt_only(image):
  deconvolved = BlurXTerminator(image)


@PixInsight(dry_run=True)
def bxt_and_sxt(image):
  deconvolved = BlurXTerminator(image)
  starless, stars = StarXTerminator(deconvolved)


class TestDecorators(unittest.TestCase):

  def test_bxt_only(self):
    temp_dir = tempfile.TemporaryDirectory()
    filename = os.path.join(temp_dir.name, 'foo.xisf')
    with open(filename, 'w') as f:
      f.write('foo')

    bxt_only(filename)

  def test_bxt_and_sxt(self):
    temp_dir = tempfile.TemporaryDirectory()
    filename = os.path.join(temp_dir.name, 'foo.xisf')
    with open(filename, 'w') as f:
      f.write('foo')

    bxt_and_sxt(filename)


if __name__ == "__main__":
  unittest.main()
