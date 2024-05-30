import unittest
import tempfile
import os

from .decorators import PixInsight
from .abstract_image import AbstractImage
from .blurxterminator import BlurXTerminator


# This is my best take on the API. The `deconvolved` is not used directly
# anywhere inside the function, but the tracer should see it and save it to
# disk after the function is done. End users may have a file where they can
# write (and share) many such functions, and then just associate them with
# stacks / targets / mosaics by name. These are legit Python functions, not
# fragments or other bastardizations.
@PixInsight()
def post_process_stack(image):
  deconvolved = BlurXTerminator(image)


class TestDecorators(unittest.TestCase):

  def test_post_process_stack(self):
    temp_dir = tempfile.TemporaryDirectory()
    filename = os.path.join(temp_dir.name, 'foo.xisf')
    with open(filename, 'w') as f:
      f.write('foo')

    post_process_stack(filename)


if __name__ == "__main__":
  unittest.main()
