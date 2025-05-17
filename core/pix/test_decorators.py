import unittest
import tempfile
import os
import shutil
from typing import Any

import context

from pix.classes import FilenameList, ImageList
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, StarXTerminator, StarAlignment, ChannelCombination

# TODO: Automatically rename images by looking at the stem they have in common.
# Append the function's name.
#
# Example: a single input file named /a/b/foo.xisf, processed by a function called
# astro_post(), would produce a file named /a/b/foo_astro_post.xisf.
#
# Example: two input files named /a/b/foo.xist and /a/b/bar.xisf, processed by
# astro_post(), would produce a single output file /a/b/astro_post.xisf.
#
# Example: three input files names /a/b/M51_R.xisf and /a/b/M51_G.xisf and
# /a/b/M51_B.xisf, processed with astro_post(), would produce a single
# output file /a/b/M51_astro_post.xisf.
#
# Example: three input files names /a/b/M51_R.xisf and /a/b/M51_G.xisf and
# /a/b/M51_B.xisf, processed with astro_post(), would produce three
# output files /a/b/M51_R_astro_post.xisf, /a/b/M51_G_astro_post.xisf,
# and /a/b/M51_B_astro_post.xisf. The filters must be passed through the
# system as a new `id` flag added to the AbstractImage.

_TESTDATA_TARGETS = {
    'IC4701': [
        'master_light_IC4701_R.xisf',
        'master_light_IC4701_G.xisf',
        'master_light_IC4701_B.xisf',
        'master_light_IC4701_H.xisf',
        'master_light_IC4701_O.xisf',
        'master_light_IC4701_S.xisf',
    ]
}


# TODO: Use Path everywhere instead of str.
def copy_testdata_frames(target_name: str,
                         dest_dir: str,
                         prefix: str = '',
                         number: int | None = None) -> list[str]:
  dest_filenames = []
  for filename in _TESTDATA_TARGETS[target_name][:number]:
    path = os.path.dirname(os.path.realpath(__file__))
    src_filename = os.path.join(path, 'testdata', 'Stacks', target_name,
                                'masters', filename)
    print('### src_filename is', src_filename)
    dest_filename = os.path.join(dest_dir, prefix + filename)
    shutil.copy(src_filename, dest_filename)
    print(f'### Copied filename {dest_filename}')
    dest_filenames.append(dest_filename)

  return dest_filenames


# TODO: Desired qualities of the API that are not correct yet, and not tested:
#
# The `debug` option to the decorator will save all intermediate images, as
# declared by calling add_debug_images() inside the Tasks.
#
# If there is no explicit pix.SaveImages task, images will only be written to
# the temporary directory. What's the best way to get them out?
#
# Users can make it explicit, if they want:
#
# @pix.PixInsight()
# def _post_process_stack(image, filename):
#   image = pix.BlurXTerminator(image)
#   image = pix.AutomaticBackgroundExtractor(image)
#   pix.SaveImages(image, filename)
#
# If the function merely returns the image, without an explicit call to
# SaveImages, what do we do? When the decorator exits, the temporary directory
# will be deleted.
#
# It's also possible -- though maybe awkward -- for the function to return
# the temp directory object. So the "image" container is really several things:
#
# - The temp directory object.
# - The filename
# - The view id
# - Metadata like the filter
#
# This guarantees that the filename is valid until the user destroys the image
# object, which is nice. Evaluating the image object as a string will just
# return the filename.


# This is my best take on the API. The `deconvolved` is not used directly
# anywhere inside the function, but the tracer should see it and save it to
# disk after the function is done. End users may have a file where they can
# write (and share) many such functions, and then just associate them with
# stacks / targets / mosaics by name. These are legit Python functions, not
# fragments or other bastardizations.
@PixInsight()
def bxt_only(images: ImageList) -> ImageList:
  return BlurXTerminator(images)


@PixInsight()
def bxt_and_sxt(images: ImageList) -> ImageList:
  deconvolved = BlurXTerminator(images)
  return StarXTerminator(deconvolved)


@PixInsight()
def bxt_align_sxt(images: ImageList) -> ImageList:
  deconvolved = BlurXTerminator(images)
  aligned = StarAlignment(deconvolved)
  return StarXTerminator(aligned)


@PixInsight()
def channel_combination(images: ImageList) -> ImageList:
  aligned = StarAlignment(images)
  return ChannelCombination(aligned)


@PixInsight(timeout=10)
def stuck_task(images: ImageList) -> ImageList:
  pass


class TestDecorators(unittest.TestCase):

  def test_bxt_only(self):
    if context.get('mock_pixinsight', False):
      return

    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=1)

    images = bxt_only(filenames)

    pix_tmp_dir = images[0].tmp_dir_obj.name
    filenames = os.listdir(pix_tmp_dir)
    self.assertCountEqual(filenames, ['blurx.xisf'])

  def test_bxt_and_sxt(self):
    if context.get('mock_pixinsight', False):
      return

    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=1)

    images = bxt_and_sxt(filenames)

    pix_tmp_dir = images[0].tmp_dir_obj.name
    filenames = os.listdir(pix_tmp_dir)
    self.assertCountEqual(filenames, ['stars.xisf', 'starless.xisf'])

  def test_bxt_align_sxt(self):
    if context.get('mock_pixinsight', False):
      return

    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=3)

    images = bxt_align_sxt(filenames)

    pix_tmp_dir = images[0].tmp_dir_obj.name
    filenames = os.listdir(pix_tmp_dir)
    self.assertCountEqual(filenames, ['stars.xisf', 'starless.xisf'])

  def test_channel_combination(self):
    if context.get('mock_pixinsight', False):
      return

    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=3)

    images = channel_combination(filenames)

    pix_tmp_dir = images[0].tmp_dir_obj.name
    filenames = os.listdir(pix_tmp_dir)
    self.assertCountEqual(filenames, ['RGB.xisf'])

  def test_stuck_task(self) -> None:
    images = stuck_task([])


if __name__ == "__main__":
  unittest.main()
