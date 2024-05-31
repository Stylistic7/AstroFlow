import unittest
import tempfile
import os
import shutil

from .decorators import PixInsight
from .abstract_image import AbstractImage
from .blurxterminator import BlurXTerminator
from .starxterminator import StarXTerminator

# TODO: Replace AbstractImage with an addict.Dict. Modify OpenFiles so that
# it optionally reads metadata from the files and populates the Dicts, so that
# users can work with that metdata in Python. `content_hash` and `filter`
# should obviously be included.

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

# TODO: Make a @skip_if_file_exists decorator that does not run the underlying
# function if the output file exists. TBH this is still awkward -- it would be
# better to just not include the node in the graph at all if it doesn't need
# to run. The function itself is the only place where the output file's name
# is known. I tried adding a .needs_to_run() function to functions, which is
# ergonomic, too. Explore these ideas. This is going to get exposed to the user,
# so it really needs to be as slick and invisible as possible.
#
# For @PixInsight functions, we actually know all the filenames before ever
# sending anything to PixInsight, so we can make the decision to run (or not)
# very easily.
#
# For general @ray.remote functions, the situation is less clear, because the
# function is its own payload.
#
# Can one mechanism work everywhere? TBH the only place in AstroFlow where
# it matters is stacking.py, so maybe this is a non-issue.

_TESTDATA_TARGETS = {
    'IC4701': [
        'master_light_IC4701_B.xisf',
        'master_light_IC4701_G.xisf',
        'master_light_IC4701_H.xisf',
        'master_light_IC4701_O.xisf',
        'master_light_IC4701_R.xisf',
        'master_light_IC4701_S.xisf',
    ]
}


def copy_testdata_frames(target_name, dest_dir, prefix=''):
  dest_filenames = []
  for filename in _TESTDATA_TARGETS[target_name]:
    path = os.path.dirname(os.path.realpath(__file__))
    src_filename = os.path.join(path, 'testdata', 'Stacks', target_name,
                                'masters', filename)
    print('### src_filename is', src_filename)
    dest_filename = os.path.join(dest_dir, prefix + filename)
    shutil.copy(src_filename, dest_filename)
    print(f'### Copied filename {dest_filename}')
    dest_filenames.append(dest_filename)

  return dest_filenames


# This is my best take on the API. The `deconvolved` is not used directly
# anywhere inside the function, but the tracer should see it and save it to
# disk after the function is done. End users may have a file where they can
# write (and share) many such functions, and then just associate them with
# stacks / targets / mosaics by name. These are legit Python functions, not
# fragments or other bastardizations.
@PixInsight(dry_run=True)
def bxt_only(images):
  return BlurXTerminator(images)


@PixInsight(dry_run=True)
def bxt_and_sxt(images):
  deconvolved = BlurXTerminator(images)
  return StarXTerminator(deconvolved)


class TestDecorators(unittest.TestCase):

  def test_bxt_only(self):
    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name)
    bxt_only(filenames)

    # TODO: Test that there are now twice as many images, some with the
    #`_bxt_only` suffix.

  def test_bxt_and_sxt(self):
    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name)
    bxt_and_sxt(filenames)


if __name__ == "__main__":
  unittest.main()
