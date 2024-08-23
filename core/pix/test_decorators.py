import unittest
import tempfile
import os
import shutil

from .decorators import PixInsight
from .blurxterminator import BlurXTerminator
from .starxterminator import StarXTerminator
from .star_alignment import StarAlignment

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


def copy_testdata_frames(target_name, dest_dir, prefix='', number=None):
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


@PixInsight(dry_run=True)
def bxt_sxt_align(images):
  deconvolved = BlurXTerminator(images)
  aligned = StarAlignment(deconvolved)
  return StarXTerminator(aligned)


class TestDecorators(unittest.TestCase):

  def test_bxt_only(self):
    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=1)

    debug_images = bxt_only(filenames)
    print('### debug_images', debug_images)

    # TODO: Test that there are now twice as many images, some with the
    #`_bxt_only` suffix.

    # TODO: Maybe the best way to test these functions is to mock PixInsight.
    # All of the stuff that computes content hashes and stores them in the
    # XISF files is in the decorator. The pix functions can also have a
    # debug option in the context, which will cause them to call
    # a function like context.add_debug_image() at all important debugging
    # points. I can probably test everything that matters by just inspecting
    # those

  def test_bxt_and_sxt(self):
    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=1)

    bxt_and_sxt(filenames)

    # TODO: Check the debug_images for correctness.
    print(f'### temp_dir contains:')
    filenames = os.listdir(temp_dir.name)
    for filename in filenames:
      print(f'### filename {filename}')

  def test_bxt_sxt_align(self):
    temp_dir = tempfile.TemporaryDirectory()
    filenames = copy_testdata_frames('IC4701', temp_dir.name, number=3)

    bxt_sxt_align(filenames)

    # TODO: Check the debug_images for correctness.
    print(f'### temp_dir contains:')
    filenames = os.listdir(temp_dir.name)
    for filename in filenames:
      print(f'### filename {filename}')


if __name__ == "__main__":
  unittest.main()
