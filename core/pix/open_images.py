import os

from .classes import Image, ShellScriptTask, ImageList, FilenameList
from .context import append_task, add_images_to_save

from file_io.generic import get_content_hash

_javascript = """
var filenames = {filenames};
var view_ids = {view_ids};

for (let i = 0; i < filenames.length; ++i) {{
  let w = ImageWindow.open(filenames[i]);
  w[0].show();
  w[0].currentView.id = view_ids[i];
}}
"""


def OpenImages(filenames: FilenameList) -> ImageList:
  print(f'### OpenImages called with filenames {filenames} ')

  # The view ids are just the images' content hashes, with a 'V_' in front to
  # satisfy PixInsight.
  #
  # TODO: This doesn't work for FITS files. Sort out the organization of
  # the various FITS and XISF file I/O functions, and where they should
  # live, so they are accessible to both the AutoPix library and AstroFlow.
  content_hashes = [get_content_hash(f) for f in filenames]

  view_ids = ['V_' + h for h in content_hashes]

  # TODO: This doesn't work for FITS files.
  # filters = [read_xisf_header(f).get('FILTER', None) for f in filenames]
  filters = ['H' for f in filenames]

  append_task(
      ShellScriptTask(javascript=_javascript.format(filenames=filenames,
                                                    view_ids=view_ids)))

  images = [
      Image(filename=f, view_id=v, metadata={'filter': filt})
      for f, v, filt in zip(filenames, view_ids, filters)
  ]

  # add_images_to_save(images)

  print('### OpenImages returning images', images)
  return images
