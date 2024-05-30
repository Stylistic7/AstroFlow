import os

from .classes import ShellScriptTask
from .context import append_task
from .abstract_image import AbstractImage

_javascript = """
var filenames = {filenames};
var view_ids = {view_ids};

for (let i = 0; i < filenames.length; ++i) {{
  let w = ImageWindow.open(filenames[i]);
  w[0].show();
  w[0].currentView.id = view_ids[i];
}}
"""


def OpenImages(filenames, view_ids):
  print(f'### OpenImages called with filenames {filenames} '
        f'and view_ids {view_ids}')
  assert len(filenames) == len(view_ids)
  assert all(
      os.path.exists(f)
      for f in filenames), f"Some files do not exist, filenames {filenames}"

  append_task(
      ShellScriptTask(javascript=_javascript.format(filenames=filenames,
                                                    view_ids=view_ids)))

  print(f'### OpenImages returning result {view_ids}')
  return [
      AbstractImage(filename=f, view_id=v) for f, v in zip(filenames, view_ids)
  ]
