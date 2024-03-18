import os

from .classes import ShellScriptTask, ViewIdList
from .decorators import accepts_filenames
from .context import append_task

_javascript = """
    var filenames = {filenames};
    var view_ids = {view_ids};

    for (let i = 0; i < filenames.length; ++i) {{
      let w = ImageWindow.open(filenames[i]);
      w[0].show();
      w[0].currentView.id = view_ids[i];
    }}
    """


@accepts_filenames
def OpenImages(filenames, view_ids):
  print(f'### OpenImages called with filenames {filenames}')
  assert all(
      os.path.exists(f)
      for f in filenames), f"Some files do not exist, filenames {filenames}"

  append_task(
      ShellScriptTask(javascript=_javascript.format(filenames=filenames,
                                                    view_ids=view_ids)))

  return ViewIdList(view_ids)
