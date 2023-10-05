from .classes import ShellScriptTask, ViewIdList
from .processes import code_for_process
from .decorators import accepts_view_ids
from .context import append_task

_javascript = """
var view_ids = {view_ids};

for (let i = 0; i < view_ids.length; ++i) {{
  let window = View.viewById(view_ids[i]).window;
  window.forceClose();
}}
"""


@accepts_view_ids
def CloseImages(view_ids, **kwargs):
  print(f'### closing images with view ids {view_ids}')
  append_task(
      ShellScriptTask(javascript=_javascript.format(view_ids=view_ids)))

  return ViewIdList([])