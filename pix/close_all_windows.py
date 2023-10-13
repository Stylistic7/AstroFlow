from .classes import ShellScriptTask, FileList
from .decorators import accepts_view_ids
from .context import append_task

_javascript = """
var windows = ImageWindow.windows;
for (var i = 0; i < windows.length; i++) {{
  windows[i].forceClose();
}}
"""


# TODO: None should be a valid return type.
@accepts_view_ids
def CloseAllWindows(inputs, **kwargs):
  append_task(ShellScriptTask(javascript=_javascript))
  return FileList([])