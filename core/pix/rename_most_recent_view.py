from .classes import ShellScriptTask
from .context import append_task

_javascript = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{view_id}";
"""


# TODO: None should be a valid return type.
def RenameMostRecentView(view_ids, **kwargs):
  append_task(
      ShellScriptTask(javascript=_javascript.format(view_id=view_ids[0])))

  return view_ids
