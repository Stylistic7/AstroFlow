from .classes import ShellScriptTask, ViewIdList
from .decorators import accepts_view_ids
from .context import append_task

_javascript = """
var view_ids = {view_ids};

for (let i = 0; i < view_ids.length; ++i) {{
  let view = View.viewById(view_ids[i]);
  if (!view) {{
    console.writeln("Couldn't find view with id " + view_ids[i]);
    continue;
  }}

  if (view.window) {{
    view.window.forceClose();
  }} else {{
    console.writeln("View id " + view_ids[i] + " didn't have a window");
  }}
}}
"""


@accepts_view_ids
def CloseImages(view_ids, **kwargs):
  print(f'### closing images with view ids {view_ids}')
  append_task(
      ShellScriptTask(javascript=_javascript.format(view_ids=view_ids)))

  return ViewIdList([])