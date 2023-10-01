from .shell_script import append_task, ShellScriptTask, accepts_view_ids, FileList
from .processes import code_for_process

_javascript = """
var view_ids = {view_ids};
var output_filenames = {output_filenames};

for (let i = 0; i < output_filenames.length; ++i) {{
  let window = View.viewById(view_ids[i]).window;
  window.saveAs(output_filenames[i], false, false, true, false);
  window.forceClose();
}}
"""


@accepts_view_ids
def SaveImages(view_ids, output_filenames, **kwargs):
  append_task(
      ShellScriptTask(javascript=_javascript.format(
          view_ids=view_ids, output_filenames=output_filenames)))

  return FileList(output_filenames)