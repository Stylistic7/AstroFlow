from .classes import ShellScriptTask
from .context import append_task

_javascript = """
var view_ids = {view_ids};
var output_filenames = {output_filenames};

for (let i = 0; i < output_filenames.length; ++i) {{
  console.writeln('Saving view ' + view_ids[i] + ' to filename ' + output_filenames[i]);
  let window = View.viewById(view_ids[i]).window;
  window.saveAs(output_filenames[i], false, false, true, false);
}}
"""


def SaveImages(images):
  view_ids = [i.view_id for i in images]
  filenames = [i.filename for i in images]

  append_task(
      ShellScriptTask(javascript=_javascript.format(
          view_ids=view_ids, output_filenames=filenames)))

  return images
