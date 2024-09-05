from .classes import ShellScriptTask, Image
from .context import append_task
from .processes import enum, execute_global

_default_parameters = {
    'colorSpace': enum('ChannelCombination.prototype.RGB'),
    'channels': [],
    'inheritAstrometricSolution': True,
}

_rename_javascript = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{view_id}";
"""


def ChannelCombination(images, output_view_id="RGB", **kwargs):
  # TODO: This is horrifically ugly.
  view_ids = [i.view_id for i in images]
  parameters = _default_parameters | kwargs
  channels = [f'[true, "{v}"]' for v in view_ids]
  parameters['channels'] = enum(f'[{", ".join(channels)}]')

  append_task(
      ShellScriptTask(
          javascript=execute_global('ChannelCombination', parameters)))

  append_task(
      ShellScriptTask(javascript=_rename_javascript.format(
          view_id=output_view_id)))

  # TODO: ChannelComnbination just creates a new image with a "random" view id.
  # Find some way to either specify the view id, or grab the "most recent" view
  # id from PixInsight.
  return [Image(view_id=output_view_id, tag=output_view_id, dirty=True)]
