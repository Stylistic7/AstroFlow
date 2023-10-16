from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
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


# TODO: None should be a valid return type.
@accepts_view_ids
def RenameMostRecentView(view_ids, **kwargs):

  return ViewIdList(view_ids)


@accepts_view_ids
def ChannelCombination(view_ids, output_view_id=None, **kwargs):
  # TODO: This is horrifically ugly.
  parameters = _default_parameters | kwargs
  channels = [f'[true, "{v}"]' for v in view_ids]
  parameters['channels'] = enum(f'[{", ".join(channels)}]')

  append_task(
      ShellScriptTask(
          javascript=execute_global('ChannelCombination', parameters)))

  if output_view_id:
    append_task(
        ShellScriptTask(javascript=_rename_javascript.format(
            view_id=output_view_id)))

  # TODO: ChannelComnbination just creates a new image with a "random" view id.
  # Find some way to either specify the view id, or grab the "most recent" view
  # id from PixInsight.
  return ViewIdList(["RGB"])