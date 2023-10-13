from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'ai_file': enum('BlurXTerminator.2.mlpackage'),
    'correct_only': False,
    'correct_first': False,
    'nonstellar_then_stellar': False,
    'lum_only': False,
    'sharpen_stars': 0.25,
    'adjust_halos': 0.00,
    'nonstellar_psf_diameter': 0.00,
    'auto_nonstellar_psf': True,
    'sharpen_nonstellar': 0.90,
}


@accepts_view_ids
def BlurXTerminator(view_ids, **kwargs):
  append_task(
      ShellScriptTask(
          javascript=execute_on_view_ids('BlurXTerminator', _default_parameters
                                         | kwargs, view_ids)))

  return ViewIdList(view_ids)