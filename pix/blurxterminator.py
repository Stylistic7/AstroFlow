from .shell_script import append_task, ShellScriptTask, accepts_view_ids, ViewIdList
from .processes import code_for_process

_default_parameters = {
    'ai_file': '"BlurXTerminator.2.mlpackage"',
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
          javascript=code_for_process('BlurXTerminator', _default_parameters
                                      | kwargs, view_ids)))

  return ViewIdList(view_ids)