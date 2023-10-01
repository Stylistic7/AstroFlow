from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import code_for_process

_default_parameters = {
    'tolerance': 1.000,
    'deviation': 0.800,
    'unbalance': 1.800,
    'minBoxFraction': 0.050,
    'maxBackground': 1.0000,
    'minBackground': 0.0000,
    'useBrightnessLimits': False,
    'polyDegree': 4,
    'boxSize': 5,
    'boxSeparation': 5,
    'modelImageSampleFormat': 'AutomaticBackgroundExtractor.prototype.f32',
    'abeDownsample': 2.00,
    'writeSampleBoxes': False,
    'justTrySamples': False,
    'targetCorrection': 'AutomaticBackgroundExtractor.prototype.Subtract',
    'normalize': False,
    'discardModel': True,
    'replaceTarget': True,
    'correctedImageId': '""',
    'correctedImageSampleFormat':
    'AutomaticBackgroundExtractor.prototype.SameAsTarget',
    'verboseCoefficients': False,
    'compareModel': False,
    'compareFactor': 10.00,
}


@accepts_view_ids
def AutomaticBackgroundExtractor(view_ids, **kwargs):
  append_task(
      ShellScriptTask(javascript=code_for_process(
          'AutomaticBackgroundExtractor', _default_parameters
          | kwargs, view_ids)))

  return ViewIdList(view_ids)