import base64

from .classes import ShellScriptTask
from .decorators import PixInsight
from .context import append_task

_default_parameters = {'outputDirectory': ''}


def WeightedBatchPreprocessing(unused, *args, **kwargs):
  print('### Entering WeightedBatchPreprocessing')

  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

  arguments = {
      'automationMode': 'true',
      'dir': kwargs['dir'],
      'smartNamingOverride': 'true',
      'keywords': 'RUN',
      'platesolveFallbackManual': 'false',
  }

  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters,
                      arguments=arguments))

  print('### Exiting WeightedBatchPreprocessing')
  return unused
