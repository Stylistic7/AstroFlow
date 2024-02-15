import base64

from .classes import ShellScriptTask
from .decorators import PixInsight
from .context import append_task

_default_parameters = {'outputDirectory': ''}


@PixInsight
def WeightedBatchPreprocessing(dir, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

  arguments = {
      'automationMode': 'true',
      'dir': dir,
      'smartNamingOverride': 'true',
      'keywords': 'RUN',
  }

  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters,
                      arguments=arguments))
