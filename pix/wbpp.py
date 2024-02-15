import base64

from .classes import ShellScriptTask
from .decorators import PixInsight
from .context import append_task

_default_parameters = {'outputDirectory': '', 'keywords': '["RUN"]'}


@PixInsight
def WeightedBatchPreprocessing(dir, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

  arguments = {
      'automationMode': 'true',
      'dir': dir,
      'smartNamingOverride': 'true'
  }

  parameters['keywords'] = base64.b64encode(
      parameters['keywords'].encode()).decode()

  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters,
                      arguments=arguments))
