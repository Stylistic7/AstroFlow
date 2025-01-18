import base64
from typing import Any

from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task

_default_parameters = {'outputDirectory': ''}


def WeightedBatchPreprocessing(unused: ImageList, **kwargs: Any) -> ImageList:
  print('### Entering WeightedBatchPreprocessing')

  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/BatchPreprocessing/WBPP.js"

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
