import base64

from .classes import ShellScriptTask
from .decorators import PixInsight
from .context import append_task

# run -x -a=automationMode=true
#   -a=dir=/tmp/foo -p="outputDirectory,/tmp/foo"
#   -p="keywords,WyJSSUciLCAiSU5TVFJVTUUiLCAiRklMVEVSIiwgIkZMQVRWRVJTIl0="
#   "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

_default_parameters = {
    'outputDirectory': '',
    'keywords': '["RIG", "INSTRUME", "FILTER", "FLATVERS"]'
}


@PixInsight
def WeightedBatchPreprocessing(dir, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

  arguments = {'automationMode': 'true', 'dir': dir}

  parameters['keywords'] = base64.b64encode(
      parameters['keywords'].encode()).decode()

  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters,
                      arguments=arguments))