import base64

import pixinsight_shell_script

# run -x -a=automationMode=true
#   -a=dir=/tmp/foo -p="outputDirectory,/tmp/foo"
#   -p="keywords,WyJSSUciLCAiSU5TVFJVTUUiLCAiRklMVEVSIiwgIkZMQVRWRVJTIl0="
#   "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

_default_parameters = {
    'outputDirectory': '',
    'keywords': '["RIG", "INSTRUME", "FILTER", "FLATVERS"]'
}


def run(dir, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js"

  arguments = {'automationMode': 'true', 'dir': dir}

  parameters['keywords'] = base64.b64encode(
      parameters['keywords'].encode()).decode()

  task = pixinsight_shell_script.PixInsightShellScriptTask(
      javascript_filename, parameters, arguments)

  pixinsight_shell_script.run_script([task])
