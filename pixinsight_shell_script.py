from dataclasses import dataclass, field
import tempfile
import os
import subprocess
import time


@dataclass
class PixInsightShellScriptTask:
  javascript_filename: str
  parameters: dict = field(default_factory=dict)
  arguments: dict = field(default_factory=dict)


def _create_run_command(task):
  arguments = ' '.join(
      f'-a="{name}={value}"' for name, value in task.arguments.items())
  parameters = ' '.join(
      f'-p="{name},{value}"' for name, value in task.parameters.items())
  return f'run -x {arguments} {parameters} "{task.javascript_filename}"'


def run_script(tasks):
  with tempfile.TemporaryDirectory() as tmp_dir:
    sentinel_filename = os.path.join(tmp_dir, 'sentinel.txt')
    script_filename = os.path.join(tmp_dir, 'script.scp')

    script = '\n\n'.join(_create_run_command(t) for t in tasks)
    script += f'\n\n![echo "done" > {sentinel_filename}'

    with open(script_filename, "w") as f:
      print('script is', script)
      f.write(script)

    print(f'Running PixInsight shell script...')
    ret = subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        '-n', '--no-startup-check-updates', '--no-startup-gui-messages',
        '--no-splash', '--without-printer', f'--execute={script_filename}'
    ]).returncode

    while not os.path.exists(sentinel_filename):
      time.sleep(1)
