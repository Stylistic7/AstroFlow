from metaflow import Flow, FlowSpec, Parameter, step, JSONType, schedule
from pathlib import Path
import json
import os


class AstroFlow(FlowSpec):
  config_json = Parameter('config',
                          help='Configuration object, as JSON string',
                          type=JSONType,
                          default=json.dumps({}))

  project_dir = Parameter('project_dir',
                          help='Project Directory',
                          required=True,
                          type=str)

  download_flag = Parameter(
      'download',
      help="True to download images from ASIAIRs",
      type=bool,
      default=True,
  )

  now_flag = Parameter(
      'now',
      help="True to run all steps now (rather than after twilight)",
      type=bool,
      default=False,
  )

  C11_filter = Parameter('C11_filter',
                         help="Filter installed on the C11",
                         type=str,
                         required=True)

  @step
  def start(self):
    self.abs_project_dir = os.path.abspath(self.project_dir)
    print(f'Project directory is {self.abs_project_dir}')
    self.config = self.config_json
    self.next(self.find_files)

  @step
  def find_files(self):
    self.filenames = sorted(
        list(str(f) for f in Path(self.abs_project_dir).rglob("*.fit")))
    self.next(self.a, foreach='filenames')

  @step
  def a(self):
    print('step a input', self.input)
    self.filename = self.input

    for task in Flow('ForeachFlow').latest_successful_run["a"]:
      if not task.successful:
        continue

      if 'filename' in task.data and task.data.filename == self.filename:
        print('Returning previous run result')
        self.title = task.data.title
      else:
        self.title = '%s processed' % self.input

    self.next(self.join)

  @step
  def join(self, inputs):
    self.results = [input.title for input in inputs]
    self.next(self.end)

  @step
  def end(self):
    print('\n'.join(self.results))


if __name__ == '__main__':
  AstroFlow()