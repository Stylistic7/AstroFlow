from metaflow import Flow, FlowSpec, Parameter, step, JSONType, schedule
from pathlib import Path
import json
import os

import plate_solving


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
    print(f"Project directory is {self.abs_project_dir}")
    self.config = self.config_json
    self.next(self.find_files)

  @step
  def find_files(self):
    self.filenames = sorted(
        list(str(f) for f in Path(self.abs_project_dir).rglob("*.fit")))
    self.next(self.plate_solve, foreach='filenames')

  @step
  def plate_solve(self):
    self.filename = self.input

    # Try to find the plate solution in previous runs of the DAG. If no previous
    # plate solution can be found, compute it.
    self.result = None
    try:
      for task in Flow('AstroFlow').latest_successful_run['plate_solve']:
        if not task.successful:
          continue

        if 'filename' in task.data and task.data.filename == self.filename:
          self.result = task.data.result
    except:
      pass

    if not self.result:
      self.result = plate_solving.solve_field(self.filename)

    self.next(self.join)

  @step
  def join(self, inputs):
    self.results = [input.result for input in inputs]
    self.next(self.end)

  @step
  def end(self):
    num_successes = sum(map(lambda r: r[0] != None, self.results))
    print(
        f'Successfully plate-solved {num_successes} / {len(self.results)} images'
    )


if __name__ == '__main__':
  AstroFlow()