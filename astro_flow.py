from metaflow import Flow, FlowSpec, Parameter, step, JSONType, schedule
from pathlib import Path
import json
import os
import file_io

import plate_solving
import spatial_association


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
    # TODO: Make sure that adding new dark, bias, or flat frames invalidates
    # stacks that depend on them.
    filenames = list(str(f) for f in Path(self.abs_project_dir).rglob("*.fit"))

    self.light_frames = sorted(
        [f for f in filenames if file_io.fits_matches(f, 'IMAGETYP', 'Light')])
    self.dark_frames = sorted(
        [f for f in filenames if file_io.fits_matches(f, 'IMAGETYP', 'Dark')])
    self.bias_frames = sorted(
        [f for f in filenames if file_io.fits_matches(f, 'IMAGETYP', 'Bias')])
    self.flat_frames = sorted(
        [f for f in filenames if file_io.fits_matches(f, 'IMAGETYP', 'Flat')])

    self.next(self.plate_solve)

  @step
  def plate_solve(self):
    # Try to find the plate solutions in previous runs of the DAG.
    try:
      prev_data = Flow(
          'AstroFlow').latest_successful_run['plate_solve'].task.data

      if self.light_frames == prev_data.light_frames:
        print(
            'Light frame list is unchanged; re-using previous plate solutions')
        self.plate_solutions = prev_data.plate_solutions
      else:
        # TODO also handle removed files
        added_files = list(
            set(self.light_frames) - set(prev_data.light_frames))
        print(f'Plate solving {len(added_files)} added light frames...')
        self.plate_solutions = (prev_data.plate_solutions
                                | plate_solving.solve_fields(added_files))
    except:
      print(f'Plate solving {len(self.light_frames)} light frames...')
      self.plate_solutions = plate_solving.solve_fields(self.light_frames)

    self.next(self.spatial_association)

  @step
  def spatial_association(self):
    positions = [
        spatial_association.get_position(wcs)
        for wcs, _ in self.plate_solutions.values()
    ]
    components = spatial_association.find_components(positions)
    print('Found components', components)

    self.next(self.end)

  @step
  def end(self):
    print('Finished!')


if __name__ == '__main__':
  AstroFlow()