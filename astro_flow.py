from metaflow import Flow, FlowSpec, Parameter, step, JSONType, schedule
from pathlib import Path
import json
import os
import file_io
import itertools
from operator import itemgetter
import tempfile
import subprocess
import shutil

import plate_solving
import spatial_association
import stacking


def group_by_fits_header(filenames, fits_keyword):
  def read_keyword(f):
    return file_io.read_fits(f)[fits_keyword]

  sorted_input = sorted(filenames, key=read_keyword)
  groups = [(k, list(g))
            for k, g in itertools.groupby(sorted_input, key=read_keyword)]
  return sorted(groups, key=itemgetter(0, 1))


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

    self.next(self.spatial_grouping)

  # TODO: Remove light frames during dithers and slews.
  # @step
  # def remove_frames_during_mount_movements(self):
  #   pass

  @step
  def spatial_grouping(self):
    positions = [
        spatial_association.get_position(wcs)
        for wcs, _ in self.plate_solutions.values()
    ]
    self.spatial_groups = spatial_association.find_components(
        self.plate_solutions.keys(), positions)

    for i, g in enumerate(self.spatial_groups):
      print(f'Spatial group #{i + 1} has {len(g)} light frames')

    # This association is from filename to component, e.g. ('file.fits', 2).
    self.next(self.rig_grouping, foreach='spatial_groups')

  def extract_group_keys_from_stack(self):
    stack = self.foreach_stack()

    group_keys = []

    if len(stack) > 0:
      component_index, _, _ = stack[0]
      group_keys.append(f'Component {component_index}')

    if len(stack) > 1:
      rig_index, num_rigs, value = stack[1]
      rig, filenames = value
      group_keys.append(rig)

    if len(stack) > 2:
      filter_index, num_filters, value = stack[2]
      filter, filenames = value
      group_keys.append(filter)

    return group_keys

  @step
  def rig_grouping(self):
    self.rig_groups = group_by_fits_header(self.input, 'RIG')
    self.next(self.filter_grouping, foreach='rig_groups')

  @step
  def filter_grouping(self):
    rig, filenames = self.input
    self.filter_groups = group_by_fits_header(filenames, 'FILTER')
    self.next(self.stack, foreach='filter_groups')

  @step
  def stack(self):
    filter, self.light_frames = self.input
    group_keys = self.extract_group_keys_from_stack()
    _, self.rig, self.filter = group_keys

    # TODO: Maybe filter the flat frames at the same time I filter the
    # light frames.
    keys, values = ['IMAGETYP', 'RIG',
                    'FILTER'], ['Flat', self.rig, self.filter]
    self.flat_frames = [
        f for f in self.flat_frames if file_io.fits_matches(f, keys, values)
    ]

    # Try to find a previous result with the same rig, filter, and sets of
    # light and flat frames.
    #
    # If a previous result with exactly the same inputs is found, use it.
    #
    # If a previous result with the same rig and filter, but different light
    # and/or flat frames are found, create a new version.
    # TODO: Also include changes to dark or bias frames.
    self.masters = None
    self.version = 1
    try:
      prev_step = Flow('AstroFlow').latest_successful_run['stack']

      found = False
      for task in prev_step:
        if task.data.rig != self.rig or task.data.filter != self.filter:
          continue

        # print('Found task with matching rig and filter')
        if (self.light_frames == task.data.light_frames and
            self.flat_frames == task.data.flat_frames):
          # TODO: Also detect new dark or bias frames.
          print(
              'Light and flat frames are unchanged; re-using previous stack,',
              f'v{task.data.version}')
          self.masters = task.data.masters
          found = True
          break

      if not found:
        self.version = task.data.version + 1
        print(f'Lights or flats changed, stacking new version v{self.version}')
    except:
      pass

    if not self.masters:
      print(f'Stacking {len(self.light_frames)} light frames...')
      self.masters = stacking.stack(self.version, self.light_frames,
                                    self.flat_frames)

    self.next(self.join_filter_grouping)

  @step
  def join_filter_grouping(self, inputs):
    self.next(self.join_rig_grouping)

  @step
  def join_rig_grouping(self, inputs):
    self.next(self.join_spatial_grouping)

  @step
  def join_spatial_grouping(self, inputs):
    self.next(self.end)

  @step
  def end(self):
    print('Finished!')


if __name__ == '__main__':
  AstroFlow()