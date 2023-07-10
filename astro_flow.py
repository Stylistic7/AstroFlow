from metaflow import Flow, FlowSpec, Parameter, catch, step, JSONType, schedule
from pathlib import Path
import json
import os
import file_io
import itertools
from operator import itemgetter
import tempfile
import subprocess
import shutil

import config
import plate_solving
import spatial_association
import stacking
import post_processing_per_stack


def get_latest_run():
  if config.USE_ARTIFACTS_FROM_FAILED_RUNS:
    *_, run = Flow('AstroFlow').runs()
    return run
  else:
    return Flow('AstroFlow').latest_successful_run


def unique_fits_values(filenames, fits_keyword):
  return set([file_io.read_fits(f)[fits_keyword] for f in filenames])


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
    self.next(self.download)

  # TODO: Add download functionality, which may repeat until sunrise.
  @step
  def download(self):
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

    if config.MAX_FRAMES:
      self.light_frames = self.light_frames[0:config.MAX_FRAMES]
      self.flat_frames = self.flat_frames[0:config.MAX_FRAMES]

    print(f'Found {len(self.light_frames)} light frames '
          f'and {len(self.flat_frames)} flat frames')

    self.next(self.measure)

  # TODO: Add measurement.
  @step
  def measure(self):
    self.next(self.plate_solve)

  @step
  def plate_solve(self):
    # Try to find the plate solutions in previous runs of the DAG.
    try:
      prev_data = get_latest_run()['plate_solve'].task.data

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
      component_index, num_components, component = stack[0]
      group_keys.append(component_index)

    if len(stack) > 1:
      rig_index, num_rigs, rig = stack[1]
      group_keys.append(rig)

    if len(stack) > 2:
      filter_index, num_filters, filter = stack[2]
      group_keys.append(filter)

    return group_keys

  # TODO: Just pass the branches down into subtasks via the foreach_stack().
  # Then, these branches do no work, except dividing up the rigs and filters.
  @step
  def rig_grouping(self):
    self.light_frames = self.input
    self.rigs = unique_fits_values(self.light_frames, 'RIG')
    print('rig_grouping, rigs is', self.rigs)
    self.next(self.filter_grouping, foreach='rigs')

  @step
  def filter_grouping(self):
    self.filters = unique_fits_values(self.light_frames, 'FILTER')
    print('filter_grouping, filters is', self.filters)
    self.next(self.stack, foreach='filters')

  @catch(var='stack_failed')
  @step
  def stack(self):
    group_keys = self.extract_group_keys_from_stack()
    self.component, self.rig, self.filter = group_keys
    print('stack group_keys is', group_keys)

    # TODO: Maybe filter the flat frames at the same time I filter the
    # light frames.
    self.light_frames = [
        f for f in self.light_frames if file_io.fits_matches(
            f, ['IMAGETYP', 'RIG', 'FILTER'], ['Light', self.rig, self.filter])
    ]
    self.flat_frames = [
        f for f in self.flat_frames if file_io.fits_matches(
            f, ['IMAGETYP', 'RIG', 'FILTER'], ['Flat', self.rig, self.filter])
    ]

    # TODO avoid arrow code here.
    if self.light_frames:
      # Try to find a previous result with the same rig, filter, and sets of
      # light and flat frames.
      #
      # If a previous result with exactly the same inputs is found, use it.
      #
      # If a previous result with the same rig and filter, but different light
      # and/or flat frames are found, create a new version.
      # TODO: Also include changes to dark or bias frames.
      self.master = None
      self.new_stack = False
      self.version = 1
      try:
        prev_step = get_latest_run()['stack']

        found = False
        for task in prev_step:
          if task.data.rig != self.rig or task.data.filter != self.filter:
            continue

          if (self.light_frames == task.data.light_frames and
              self.flat_frames == task.data.flat_frames and
              os.path.exists(task.data.master)):
            # TODO: Also detect new dark or bias frames.
            print(
                'Light and flat frames are unchanged; re-using previous stack,',
                f'v{task.data.version}')
            self.master = task.data.master
            found = True
            break

        if not found:
          self.version = task.data.version + 1
          print(f'Lights or flats changed, or master file is missing.')
          print(f'Stacking new version v{self.version}')
      except:
        pass

      if not self.master:
        print(f'Stacking {len(self.light_frames)} light frames...')
        self.new_stack = True
        self.master = stacking.stack(self.rig, self.filter, self.light_frames,
                                     self.flat_frames, self.version)
        # TODO: masters can be more elaborate than just the list of files.
        print('self.master is', self.master)
    else:
      print(f'There are no light frames for rig '
            f'{self.rig} and filter {self.filter}')
      self.master = None

    self.next(self.post_process_per_stack)

  # TODO: Add per-rig, per-filter post-processing here. This will apply:
  # Run ABE on all images.
  # Run BXT on all images.
  # Run IntegerResample on all C11 images.
  # Run horizontal mirror on all RC71 images.
  @catch(var='post_process_per_stack_failed')
  @step
  def post_process_per_stack(self):
    # TODO: Don't rerun this if all previous steps were no-ops.
    if self.new_stack and self.master:
      print('post_process_per_stack', self.rig, self.filter, self.master)
      self.master_post = post_processing_per_stack.run_post_processing(
          self.rig, self.master)

    self.next(self.join_filter_grouping)

  @step
  def join_filter_grouping(self, inputs):
    print('join_filter_grouping, inputs are')
    for input in inputs:
      print('input', input)

    self.next(self.post_process_per_rig)

  # TODO: Add per-rig post-processing here. It will:
  # Combine LRGB channels.
  # Run ImageSolver and SPCC.
  @step
  def post_process_per_rig(self):
    self.next(self.join_rig_grouping)

  @step
  def join_rig_grouping(self, inputs):
    self.next(self.post_process_per_spatial_component)

  # TODO: Add per-spatial-component post-processing here. It will star align
  # images from multiple rigs.
  @step
  def post_process_per_spatial_component(self):
    self.next(self.join_spatial_grouping)

  @step
  def join_spatial_grouping(self, inputs):
    self.next(self.post_process_mosaics)

  # TODO: Add mosaic processing here.
  @step
  def post_process_mosaics(self):
    self.next(self.end)

  @step
  def end(self):
    print('Finished!')


if __name__ == '__main__':
  AstroFlow()