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
from collections import Counter

import config
import plate_solving
import spatial_association
import stacking
import post_processing_per_stack
import post_processing_per_spatial_component
import guide_log_analysis
import autorun_log_analysis
import subframe_selector
import metaflow_data_helper
import file_locking


def get_latest_run():
  if config.USE_ARTIFACTS_FROM_FAILED_RUNS:
    sorted_runs = list(sorted(Flow('AstroFlow').runs(), key=lambda r: r.id))
    print('sorted_runs', sorted_runs)
    return sorted_runs[-2]
  else:
    return Flow('AstroFlow').latest_successful_run


def unique_fits_values(filenames, fits_keyword):
  return sorted(
      list(set([file_io.read_fits(f)[fits_keyword] for f in filenames])))


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
    file_locking.remove_lockfile(self.abs_project_dir, 'PixInsight')
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

    if config.MAX_FRAMES:
      self.light_frames = self.light_frames[0:config.MAX_FRAMES]

    print(f'Found {len(self.light_frames)} light frames')

    self.next(self.check_duplicates)

  def process_and_merge_added_keys(self, step_name, input_names, input_values,
                                   current_keys, compute_fn, output_name):
    # Find the most recent successful run of the step. If the input arguments
    # match exactly, just use the previous result.
    task = metaflow_data_helper.find_most_recent_successful_task(
        step_name, input_names, input_values)
    if task:
      print(f'Re-using previous artifact for step {step_name}')
      return task[output_name].data
    else:
      # No successful task was found with the same input arguments, so look for
      # the most recent successful task with any input arguments.
      task = metaflow_data_helper.find_most_recent_successful_task(step_name)
      if task:
        added_keys = list(
            set(current_keys) - set(task[output_name].data.keys()))
        print(f'Processing {len(added_keys)} added keys.')
        return task[output_name].data | compute_fn(added_keys)
      else:
        print(f'Processing {len(current_keys)} keys from scratch.')
        return compute_fn(current_keys)

  # @catch(var='check_duplicates_failed')
  @step
  def check_duplicates(self):
    def compute_fn(light_frames):
      return {f: file_io.hash_of_image_data(f) for f in light_frames}

    self.image_hashes = self.process_and_merge_added_keys(
        'check_duplicates', ['light_frames'], [self.light_frames],
        self.light_frames, compute_fn, 'image_hashes')

    # Remove any image files that have been deleted.
    # TODO: This is a bit ugly.
    self.image_hashes = {
        k: v
        for k, v in self.image_hashes.items() if os.path.exists(k)
    }

    # If there are any duplicated images, inform the user and fail.
    image_hashes = list(self.image_hashes.values())
    self.image_hash_counts = Counter(image_hashes)
    if not all(x == 1 for x in self.image_hash_counts.values()):
      for image_hash, count in self.image_hash_counts.items():
        if count == 1:
          continue
        print('The following files have the same image data hash:')
        filenames = [
            k for k, v in self.image_hashes.items() if v == image_hash
        ]
        for filename in filenames:
          print(f'  {filename}')

      raise Exception('Some images were duplicated')

    self.next(self.remove_frames_during_mount_movements)

  @step
  def remove_frames_during_mount_movements(self):
    filtered_light_frames = autorun_log_analysis.remove_frames_during_settling(
        self.light_frames)
    print(f'Removed {len(self.light_frames) - len(filtered_light_frames)} '
          f'light frames during settling periods.')
    self.light_frames = filtered_light_frames
    self.next(self.measure)

  @step
  def measure(self):
    self.rigs = unique_fits_values(self.light_frames, 'RIG')
    print('rig_grouping, rigs is', self.rigs)
    self.next(self.measure_for_rig, foreach='rigs')

  # TODO: Add measurement.
  @step
  def measure_for_rig(self):
    self.rig = self.input

    # TODO: Put this into a config struct.
    if self.rig == 'C11':
      image_scale = 1.38
    elif self.rig == 'C11_F7':
      image_scale = 0.38
    elif self.rig == 'RC71':
      image_scale = 2.22

    def compute_fn(light_frames):
      return subframe_selector.measure_frames(light_frames, image_scale)

    self.light_frames = [
        f for f in self.light_frames
        if file_io.fits_matches(f, 'RIG', self.rig)
    ]

    self.measurements = self.process_and_merge_added_keys(
        'measure_for_rig', ['light_frames'], [self.light_frames],
        self.light_frames, compute_fn, 'measurements')

    self.next(self.join_measurements)

  @step
  def join_measurements(self, inputs):
    self.merge_artifacts(inputs,
                         exclude=['measurements', 'rig', 'light_frames'])

    self.measurements = {
        k: v
        for d in inputs for k, v in d.measurements.items()
    }
    print(f'Joined measurements for a total of '
          f'{len(self.measurements)} light frames')

    # TODO: remove images that could not be measured.
    self.light_frames = list(sorted(self.measurements.keys()))
    self.next(self.plate_solve)

  @step
  def plate_solve(self):
    self.plate_solutions = self.process_and_merge_added_keys(
        'plate_solve', ['light_frames'], [self.light_frames],
        self.light_frames, plate_solving.solve_fields, 'plate_solutions')

    # Remove any image files that have been deleted.
    # TODO: This is a bit ugly.
    self.plate_solutions = {
        k: v
        for k, v in self.plate_solutions.items() if os.path.exists(k)
    }

    self.next(self.spatial_grouping)

  @step
  def spatial_grouping(self):
    # TODO: This should only group the light frames actually selected, even
    # if there are more plate solutions in the artifact database.
    #
    # TODO: All of the spatial positions should be printed out, along with their
    # common directory. I need to make sure the tolerances (0.1 degree) are
    # actually reasonable, and won't ever get confused by dither.
    positions = [
        spatial_association.get_position(wcs)
        for wcs, _ in self.plate_solutions.values()
    ]
    self.spatial_groups = spatial_association.find_components(
        self.plate_solutions.keys(), positions)

    for position_string, g in self.spatial_groups:
      print(f'Spatial group {position_string} has {len(g)} light frames')

    # This association is from filename to component, e.g. ('file.fits', 2).
    self.next(self.rig_grouping, foreach='spatial_groups')

  # TODO: Just pass the branches down into subtasks via the foreach_stack().
  # Then, these branches do no work, except dividing up the rigs and filters.
  @step
  def rig_grouping(self):
    self.position_string, self.light_frames = self.input
    self.rigs = unique_fits_values(self.light_frames, 'RIG')
    print('rig_grouping, rigs is', self.rigs)
    self.next(self.filter_grouping, foreach='rigs')

  @step
  def filter_grouping(self):
    self.rig = self.input
    self.filters = unique_fits_values(self.light_frames, 'FILTER')
    print('filter_grouping, filters is', self.filters)
    self.next(self.stack, foreach='filters')

  # TODO: The stacked (and post-processed) filenames need to include some id
  # that is unique for the spatial location. Otherwise, the files for each
  # panel just get overwritten.
  @catch(var='stack_failed')
  @step
  def stack(self):
    self.filter = self.input
    print(f'Top of stack(), canonical position is '
          f'{self.position_string}, rig is {self.rig}, '
          f'filter is {self.filter}')

    self.light_frames = [
        f for f in self.light_frames if file_io.fits_matches(
            f, ['IMAGETYP', 'RIG', 'FILTER'], ['Light', self.rig, self.filter])
    ]

    self.master = None
    self.new_stack = False

    # TODO avoid arrow code here.
    if len(self.light_frames) > 5:
      light_frames_path = os.path.commonpath(self.light_frames)
      print(f'Looking for flat frames in common path {light_frames_path}')
      filenames = list(str(f) for f in Path(light_frames_path).rglob("*.fit"))
      self.flat_frames = sorted([
          f for f in filenames
          if file_io.fits_matches(f, ['IMAGETYP', 'RIG', 'FILTER'],
                                  ['Flat', self.rig, self.filter])
      ])

      # Try to find a previous result with the same rig, filter, and sets of
      # light and flat frames.
      #
      # If a previous result with exactly the same inputs is found, use it.
      #
      # If a previous result with the same rig and filter, but different light
      # and/or flat frames are found, create a new version.
      self.version = 1

      # TODO: Also detect new dark or bias frames.
      task = metaflow_data_helper.find_most_recent_successful_task(
          'stack',
          ['position_string', 'rig', 'filter', 'light_frames', 'flat_frames'],
          [
              self.position_string, self.rig, self.filter, self.light_frames,
              self.flat_frames
          ])

      if (task and 'master' in task.data and task['master'].data and
          os.path.exists(task.data.master)):
        print('Light and flat frames are unchanged; re-using previous stack,',
              f'v{task.data.version}')
        self.new_stack = False
        self.master = task.data.master
      else:
        print(f'Stacking {len(self.light_frames)} light frames for '
              f'canonical position {self.position_string}, rig '
              f'{self.rig} and filter {self.filter}...')
        self.new_stack = True
        self.master = file_locking.serialize(
            self.abs_project_dir, 'PixInsight', stacking.stack,
            self.position_string, self.rig, self.filter, self.light_frames,
            self.flat_frames, self.version)

        # TODO: masters can be more elaborate than just the list of files.
        print('self.master is', self.master)
    else:
      print(f'There are too few light frames for rig '
            f'{self.rig} and filter {self.filter}')
      self.new_stack = False
      self.master = None

    self.next(self.post_process_per_stack)

  @catch(var='post_process_per_stack_failed')
  @step
  def post_process_per_stack(self):
    # TODO: Don't rerun this if all previous steps were no-ops.
    # This looks ugly, maybe it's a pattern I can factor out.
    self.post_per_stack = None
    self.output_changed = False

    if self.master:
      previous_results = (
          metaflow_data_helper.find_most_recent_successful_task_output(
              'post_process_per_stack', ['rig', 'filter', 'master'],
              [self.rig, self.filter, self.master], ['post_per_stack']))
      if previous_results:
        if all(os.path.exists(f) for f in previous_results[0]):
          self.post_per_stack = previous_results[0]

      if self.new_stack or not self.post_per_stack:
        print('Post-processing stacked master', self.rig, self.filter,
              self.master)
        self.output_changed = True
        self.post_per_stack = file_locking.serialize(
            self.abs_project_dir, 'PixInsight',
            post_processing_per_stack.run_post_processing, self.rig,
            self.master)

    self.next(self.join_filter_grouping)

  @step
  def join_filter_grouping(self, inputs):
    self.output_changed = any(i.output_changed for i in inputs)

    self.rigs = [i.rig for i in inputs]
    self.filters = [i.filter for i in inputs]
    self.posts_per_stack = [i.post_per_stack for i in inputs]

    print(f'join_filter_grouping, '
          f'output_changed {self.output_changed}, '
          f'rigs {self.rigs}, '
          f'filters {self.filters}, '
          f'posts_per_stack {self.posts_per_stack}')

    self.next(self.post_process_per_rig)

  @step
  def post_process_per_rig(self):
    self.next(self.join_rig_grouping)

  @step
  def join_rig_grouping(self, inputs):
    self.output_changed = any(i.output_changed for i in inputs)

    self.rigs = sum([i.rigs for i in inputs], [])
    self.filters = sum([i.filters for i in inputs], [])
    self.posts_per_stack = sum([i.posts_per_stack for i in inputs], [])

    print(f'join_rig_grouping, '
          f'output_changed {self.output_changed}, '
          f'rigs {self.rigs}, '
          f'filters {self.filters}, '
          f'posts_per_stack {self.posts_per_stack}')

    self.next(self.post_process_per_spatial_component)

  # TODO: Add per-spatial-component post-processing here. It will:
  # - star align images from multiple rigs
  # - combine LRGB channels
  # - plate solve
  # - spectro-photometric color calibration
  @step
  def post_process_per_spatial_component(self):
    print(f'post_process_per_spatial_component, '
          f'output_changed {self.output_changed}, '
          f'rigs {self.rigs}, '
          f'filters {self.filters}, '
          f'posts_per_stack {self.posts_per_stack}')

    if self.output_changed:
      self.aligned_masters = (
          post_processing_per_spatial_component.run_post_processing(
              self.rigs, self.filters, self.posts_per_stack))

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
  # FIXME don't hardcode this name.
  if os.path.exists("PixInsight.lock"):
    os.remove("PixInsight.lock")

  AstroFlow()