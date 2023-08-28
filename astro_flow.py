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
from attrdict import AttrDict
import functools
from copy import copy

# TODO: This file should only include step*.py files, never any lower granularity.
import config
from structs import Stack
import stacking
import guide_log_analysis
import subframe_selector
import metaflow_data_helper
from serialization import serialize
import file_io
import camera_configs
import dark_and_bias_association

from debug_decorator import debug
from compact_repr import compact_repr

import step_download_asiair
import step_find_files
import step_compute_image_hashes
import step_check_duplicates
import step_remove_frames_during_mount_movements
import step_plate_solve
import step_spatial_grouping
import step_measure
import step_flat_frame_association
import step_sanity_check_stack
import step_stack
import step_post_process_stack
import step_post_process_target


def _merge_lists(*args):
  return list(itertools.chain.from_iterable(*args))


def new_stack(stack, **kwargs):
  stack = copy(stack)
  for k, v in kwargs.items():
    stack.__setattr__(k, v)
  return stack


class AstroFlow(FlowSpec):
  config_json = Parameter('config',
                          help='Configuration object, as JSON string.',
                          type=JSONType,
                          default=json.dumps({}))

  input_dir_in = Parameter(
      'input_dir',
      help='Directory containing raw FITS images and log files.',
      required=True,
      type=str)

  quarantine_flag = Parameter(
      'quarantine',
      help="True to quarantine images that cannot be plate solved, or were "
      "taken during settling periods.",
      type=bool,
      default=False,
  )

  output_dir_in = Parameter('output_dir',
                            help='Directory containing processed images',
                            required=True,
                            type=str)

  download_flag = Parameter(
      'download',
      help="True to download images from ASIAIRs.",
      type=bool,
      default=True,
  )

  now_flag = Parameter(
      'now',
      help="True to run all steps now (rather than after twilight).",
      type=bool,
      default=False,
  )

  default_filter = Parameter('default_filter',
                             help="Default filter",
                             type=str,
                             default=None)

  debug_flag = Parameter(
      'debug',
      help="True to print debugging information.",
      type=bool,
      default=False,
  )

  def step_with_debugging(func):
    return step(debug(func))

  @step
  def start(self):
    self.config = self.config_json

    self.input_dir = os.path.abspath(self.input_dir_in)
    print(f"Input directory is {self.input_dir}")
    self.output_dir = os.path.abspath(self.output_dir_in)
    print(f"Output directory is {self.output_dir}")

    if self.quarantine_flag:
      self.bad_file_dir = os.path.join(self.input_dir, 'Bad')
    else:
      self.bad_file_dir = None

    self.next(self.download)

  # TODO: Add download functionality, which may repeat until sunrise.
  @step
  def download(self):
    # TODO: Put these into the config file.
    self.rigs = [
        {
            'name': 'RC71',
            'ip': '192.168.50.15',
        },
        {
            'name': 'C11',
            'ip': '192.168.50.16',
        },
    ]

    self.search_defs = [{
        'volume': 'EMMC%20Images',
        'log_dir': 'log',
        'fits_paths': ['Autorun', 'Plan']
    }, {
        'volume': 'TF%20Images',
        'log_dir': 'ASIAIR/log',
        'fits_paths': ['ASIAIR/Autorun', 'ASIAIR/Plan']
    }]

    self.next(self.download_from_rig, foreach='rigs')

  @step_with_debugging
  def download_from_rig(self):
    # Foreach stack: [rig]
    self.rig = self.input
    if self.download_flag:
      print(
          f'Download from rig {self.rig} default_filter {self.default_filter}')
      step_download_asiair.run(self.rig['name'], self.rig['ip'],
                               self.search_defs, self.input_dir,
                               self.default_filter)
    self.next(self.join_downloads)

  @step
  def join_downloads(self, inputs):
    # Foreach stack: [rig]
    self.merge_artifacts(inputs, exclude=['rig'])
    self.next(self.find_files)

  @step
  def find_files(self):
    self.light_frames, self.dark_frames, self.bias_frames, self.flat_frames = (
        step_find_files.run(self.input_dir))
    print(f'Found {len(self.light_frames)} light frames')
    print(f'Found {len(self.dark_frames)} dark frames')
    print(f'Found {len(self.bias_frames)} bias frames')
    print(f'Found {len(self.flat_frames)} flat frames')
    self.next(self.compute_image_hashes)

  @step
  def compute_image_hashes(self):
    prev_result = metaflow_data_helper.find('compute_image_hashes',
                                            'image_hashes') or {}
    self.image_hashes = step_compute_image_hashes.run(prev_result,
                                                      self.light_frames)
    self.next(self.check_duplicates)

  @catch(var='check_duplicates_failed')
  @step
  def check_duplicates(self):
    step_check_duplicates.run(self.image_hashes)
    self.next(self.remove_frames_during_mount_movements)

  @step
  def remove_frames_during_mount_movements(self):
    # TODO: This currently only removes frames during dithers. Should we also
    # remove frames during slews?
    self.light_frames = step_remove_frames_during_mount_movements.run(
        self.light_frames, self.bad_file_dir)
    self.next(self.plate_solve)

  # TODO: Put plate-solving here, before measurement. Let the user pick which targets
  # to process next.
  @step
  def plate_solve(self):
    prev_result = metaflow_data_helper.find('plate_solve',
                                            'plate_solutions') or {}
    self.plate_solutions = step_plate_solve.run(prev_result, self.light_frames,
                                                self.bad_file_dir)
    self.next(self.split_by_target)

  @step
  def split_by_target(self):
    targets_and_light_frames = step_spatial_grouping.run(self.plate_solutions)
    for i, x in enumerate(targets_and_light_frames):
      print(f'targets_and_light_frames[{i}] = {compact_repr(x)}')

    # FIXME hack
    # TODO: Put real target selection here.
    targets_and_light_frames = [
        x for x in targets_and_light_frames if x[0].startswith('Filamentary')
    ]

    self.stacks = [
        Stack(target=target, light_frames=list(sorted(light_frames)))
        for target, light_frames in targets_and_light_frames
    ]

    self.next(self.split_by_rig, foreach='stacks')

  @step_with_debugging
  def split_by_rig(self):
    # Foreach stack: [target]
    self.stacks = [
        new_stack(self.input, rig=rig, light_frames=list(sorted(light_frames)))
        for rig, light_frames in file_io.group_by_fits_keyword(
            self.input.light_frames, 'RIG')
    ]

    self.next(self.split_by_filter, foreach='stacks')

  @step_with_debugging
  def split_by_filter(self):
    # Foreach stack: [target, rig]
    self.stacks = [
        new_stack(self.input,
                  filter=filter,
                  light_frames=list(sorted(light_frames)))
        for filter, light_frames in file_io.group_by_fits_keyword(
            self.input.light_frames, 'FILTER')
    ]
    self.next(self.measure, foreach='stacks')

  @step_with_debugging
  @serialize('PixInsight')
  def measure(self):
    # Foreach stack: [target, rig, filter]
    self.stack = self.input

    # TODO: Maybe cleaner to just read this from the disk cache.
    prev_result = metaflow_data_helper.find(
        'measure_for_rig', 'measurements', ['target', 'rig'],
        [self.stack.target, self.stack.rig]) or {}

    # TODO: Put this in a config file.
    image_scales = {'C11': 1.38, 'C11_F7': 0.38, 'RC71': 2.22}
    self.measurements = step_measure.run(
        prev_result, image_scales[self.stack.rig], self.stack.light_frames)

    # FIXME eliminate light frames that failed measurement.
    self.next(self.get_camera_configs)

  @step_with_debugging
  def get_camera_configs(self):
    print('self.stack', self.stack)
    self.stack.camera_configs = camera_configs.get_unique_camera_configs(
        self.stack.light_frames)
    print('camera_configs', self.stack.camera_configs)
    self.next(self.bias_frame_association)

  @step_with_debugging
  def bias_frame_association(self):
    # Foreach stack: [target, rig, filter]
    print(f'Trying to associate {len(self.bias_frames)} bias frames')
    self.stack.bias_frames = dark_and_bias_association.find_bias_frames(
        self.bias_frames, self.stack.camera_configs)
    print(f'Found {len(self.stack.bias_frames)} bias frames')
    self.next(self.dark_frame_association)

  @step_with_debugging
  def dark_frame_association(self):
    # Foreach stack: [target, rig, filter]
    print(f'Trying to associate {len(self.dark_frames)} dark frames')
    self.stack.dark_frames = dark_and_bias_association.find_dark_frames(
        self.dark_frames, self.stack.camera_configs)
    print(f'Found {len(self.stack.dark_frames)} dark frames')
    self.next(self.flat_frame_association)

  @step_with_debugging
  def flat_frame_association(self):
    self.stack.flat_frames = step_flat_frame_association.filter_flat_frames(
        self.flat_frames, self.stack.rig, self.stack.filter,
        self.stack.camera_configs)
    self.stack.flat_frame_associations = step_flat_frame_association.run(
        self.stack.light_frames, self.flat_frames)
    print(f'associations {self.stack.flat_frame_associations}')

    self.next(self.sanity_check_stack)

  @step_with_debugging
  def sanity_check_stack(self):
    # FIXME just pass the Stack object to step_sanity_check_stack.run()
    self.stack_is_good = step_sanity_check_stack.run(self.stack,
                                                     self.plate_solutions)
    if self.stack_is_good:
      print('Stack passes sanity checks!')

    self.next(self.stack_subframes)

  @step_with_debugging
  @serialize('PixInsight')
  def stack_subframes(self):
    # Foreach stack: [target, rig, filter]
    # TODO: Pull this logic into step_stack.py
    # TODO: Don't pass flat frames with FLATVERS that are not used by any of
    # the light frames.
    # TODO: Remove the version=1 hack.
    # FIXME just pass the Stack object to step_stack.run()
    if self.stack_is_good:
      self.stack.master = step_stack.run(
          self.stack.target, self.stack.rig, self.stack.filter,
          self.stack.light_frames, self.stack.dark_frames,
          self.stack.bias_frames, self.stack.flat_frames, 1, self.output_dir)
    else:
      self.stack.master = None

    self.next(self.post_process_stack)

  @catch(var='post_process_stack_failed')
  @step_with_debugging
  @serialize('PixInsight')
  def post_process_stack(self):
    # Foreach stack: [target, rig, filter]
    if self.stack.master:
      self.stack.post_processed = step_post_process_stack.run(
          self.stack.rig, self.stack.master)
    else:
      self.stack.post_processed = None

    self.next(self.join_filters)

  @step_with_debugging
  def join_filters(self, inputs):
    self.next(self.join_rigs)

  @step_with_debugging
  def join_rigs(self, inputs):
    self.next(self.join_targets)

  @step_with_debugging
  def join_targets(self, inputs):
    self.next(self.end)

  # @step_with_debugging
  # def join_measure_for_rig(self, inputs):
  #   # Foreach stack: [target, rig]
  #   # for i, input in enumerate(inputs):
  #   #   for j, x in enumerate(input.stack):
  #   #     print(f'input {i} stack[{j}] = {repr(x)}')

  #   self.merge_artifacts(inputs,
  #                        exclude=['rig', 'measurements', 'light_frames'])
  #   self.measurements = {
  #       k: v
  #       for d in inputs for k, v in d.measurements.items()
  #   }
  #   self.stack.light_frames = list(self.measurements.keys())
  #   self.next(self.process_target)

  # @step_with_debugging
  # def process_target(self):
  #   # Foreach stack: [target]
  #   # print(
  #   #     f"Target '{self.stack.target}' has {len(self.stack.light_frames)} light frames"
  #   # )
  #   # rigs_and_light_frames = file_io.group_by_fits_keyword(
  #   #     self.stack.light_frames, 'RIG')

  #   # FIXME hack
  #   # TODO: Put real rig selection here.
  #   # rigs_and_light_frames = [
  #   #     x for x in rigs_and_light_frames if x[0] == 'C11'
  #   # ]

  #   self.next(self.process_target_rig, foreach='rigs_and_light_frames')

  # @step_with_debugging
  # def process_target_rig(self):
  #   # Foreach stack: [target, rig]
  #   self.rig, self.stack.light_frames = self.input
  #   print(f"Target '{self.target}' rig {self.rig} has "
  #         f"{len(self.stack.light_frames)} light frames")
  #   self.filters_and_light_frames = file_io.group_by_fits_keyword(
  #       self.stack.light_frames, 'FILTER')

  #   # TODO: pick specific filters to process here.
  #   # self.filters_and_light_frames = [
  #   #     x for x in self.filters_and_light_frames if x[0] == 'H'
  #   # ]

  #   self.next(self.process_target_rig_filter,
  #             foreach='filters_and_light_frames')

  # @step_with_debugging
  # def process_target_rig_filter(self):
  #   # Foreach stack: [target, rig, filter]
  #   # TODO: There's probably some neat way to put the foreach_stack in here as the
  #   # input to the find() function.
  #   self.filter, self.stack.light_frames = self.input
  #   self.stack.light_frames = list(sorted(self.stack.light_frames))
  #   self.camera_configs = camera_configs.get_unique_camera_configs(
  #       self.stack.light_frames)
  #   self.next(self.bias_frame_association)

  # @step_with_debugging
  # def join_process_target_rig_filter(self, inputs):
  #   # Foreach stack: [target, rig]
  #   self.targets = [input.target for input in inputs if input.master]
  #   self.rigs = [input.rig for input in inputs if input.master]
  #   self.filters = [input.filter for input in inputs if input.master]
  #   self.masters = [input.master for input in inputs if input.master]
  #   self.post_processed = [
  #       input.post_processed for input in inputs if input.master
  #   ]

  #   print('self.targets', self.targets)
  #   print('self.rigs', self.rigs)
  #   print('self.filters', self.filters)
  #   print('self.masters', self.masters)
  #   print('self.post_processed', self.post_processed)

  #   self.next(self.join_process_target_rig)

  # @step_with_debugging
  # def join_process_target_rig(self, inputs):
  #   # Foreach stack: [target]
  #   self.targets = _merge_lists([input.targets for input in inputs])
  #   self.rigs = _merge_lists([input.rigs for input in inputs])
  #   self.filters = _merge_lists([input.filters for input in inputs])
  #   self.masters = _merge_lists([input.masters for input in inputs])
  #   self.post_processed = _merge_lists(
  #       [input.post_processed for input in inputs])

  #   print('self.targets', self.targets)
  #   print('self.rigs', self.rigs)
  #   print('self.filters', self.filters)
  #   print('self.masters', self.masters)
  #   print('self.post_processed', self.post_processed)

  #   self.next(self.join_process_target)

  # @step
  # def join_process_target(self, inputs):
  #   # Foreach stack: []
  #   # TODO: implement per-target combining here (e.g. Star Alignment).
  #   self.targets = _merge_lists([input.targets for input in inputs])
  #   self.rigs = _merge_lists([input.rigs for input in inputs])
  #   self.filters = _merge_lists([input.filters for input in inputs])
  #   self.masters = _merge_lists([input.masters for input in inputs])
  #   self.post_processed = _merge_lists(
  #       [input.post_processed for input in inputs])

  #   print('self.targets', self.targets)
  #   print('self.rigs', self.rigs)
  #   print('self.filters', self.filters)
  #   print('self.masters', self.masters)
  #   print('self.post_processed', self.post_processed)

  #   self.post_processed_for_target = step_post_process_target.run(
  #       self.rigs, self.filters, self.masters)

  #   self.next(self.identify_mosiacs)

  # @step
  # def identify_mosiacs(self):
  #   # Identify images that belong to mosaics
  #   # Rough idea is to find stacks that overlap with an overlap area that
  #   # has a large aspect ratio.
  #   #
  #   self.mosaics = []
  #   self.next(self.order_mosaic_tiles, foreach='mosaics')

  # @step_with_debugging
  # def order_mosaic_tiles(self):
  #   # Order them by rows, and then columns (or vice versa).
  #   # The output here is a list of pairs of (reference, target) that,
  #   # when executed in order, will produce the final desired mosaic.
  #   self.tile_pairs = []
  #   self.next(self.merge_mosaic_tile_pairs, foreach='tile_pairs')

  # @step_with_debugging
  # def merge_mosaic_tile_pairs(self):
  #   # Run ImageSolver, MosaicByCoordinates on each image
  #   # Create an automatic trim function, like TrimMosaicTile.
  #   #   Write FITS headers the same way.
  #   # Run PhotometricMosaic on each pair, growing the mosaic
  #   self.next(self.join_mosaic_tile_pairs)

  # @step_with_debugging
  # def join_mosaic_tile_pairs(self, inputs):
  #   self.next(self.post_process_mosaic)

  # @step_with_debugging
  # def post_process_mosaic(self):
  #   # Run StarAlignment on all the color channels in the mosaic.
  #   # Run StarXTerminator on all color channels.
  #   # Combine RGB or LRGB, run SPCC.
  #   # Is this exactly the same logic as in post_process_stack?
  #   self.next(self.join_mosaics)

  # @step_with_debugging
  # def join_mosaics(self, inputs):
  #   self.next(self.end)

  # # TODO: Add per-spatial-component post-processing here. It will:
  # # - star align images from multiple rigs
  # # - combine LRGB channels
  # # - plate solve
  # # - spectro-photometric color calibration
  # @step
  # @serialize('PixInsight')
  # def post_process_per_spatial_component(self):
  #   # Eliminate inputs where there was no stack.
  #   # TODO: This is awkward.
  #   rigs = []
  #   filters = []
  #   posts_per_stack = []
  #   for rig, filter, post in zip(self.rigs, self.filters,
  #                                self.posts_per_stack):
  #     if post == None:
  #       continue
  #     rigs.append(rig)
  #     filters.append(filter)
  #     posts_per_stack.append(post)

  #   self.rigs = rigs
  #   self.filters = filters
  #   self.posts_per_stack = posts_per_stack

  #   print(f'post_process_per_spatial_component, '
  #         f'output_changed {self.output_changed}, '
  #         f'rigs {self.rigs}, '
  #         f'filters {self.filters}, '
  #         f'posts_per_stack {self.posts_per_stack}')

  #   # TODO: The pattern below is fairly common. Factor it out.
  #   self.output_changed = False
  #   self.aligned_masters = None

  #   if self.posts_per_stack:
  #     previous_results = (
  #         metaflow_data_helper.find_most_recent_successful_task_output(
  #             'post_process_per_spatial_component',
  #             ['rigs', 'filters', 'posts_per_stack'],
  #             [self.rigs, self.filters, self.posts_per_stack],
  #             ['aligned_masters']))
  #     if previous_results:
  #       if all(os.path.exists(f) for f in previous_results[0]):
  #         self.aligned_masters = previous_results[0]

  #     if self.output_changed or not self.aligned_masters:
  #       print('Post-processing spatial component', self.rigs, self.filters,
  #             self.posts_per_stack)
  #       self.output_changed = True
  #       self.aligned_masters = (
  #           post_processing_per_spatial_component.run_post_processing(
  #               self.rigs, self.filters, self.posts_per_stack))
  #     else:
  #       print('Re-using previous post-processing per spatial component result')

  #   self.next(self.join_spatial_grouping)

  # @step
  # def join_spatial_grouping(self, inputs):
  #   self.next(self.post_process_mosaics)

  # # TODO: Add mosaic processing here.
  # @step
  # def post_process_mosaics(self):
  #   self.next(self.end)

  @step
  def end(self):
    print('Finished!')


if __name__ == '__main__':
  AstroFlow()