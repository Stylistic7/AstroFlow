import argparse
import sys
import os
import ray
import datetime

# TODO: This feels hacky.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

from ray_infra import ray_init
from context import context
from compute_graph import build_regroup_graph
from previews.types import QualityLevel
from mosaics.compute_graph import build_mosaic_subgraph, stacks_for_mosaic, targets_for_mosaic
from mosaics.post_process import post_process_mosaic
from stacks.compute_graph import build_stacking_subgraph
from dependency_graph import Graph, run_graph
from metadata.image_metadata import load_metadata, scatter_compute_image_metadata
from customization.compute_graph import build_customize_graph
from decorators import cached
from targets.post_process import post_process_target


@cached.remote
def command_done() -> None:
  pass


def regroup(args):
  print(f"### Regrouping {args.regroup_type}")

  # TODO: Consider various forms of "forcing":
  # - Just deleting the toplevel graph result
  # - Ignoring all cached results
  # - Force re-solving files that previously failed to solve
  context.ignore_cached_results = args.force
  # if args.force:
  #   delete_cached_result(build_regroup_graph)

  # TODO: Put these in the UI.
  context.input_dirs = ['/Volumes/Astro/Raw', '/Volumes/Astro2/Raw']
  context.output_dir = '/Volumes/Astro/Processed'

  graph = Graph({
      build_regroup_graph: {(scatter_compute_image_metadata, args.regroup_type)}
  })

  results, logs = run_graph(graph)


def customize(args):
  print(f'### Customizing...')
  graph = Graph({build_customize_graph: {}})
  results, logs = run_graph(graph)


def make_preview(args):
  print(f"### Making preview for {args.object_type} {args.object_id}")

  if args.object_type == 'mosaic':
    stacks = context.stacks
    targets = context.targets
    mosaic = context.mosaics[args.object_id]
    mosaic_task = (build_mosaic_subgraph, QualityLevel.SUBFRAME, mosaic,
                   stacks_for_mosaic(mosaic, stacks, targets),
                   targets_for_mosaic(mosaic, targets))
    graph = Graph({command_done: {mosaic_task}, mosaic_task: {load_metadata}})
  else:
    print(f"Unsupported object_type {args.object_type}")

  run_graph(graph)


def stack(args):
  print(f"### Stacking stack with id {args.stack_id}")
  stack = context.stacks[args.stack_id]

  context.ignore_cached_results = args.force
  context.dangerously_accept_existing_files = args.force

  run_graph(Graph({(build_stacking_subgraph, stack): {}}))


def merge_mosaic(args):
  print(f"### Merging mosaic with ID: {args.mosaic_id} mode {args.mode}")

  # TODO: Consider various forms of "forcing":
  # - Just deleting the toplevel graph result
  # - Ignoring all cached results
  context.ignore_cached_results = args.force
  context.dangerously_accept_existing_files = True

  run_graph(Graph({build_customize_graph: {load_metadata}}))

  mosaic = context.mosaics[args.mosaic_id]

  tasks = []
  for mode in ['subframe', 'master']:
    if args.mode in ['all', mode]:
      stacks = context.stacks
      targets = context.targets
      task = (build_mosaic_subgraph, QualityLevel(mode), mosaic,
              stacks_for_mosaic(mosaic, stacks,
                                targets), targets_for_mosaic(mosaic, targets))
      tasks.append(task)
      # if args.force:
      # delete_cached_result(task)

  run_graph(Graph({command_done: tasks}))


def post_process(args):
  print(f"### Post-processing {args.object_type} {args.object_id}")

  if args.object_type == 'target':
    target_task = (post_process_target, context.targets[args.object_id])
    graph = Graph({command_done: {target_task}, target_task: {load_metadata}})
  elif args.object_type == 'mosaic':
    mosaic_task = (post_process_mosaic, QualityLevel.MASTER,
                   context.mosaics[args.object_id])
    graph = Graph({command_done: {mosaic_task}, mosaic_task: {load_metadata}})
  else:
    print(f"Unsupported object_type {args.object_type}")

  run_graph(graph)


def populate_context():
  # TODO: Put these knobs in the UI.
  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160


def main():
  # Create the top-level parser
  parser = argparse.ArgumentParser(prog="astroflow")
  subparsers = parser.add_subparsers(dest="command")

  # 'regroup' command
  regroup_parser = subparsers.add_parser("regroup", help="Regroup data")
  regroup_parser.add_argument(
      "regroup_type",
      choices=["input", "master", "all"],
      nargs="?",
      default="input",
      help="Type of regrouping: input, master, or all (default: all)")
  regroup_parser.add_argument("-f",
                              "--force",
                              action="store_true",
                              help="Force regrouping")

  # 'customize' command
  customize_parser = subparsers.add_parser("customize", help="customize data")

  # 'preview' command
  preview_parser = subparsers.add_parser("preview", help="Generate previews")
  preview_parser.add_argument("object_type",
                              choices=["stack", "target", "mosaic"],
                              nargs="?",
                              help="Type of object: stack, target, or mosaic")
  preview_parser.add_argument(
      "object_id", help="ID of the object for which to generate previews")

  # 'stack' command
  stack_parser = subparsers.add_parser("stack", help="Stack images")
  stack_parser.add_argument("stack_id", help="ID of the stack to stack")
  stack_parser.add_argument("-f",
                            "--force",
                            action="store_true",
                            help="Force stacking")

  # 'merge' command
  merge_parser = subparsers.add_parser("merge", help="Merge data")
  merge_subparsers = merge_parser.add_subparsers(dest="subcommand")

  # 'merge mosaic' command
  mosaic_parser = merge_subparsers.add_parser("mosaic", help="Merge a mosaic")
  mosaic_parser.add_argument("mosaic_id", help="ID of the mosaic to merge")
  mosaic_parser.add_argument(
      "mode",
      choices=["subframe", "master", "all"],
      nargs="?",
      default="all",
      help="Type of merge: subframe, master, or all (default: all)")
  mosaic_parser.add_argument("-f",
                             "--force",
                             action="store_true",
                             help="Force merging")

  # 'post_process' command
  post_process_parser = subparsers.add_parser("post_process",
                                              help="Post-process object")
  post_process_parser.add_argument(
      "object_type",
      choices=["stack", "target", "mosaic"],
      nargs="?",
      help="Type of object: stack, target, or mosaic")
  post_process_parser.add_argument("object_id",
                                   help="ID of the object to post-processs")

  # Parse the arguments
  args = parser.parse_args()

  populate_context()

  # Handle the parsed arguments
  if args.command == "regroup":
    regroup(args)
  elif args.command == "customize":
    customize(args)
  elif args.command == 'preview':
    make_preview(args)
  elif args.command == 'stack':
    stack(args)
  elif args.command == "merge" and args.subcommand == "mosaic":
    merge_mosaic(args)
  elif args.command == "post_process":
    post_process(args)
  else:
    print("Unknown command or subcommand")
    parser.print_help()


if __name__ == "__main__":
  main()
