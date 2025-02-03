import argparse
import sys
import os
import ray

# TODO: This feels hacky.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

from ray_infra import ray_init
from context import context
from compute_graph import build_regroup_graph
from previews.types import QualityLevel
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, run_graph
from dependency_graph.astroflow_node import delete_cached_result
from metadata.image_metadata import load_metadata, scatter_compute_image_metadata
from customization.compute_graph import build_customize_graph


def merge_mosaic(args):
  print(f"Merging mosaic with ID: {args.mosaic_id} mode {args.mode}")
  mosaic = context.mosaics[args.mosaic_id]

  # TODO: Consider various forms of "forcing":
  # - Just deleting the toplevel graph result
  # - Ignoring all cached results
  context.ignore_cached_results = args.force
  context.dangerously_accept_existing_files = args.force

  graph = Graph({})
  for mode in ['subframe', 'master']:
    if args.mode in ['all', mode]:
      task = (build_mosaic_subgraph, QualityLevel(mode), mosaic)
      if args.force:
        delete_cached_result(task)
      graph[task] = {build_customize_graph}

  graph[build_customize_graph] = {load_metadata}

  run_graph(graph)


def regroup(args):
  print(f"Regrouping {args.regroup_type}")

  # TODO: Consider various forms of "forcing":
  # - Just deleting the toplevel graph result
  # - Ignoring all cached results
  # - Force re-solving files that previously failed to solve
  context.ignore_cached_results = args.force
  if args.force:
    delete_cached_result(build_regroup_graph)

  # TODO: Put these in the UI.
  context.input_dirs = ['/Volumes/Astro/Raw', '/Volumes/Astro2/Raw']
  context.output_dir = '/Volumes/Astro/Processed'

  graph = Graph({
      build_regroup_graph: {(scatter_compute_image_metadata, args.regroup_type)}
  })

  results, logs = run_graph(graph)


def main():
  # Create the top-level parser
  parser = argparse.ArgumentParser(prog="astroflow")
  subparsers = parser.add_subparsers(dest="command")

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

  regroup_parser = subparsers.add_parser("regroup", help="Regroup data")
  regroup_parser.add_argument(
      "regroup_type",
      choices=["input", "master", "all"],
      nargs="?",
      default="all",
      help="Type of regrouping: input, master, or all (default: all)")
  regroup_parser.add_argument("-f",
                              "--force",
                              action="store_true",
                              help="Force regrouping")

  # Parse the arguments
  args = parser.parse_args()

  # Handle the parsed arguments
  if args.command == "merge" and args.subcommand == "mosaic":
    merge_mosaic(args)
  elif args.command == "regroup":
    regroup(args)
  else:
    print("Unknown command or subcommand")
    parser.print_help()


if __name__ == "__main__":
  main()
