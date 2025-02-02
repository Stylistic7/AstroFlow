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
from dependency_graph import Graph, run_graph
from dependency_graph.astroflow_node import delete_cached_result
from metadata.image_metadata import scatter_compute_image_metadata


def merge_mosaic(args):
  # Handle the "merge mosaic" command
  print(f"Merging mosaic with ID: {args.mosaic_id}")
  if args.additional:
    print(f"Additional arguments for merge mosaic: {args.additional}")


def regroup_inputs(args, force=False):
  # Handle the "regroup inputs" command
  print("Regrouping inputs")
  if args.additional:
    print(f"Additional arguments for regroup inputs: {args.additional}")

  if force:
    delete_cached_result(build_regroup_graph)

  # TODO: Put these in the UI.
  context.input_dirs = ['/Volumes/Astro/Raw', '/Volumes/Astro2/Raw']
  context.output_dir = '/Volumes/Astro/Processed'

  graph = Graph(
      {build_regroup_graph: {(scatter_compute_image_metadata, 'input')}})

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
  mosaic_parser.add_argument("--additional",
                             help="Additional arguments for merge",
                             nargs="*")

  # 'regroup' command
  regroup_parser = subparsers.add_parser("regroup", help="Regroup data")
  regroup_subparsers = regroup_parser.add_subparsers(dest="subcommand")

  # 'regroup inputs' command
  inputs_parser = regroup_subparsers.add_parser("inputs", help="Regroup inputs")
  inputs_parser.add_argument("-f",
                             "--force",
                             action="store_true",
                             help="Force operation")
  inputs_parser.add_argument("--additional",
                             help="Additional arguments for regroup",
                             nargs="*")

  # Parse the arguments
  args = parser.parse_args()

  # Handle the parsed arguments
  if args.command == "merge" and args.subcommand == "mosaic":
    merge_mosaic(args)
  elif args.command == "regroup" and args.subcommand == "inputs":
    regroup_inputs(args, force=args.force)
  else:
    print("Unknown command or subcommand")
    parser.print_help()


if __name__ == "__main__":
  main()
