import ray
from context import cache
import pix
import os
import argparse
from multiprocessing import Pool
import structlog
import datetime

from ray_infra import ray_init
from context import context
from asiair import download_asiair, autorun_log_analysis

logger = structlog.get_logger()
log = logger

# Things that need to happen to make AstroFlow reach the "grown-up" point:
#
# - It needs to tell you what it's going to do.
# - It needs to tell you what's currently doing, and a reliable ETA.
# - It needs to tell you what it has done, so you know to look at new pics.
# - It needs to explain errors in much greater detail.
# - It needs a way to retry failed tasks.
# - Select photos and move / delete them.
# - It must not touch the user's photos, so there's no risk of bad changes.
#   - This isn't really possible; it needs to add the filter for HyperStar.
# - It needs to provide ETAs of all long-running tasks.
#
# Parts of AstroFlow that could be standalone projects:
#
# - The functional PixInsight library, pix.
# - The dependency graph and control modes stuff.
# - The mosaic merging code?
#
# Some simple metrics to pick images:
#
# - To find the best flat frame, look for the flat-corrected image with the
#   narrowest "background spike." The background must be sampled with GraXpert
#   squares. Puzzle: how can I find the best image to choose the squares?
#
# - To find the image with the least clouds, try to find the image with the
#   smallest median. This image is the "darkest." This depends upon the idea
#   clouds are always brighter than space, so not reliable.

# TODO: Give the user an option to sort mosaics / targets by date, so they
# can easily resume a previous project.

# TODO: The adding of new files, and the removing of deleted files, should be
# broken up into multiple smaller steps. If the user moves files around, we want
# to give the user the chance to point the program at the new locations.
def collect_results_as_dict(filenames, fn, prev_result, parallel=True):
    log = logger.bind(fn=fn.__name__)

    def run_fn(filenames):
        log.info(f"running fn", num_files=len(filenames))
        if parallel:
            with Pool() as p:
                results = p.map(fn, filenames)
                return {f: x for f, x in zip(filenames, results)}
        else:
            return {f: fn(f) for f in filenames if fn(f)}

    log.info(f"prev_result", num_files=len(prev_result))
    result = {k: v for k, v in prev_result.items() if os.path.exists(k)}
    num_removed_files = len(prev_result) - len(result)
    if num_removed_files:
        print(f"Removed {num_removed_files} that no longer exist.")

    added_files = set(filenames) - set(prev_result.keys())
    if len(added_files):
        print(f"Computing fn for {len(added_files)} new images...")
        result |= run_fn(added_files)
    else:
        print("no work to do")

    print(f"collect_results_as_dict done, len", len(result))
    return result


def download(save_dir):
    print("Downloading files...")
    # TODO: Put these into the config file, filled in by the web UI.
    #
    # TODO: Allow the user to pick the rig when they begin an import from an
    # ASIAIR.
    #
    # TODO: Eliminate the RIG keyword entirely. Instead, only associate images
    # into stacks when they ~the same image scale and resolution. Allow users
    # to give each combination a mnemonic name.
    #
    # TODO: Automatically apply Drizzle for under-sampled images.
    #
    # TODO: Scan network for ASIAIR devices, so the user doesn't have to set
    # static IPs.
    rigs = [
        {
            "name": "RC71",
            "ip": "192.168.50.15",
        },
        {
            "name": "C11",
            "ip": "192.168.50.16",
        },
    ]

    search_defs = [
        {
            "volume": "EMMC%20Images",
            "log_dir": "log",
            "fits_paths": ["Autorun", "Plan"],
        },
        {
            "volume": "TF%20Images",
            "log_dir": "ASIAIR/log",
            "fits_paths": ["ASIAIR/Autorun", "ASIAIR/Plan"],
        },
    ]

    # TODO: Mount the volumes, then repeatedly copy files from them, e.g. every
    # half hour. Any imaging runs that are closed (e.g. a pane has been finished)
    # are then ready to stack. Stacking and downloading must be done in parallel,
    # though.
    for rig in rigs:
        download_asiair.run(
            rig["name"], rig["ip"], search_defs, save_dir
        )


def remove_frames_during_mount_movements(light_frames):
    print("Removing frames during mount movements...")

    # TODO: implement quarantining.
    settled_light_frames = autorun_log_analysis.remove_frames_during_settling(
        light_frames
    )

    num_removed_files = len(light_frames) - len(settled_light_frames)
    print(f"Removed {num_removed_files} during settling periods.")

    return settled_light_frames


def assert_all_light_frames_in_stacks(light_frames, stacks):
    file_sets = [set(s.light_frames) for s in stacks]
    light_frames_in_stacks = set().union(*file_sets)
    # print(f'{len(light_frames)} light frames, '
    #       f'{len(light_frames_in_stacks)} light frames in stacks')
    assert set(light_frames) == light_frames_in_stacks


# def main():
#   log.info('Starting AstroFlow')

#   # TODO: This is a wart.
#   # TODO: Every Ray worker node will need its own cache for this.
#   # TODO: This doesn't start PixInsight on remote workers.
#   start_pixinsight_instances()

#   populate_context()

#   print('### Starting compute graph execution')
#   graph = compute_graph.build_compute_graph()
#   run_graph(graph)
#   print('### Compute graph is done! context is', context)

#   # TODO: Close all subordinate PixInsight instances.

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser(description='Run astrophotography workflow')

#   parser.add_argument(
#       '-i',
#       '--input-dir',
#       required=True,
#       dest='input_dirs',
#       action='append',
#       help='Directories containing raw FITS images and log files.')

#   parser.add_argument('-o',
#                       '--output-dir',
#                       required=True,
#                       dest='output_dir',
#                       action='store',
#                       help='Output directory')
#   parser.add_argument(
#       '-q',
#       '--quarantine',
#       dest='quarantine',
#       action='store_true',
#       help=('True to quarantine images that cannot be plate solved, or are '
#             'duplicates.'))
#   parser.add_argument('-d',
#                       '--download',
#                       dest='download',
#                       action='store_true',
#                       help='True to download images from ASIAIRs.')
#   parser.add_argument('-f',
#                       '--default-filter',
#                       dest='default_filter',
#                       action='store',
#                       help='Default filter')
#   parser.add_argument('-r',
#                       '--regroup',
#                       dest='regroup',
#                       action='store_true',
#                       help='True to regroup images.')

#   args = parser.parse_args()

#   # This is a hack that waits until the ContextActor is actually alive.
#   cache_keys = context.get_cache_keys()

#   context.input_dirs = [os.path.abspath(i) for i in args.input_dirs]
#   context.output_dir = os.path.abspath(args.output_dir)
#   context.quarantine = args.quarantine
#   context.download = args.download
#   context.regroup = args.regroup
#   context.default_filter = args.default_filter

#   main()
