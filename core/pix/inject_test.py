from pix import shell_script
from pix.tasks import ImageSolver
from pix.tasks import BlurXTerminator
from pix.tasks import AutomaticBackgroundExtractor
from pix.decorators import PixInsight


# TODO: Remove the () if possible.
#
# TODO: images eventually needs to be a class that has accessors like:
# rgb = [images.R, images.G, images.B]
#
# TODO: This code currently literally passes the output of one function to
# the next, but we need to look at its type and potentially add types in between.
@PixInsight()
def post_process_target(images):
  images = ImageSolver(images)
  images = AutomaticBackgroundExtractor(images)
  images = BlurXTerminator(images)


if __name__ == '__main__':
  shell_script._clear_cache()

  filenames = [
      "/Users/nobody/Projects/Astro/Processed/Filamentary nebula/masters/master_light_Filamentary nebula_C11_H.xisf",
      "/Users/nobody/Projects/Astro/Processed/Filamentary nebula/masters/master_light_Filamentary nebula_RC71_O.xisf"
  ]
  post_process_target(shell_script.FileList(filenames))
