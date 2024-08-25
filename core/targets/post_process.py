import ray

import pix


# TODO: Find a way to disable these during tests.
@pix.PixInsight()
def _post_process_target(images):
  # print('### Entering _post_process_target')
  # aligned = pix.StarAlignment(images)
  # print('### pix.StarAlignment returned', aligned)
  # # TODO: Run ChannelCombination, ImageSolver, SPCC before doing
  # # StarXTerminator.
  # # TODO: ImageSolver must write the target's coordinates into the RGB image
  # # so ImageSolver won't hang.
  # result = pix.StarXTerminator(aligned)
  # print('### pix.StarXTerminator returned', result)
  # return result
  return images


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_target(target_id, filenames):
  print(f'post_process_target, target_id {target_id} filenames {filenames}')
  result = _post_process_target(filenames)
  print(f'post_process_target, target_id {target_id} returning {result}')
  return result
