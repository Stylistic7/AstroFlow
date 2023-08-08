from collections import Counter


def run(image_hashes):
  # If there are any duplicated images, inform the user and fail.
  image_hashes = list(image_hashes.values())
  image_hash_counts = Counter(image_hashes)
  if not all(x == 1 for x in image_hash_counts.values()):
    for image_hash, count in image_hash_counts.items():
      if count == 1:
        continue
      print('The following files have the same image data hash:')
      filenames = [k for k, v in image_hashes.items() if v == image_hash]
      for filename in filenames:
        print(f'  {filename}')

    raise Exception('Some images were duplicated')
