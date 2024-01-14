import ray

from cache import cache


@ray.remote(resources={'diskcache': 0.001})
class GlobalActor:
  def __init__(self):
    self.image_hashes = cache.get('image_hashes', {})
    self.fits_headers = cache.get('fits_headers', {})
    self.plate_solutions = cache.get('plate_solutions', {})

    print(f'### Read {len(self.image_hashes)} image hashes from the cache')

  def get_image_hashes(self):
    return self.image_hashes

  def set_image_hashes(self, image_hashes):
    self.image_hashes = image_hashes
    print(f'### Writing {len(image_hashes)} image hashes to the cache')
    cache['image_hashes'] = image_hashes

  def get_fits_headers(self):
    return self.fits_headers

  def set_fits_headers(self, fits_headers):
    self.fits_headers = fits_headers
    cache['fits_headers'] = fits_headers

  def get_plate_solutions(self):
    return self.plate_solutions

  def set_plate_solutions(self, plate_solutions):
    self.plate_solutions = plate_solutions
    cache['plate_solutions'] = plate_solutions


global_registry = GlobalActor.remote()


@ray.remote
def print_globals():
  image_hashes = ray.get(global_registry.get_image_hashes.remote())
  fits_headers = ray.get(global_registry.get_fits_headers.remote())
  plate_solutions = ray.get(global_registry.get_plate_solutions.remote())

  return (
      f'Global registry contains {len(image_hashes)} image hashes, '
      f'{len(fits_headers)} FITS headers, and {len(plate_solutions)} plate solutions'
  )